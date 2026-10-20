from __future__ import annotations

import asyncio
import atexit
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Optional

from baselines.promptguard.agentdojo.config import REDACTION_TEXT, SAFE_LABEL, THRESHOLD
from scripts.common.method_evidence import write_method_record

_STATE: dict[str, Any] = {
    "n_results": 0,
    "n_classified": 0,
    "n_safe": 0,
    "n_injections": 0,
    "n_redacted": 0,
    "n_classifier_errors": 0,
    "n_no_text_block": 0,
    "n_unknown_shape": 0,
    "n_redaction_errors": 0,
    "n_error_results": 0,
    "n_image_blocks": 0,
    "n_over_context": 0,
    "n_segmented": 0,
    "n_segments_total": 0,
    "tools_seen": {},
    "events": [],
}
_MAX_EVENTS = 500
_LOCK = asyncio.Lock()
_TRANSPORT_RETRIES = 2
_TRANSPORT_TIMEOUT_S = 1800.0


def _note(**event: Any) -> None:
    if len(_STATE["events"]) < _MAX_EVENTS:
        _STATE["events"].append(event)


def task_dir() -> Optional[str]:
    for i, arg in enumerate(sys.argv):
        if arg == "--task-dir" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--task-dir="):
            return arg.split("=", 1)[1]
    return None


def _sidecar_url() -> str:
    return os.environ.get("PROMPTGUARD_URL", "http://127.0.0.1:8909/v1/classify")


def agentdojo_decision(label: str, score: float, threshold: float = THRESHOLD) -> tuple[bool, float]:
    safety_score = float(score) if str(label) == SAFE_LABEL else 1.0 - float(score)
    return safety_score < float(threshold), safety_score


def _post_classify(body: bytes) -> dict[str, Any]:
    req = urllib.request.Request(
        _sidecar_url(),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TRANSPORT_TIMEOUT_S) as response:
        return json.loads(response.read())


def _classify_sync(text: str) -> dict[str, Any]:
    body = json.dumps({"text": text}).encode()
    last: Exception | None = None
    for attempt in range(_TRANSPORT_RETRIES + 1):
        try:
            payload = _post_classify(body)
            break
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
            if attempt < _TRANSPORT_RETRIES:
                time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(
            f"classifier transport failed after {_TRANSPORT_RETRIES + 1} attempts: {last}"
        )
    label = payload.get("label")
    score = payload.get("score")
    if not isinstance(label, str) or not isinstance(score, (int, float)):
        raise ValueError("classifier response lacks label/score")
    if not isinstance(payload.get("n_segments"), int) or payload["n_segments"] < 1:
        raise ValueError("classifier response lacks the segment count")
    is_injection, safety_score = agentdojo_decision(label, float(score), THRESHOLD)
    payload.update({"is_injection": is_injection, "safety_score": safety_score})
    return payload


async def classify_text(text: str) -> dict[str, Any]:
    async with _LOCK:
        return await asyncio.to_thread(_classify_sync, text)


def _text_ref(block: Any) -> Optional[str]:
    value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
    return value if isinstance(value, str) else None


def _is_image(block: Any) -> bool:
    kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
    return kind in {"image", "image_url"} or hasattr(block, "data")


def _set_block_text(blocks: list[Any], index: int, value: str) -> bool:
    block = blocks[index]
    if isinstance(block, dict):
        block["text"] = value
        return True
    try:
        block.text = value
        return True
    except Exception:
        try:
            blocks[index] = block.model_copy(update={"text": value})
            return True
        except Exception:
            return False


def _shape(result: Any) -> tuple[Optional[str], Optional[Callable[[], None]], int, str]:
    raw = getattr(result, "raw", None)
    if isinstance(raw, str):
        return raw, lambda: setattr(result, "raw", REDACTION_TEXT), 0, "string"
    if raw is None:
        return None, None, 0, "empty"

    if isinstance(raw, list):
        blocks = raw
    else:
        content = raw.get("content") if isinstance(raw, dict) else getattr(raw, "content", None)
        if isinstance(content, str):
            def redact_content() -> None:
                if isinstance(raw, dict):
                    raw["content"] = REDACTION_TEXT
                else:
                    raw.content = REDACTION_TEXT
            return content, redact_content, 0, "content-string"
        if not isinstance(content, list):
            return None, None, 0, "unknown"
        blocks = content

    indices = [i for i, block in enumerate(blocks) if _text_ref(block) is not None]
    images = sum(1 for block in blocks if _is_image(block))
    if not indices:
        return None, None, images, "blocks-no-text"
    text = "\n".join(_text_ref(blocks[i]) or "" for i in indices)

    def redact_blocks() -> None:
        rewritten = [_set_block_text(blocks, i, REDACTION_TEXT) for i in indices]
        if not all(rewritten):
            raise RuntimeError("could not rewrite every text block")

    return text, redact_blocks, images, "blocks"


def _redact_fail_closed(result: Any, redact: Callable[[], None], tool: Optional[str]) -> None:
    try:
        redact()
    except Exception as exc:
        _STATE["n_redaction_errors"] += 1
        result.raw = REDACTION_TEXT
        _note(event="redaction_error_fail_closed", tool=tool, error=type(exc).__name__)


async def inspect_result(result: Any, tool: Optional[str] = None) -> Any:
    _STATE["n_results"] += 1
    if tool:
        _STATE["tools_seen"][tool] = _STATE["tools_seen"].get(tool, 0) + 1
    if getattr(result, "exception", None) is not None or getattr(result, "is_error", False):
        _STATE["n_error_results"] += 1

    text, redact, images, shape = _shape(result)
    _STATE["n_image_blocks"] += images
    if text is None or redact is None:
        if shape == "unknown":
            _STATE["n_unknown_shape"] += 1
            _note(event="unknown_shape", tool=tool, shape=type(getattr(result, "raw", None)).__name__)
            result.raw = REDACTION_TEXT
            _STATE["n_redacted"] += 1
        else:
            _STATE["n_no_text_block"] += 1
            _note(event="no_text_block", tool=tool, shape=shape, image_blocks=images)
        return result

    try:
        verdict = await classify_text(text)
        _STATE["n_classified"] += 1
        injection = bool(verdict["is_injection"])
        _STATE["n_injections" if injection else "n_safe"] += 1
        n_segments = int(verdict["n_segments"])
        _STATE["n_segments_total"] += n_segments
        if verdict.get("over_context"):
            _STATE["n_over_context"] += 1
        if n_segments > 1:
            _STATE["n_segmented"] += 1
        _note(
            event="classified",
            tool=tool,
            chars=len(text),
            label=verdict.get("label"),
            label_meaning=verdict.get("label_meaning"),
            score=verdict.get("score"),
            safety_score=verdict.get("safety_score"),
            injection=injection,
            input_tokens=verdict.get("input_tokens"),
            n_segments=n_segments,
            segment_argmax=verdict.get("segment_argmax"),
            segment_max_malicious=verdict.get("segment_max_malicious"),
            guard_truncated_segments=verdict.get("guard_truncated_segments"),
            over_context=bool(verdict.get("over_context")),
            image_blocks=images,
        )
        if injection:
            _redact_fail_closed(result, redact, tool)
            _STATE["n_redacted"] += 1
    except Exception as exc:
        _STATE["n_classifier_errors"] += 1
        _redact_fail_closed(result, redact, tool)
        _STATE["n_redacted"] += 1
        _note(
            event="classifier_error_fail_closed",
            tool=tool,
            chars=len(text),
            error=f"{type(exc).__name__}: {str(exc)[:240]}",
        )
        print(
            f"[PROMPTGUARD] classifier error for {tool}; tool text REDACTED fail-closed: "
            f"{type(exc).__name__}: {str(exc)[:240]}",
            file=sys.stderr,
            flush=True,
        )
    return result


def record() -> dict[str, Any]:
    return {
        "task_id": task_dir(),
        **{key: value for key, value in _STATE.items() if key != "events"},
        "events": _STATE["events"],
    }


def _dump_record() -> None:
    if not _STATE["n_results"]:
        return
    tid = task_dir()
    if tid is None:
        return
    try:
        write_method_record(tid, "promptguard", record())
    except Exception:
        pass


atexit.register(_dump_record)
