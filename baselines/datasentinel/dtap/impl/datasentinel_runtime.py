from __future__ import annotations

import asyncio
import atexit
import sys
import time
from collections.abc import Callable
from typing import Any, Optional

from baselines.datasentinel import core
from baselines.datasentinel.core import REDACTION_TEXT
from scripts.common.method_evidence import write_method_record

_STATE: dict[str, Any] = {
    "n_results": 0,
    "n_scanned": 0,
    "n_clean": 0,
    "n_contaminated": 0,
    "n_redacted": 0,
    "n_detector_errors": 0,
    "n_no_text_block": 0,
    "n_empty_text": 0,
    "n_unknown_shape": 0,
    "n_redaction_errors": 0,
    "n_error_results": 0,
    "n_image_blocks": 0,
    "n_truncated": 0,
    "tools_seen": {},
    "events": [],
}
_MAX_EVENTS = 500
_LOCK = asyncio.Lock()
_MAX_PROMPT_TOKENS = 8192
_TRANSPORT_WINDOW_S = 900.0
_TRANSPORT_PAUSE_S = 15.0
_CALL_TIMEOUT_S = 600.0


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


_CLIENT: core.DetectorClient | None = None


def _client() -> core.DetectorClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = core.DetectorClient(timeout_s=_CALL_TIMEOUT_S, retries=0)
    return _CLIENT


def _detect_sync(text: str) -> core.Verdict:
    deadline = time.monotonic() + _TRANSPORT_WINDOW_S
    while True:
        try:
            return _client().detect(text, max_prompt_tokens=_MAX_PROMPT_TOKENS)
        except RuntimeError as exc:
            if "unreachable" not in str(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(_TRANSPORT_PAUSE_S)


async def scan_text(text: str) -> core.Verdict:
    async with _LOCK:
        return await asyncio.to_thread(_detect_sync, text)


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
    if text == "":
        _STATE["n_empty_text"] += 1
        _note(event="empty_text", tool=tool, image_blocks=images)
        return result

    try:
        verdict = await scan_text(text)
        _STATE["n_scanned"] += 1
        _STATE["n_contaminated" if verdict.contaminated else "n_clean"] += 1
        if verdict.truncated:
            _STATE["n_truncated"] += 1
        _note(
            event="scanned",
            tool=tool,
            chars=len(text),
            contaminated=verdict.contaminated,
            response=verdict.response,
            input_tokens=verdict.input_tokens,
            classified_tokens=verdict.classified_tokens,
            truncated=verdict.truncated,
            latency_s=verdict.latency_s,
            image_blocks=images,
        )
        if verdict.contaminated:
            _redact_fail_closed(result, redact, tool)
            _STATE["n_redacted"] += 1
    except Exception as exc:
        _STATE["n_detector_errors"] += 1
        _redact_fail_closed(result, redact, tool)
        _STATE["n_redacted"] += 1
        _note(
            event="detector_error_fail_closed",
            tool=tool,
            chars=len(text),
            error=f"{type(exc).__name__}: {str(exc)[:240]}",
        )
        print(
            f"[DATASENTINEL] detector error for {tool}; tool text REDACTED fail-closed: "
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
        write_method_record(tid, "datasentinel", record())
    except Exception:
        pass


atexit.register(_dump_record)
