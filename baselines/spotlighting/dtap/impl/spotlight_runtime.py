from __future__ import annotations

import atexit
import sys
from typing import Any, Optional

from baselines.spotlighting.core import (
    DELIM_CLOSE,
    DELIM_OPEN,
    SYSTEM_SUFFIX,
    is_spotlighted,
    spotlight,
    system_prompt_with_suffix,
)
from scripts.common.method_evidence import write_method_record

_STATE: dict[str, Any] = {
    "n_results": 0,
    "n_wrapped": 0,
    "n_error_results": 0,
    "n_no_text_block": 0,
    "n_unknown_shape": 0,
    "n_failed_open": 0,
    "n_prompts_seen": 0,
    "n_prompts_patched": 0,
    "n_prompts_already": 0,
    "chars_added": 0,
    "tools_seen": {},
    "events": [],
}
_MAX_EVENTS = 400


def _note(**ev) -> None:
    if len(_STATE["events"]) < _MAX_EVENTS:
        _STATE["events"].append(ev)


def task_dir() -> Optional[str]:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--task-dir" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--task-dir="):
            return a.split("=", 1)[1]
    return None


def note_system_prompt(before: str, where: str = "agent") -> str:
    _STATE["n_prompts_seen"] += 1
    if is_spotlighted(before):
        _STATE["n_prompts_already"] += 1
        _note(event="system_prompt_already_marked", where=where)
        return before
    after = system_prompt_with_suffix(before)
    _STATE["n_prompts_patched"] += 1
    _note(event="system_prompt_marked", where=where,
          chars_before=len(before or ""), chars_after=len(after))
    return after


def system_prompt_armed() -> bool:
    return bool(_STATE["n_prompts_patched"] or _STATE["n_prompts_already"])


def _text_block_indices(blocks) -> list[int]:
    out = []
    for i, blk in enumerate(blocks):
        t = blk.get("text") if isinstance(blk, dict) else getattr(blk, "text", None)
        if isinstance(t, str):
            out.append(i)
    return out


def _set_text(blocks, i, value) -> bool:
    blk = blocks[i]
    if isinstance(blk, dict):
        blk["text"] = value
        return True
    try:
        blk.text = value
        return True
    except Exception:
        try:
            blocks[i] = blk.model_copy(update={"text": value})
            return True
        except Exception:
            return False


def _wrap_blocks(blocks) -> int:
    idx = _text_block_indices(blocks)
    if not idx:
        return 0
    first, last = idx[0], idx[-1]
    if first == last:
        t = blocks[first].get("text") if isinstance(blocks[first], dict) else blocks[first].text
        return 1 if _set_text(blocks, first, spotlight(t)) else 0
    ft = blocks[first].get("text") if isinstance(blocks[first], dict) else blocks[first].text
    lt = blocks[last].get("text") if isinstance(blocks[last], dict) else blocks[last].text
    ok = _set_text(blocks, first, DELIM_OPEN + ft)
    ok = _set_text(blocks, last, lt + DELIM_CLOSE) and ok
    return 1 if ok else 0


def wrap_result(result, tool: Optional[str] = None):
    _STATE["n_results"] += 1
    if tool:
        _STATE["tools_seen"][tool] = _STATE["tools_seen"].get(tool, 0) + 1
    if getattr(result, "exception", None) is not None or getattr(result, "is_error", False):
        _STATE["n_error_results"] += 1

    raw = getattr(result, "raw", None)
    n = 0
    if isinstance(raw, str):
        result.raw = spotlight(raw)
        n = 1
    elif raw is None:
        _STATE["n_no_text_block"] += 1
        _note(event="empty_result", tool=tool)
        return result
    else:
        content = raw.get("content") if isinstance(raw, dict) else getattr(raw, "content", None)
        if isinstance(content, list):
            n = _wrap_blocks(content)
        elif isinstance(content, str):
            if isinstance(raw, dict):
                raw["content"] = spotlight(content)
            else:
                raw.content = spotlight(content)
            n = 1
        else:
            _STATE["n_unknown_shape"] += 1
            _note(event="unknown_shape", tool=tool, shape=type(raw).__name__)
            return result

    if n:
        _STATE["n_wrapped"] += 1
        _STATE["chars_added"] += len(DELIM_OPEN) + len(DELIM_CLOSE)
        _note(event="wrapped", tool=tool)
        if not system_prompt_armed():
            _STATE["n_failed_open"] += 1
            _note(event="failed_open", tool=tool, reason="system prompt never marked")
    else:
        _STATE["n_no_text_block"] += 1
        _note(event="no_text_block", tool=tool, shape=type(raw).__name__)
    return result


def record() -> dict:
    return {
        "task_id": task_dir(),
        "system_prompt_armed": system_prompt_armed(),
        "n_prompts_seen": _STATE["n_prompts_seen"],
        "n_prompts_patched": _STATE["n_prompts_patched"],
        "n_prompts_already": _STATE["n_prompts_already"],
        "n_results": _STATE["n_results"],
        "n_wrapped": _STATE["n_wrapped"],
        "n_error_results": _STATE["n_error_results"],
        "n_no_text_block": _STATE["n_no_text_block"],
        "n_unknown_shape": _STATE["n_unknown_shape"],
        "n_failed_open": _STATE["n_failed_open"],
        "chars_added": _STATE["chars_added"],
        "suffix_chars": len(SYSTEM_SUFFIX),
        "tools_seen": _STATE["tools_seen"],
        "events": _STATE["events"],
    }


def _dump_record() -> None:
    if not _STATE["n_results"]:
        return
    tid = task_dir()
    if tid is None:
        return
    try:
        write_method_record(tid, "spotlighting", record())
    except Exception:
        pass


atexit.register(_dump_record)
