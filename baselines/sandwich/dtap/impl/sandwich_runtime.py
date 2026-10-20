from __future__ import annotations

import atexit
import sys
from pathlib import Path
from typing import Any, Optional

from baselines.sandwich.core import ASB_TEMPLATE, instruction_text
from scripts.common.method_evidence import write_method_record

_STATE: dict[str, Any] = {
    "task_text": None,
    "resolved": False,
    "n_results": 0,
    "n_appended": 0,
    "n_error_results": 0,
    "n_no_text_block": 0,
    "n_unknown_shape": 0,
    "n_failed_open": 0,
    "chars_appended": 0,
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


def task_text() -> Optional[str]:
    if _STATE["resolved"]:
        return _STATE["task_text"]
    _STATE["resolved"] = True
    td = task_dir()
    if not td:
        _note(event="task_unresolved", reason="no --task-dir in argv")
        return None
    cfg = Path(td) / "config.yaml"
    text = None
    try:
        from dt_arena.src.types.task import TaskConfig

        text = instruction_text(TaskConfig.from_yaml(str(cfg)).original_instruction)
    except Exception as exc:
        _note(event="task_unresolved", reason=f"{type(exc).__name__}: {exc}", config=str(cfg))
        return None
    if not text:
        _note(event="task_unresolved", reason="empty instruction", config=str(cfg))
        return None
    _STATE["task_text"] = text
    _note(event="task_resolved", chars=len(text), config=str(cfg))
    return text


def _append_to_blocks(blocks, rem: str) -> int:
    n = 0
    for i, blk in enumerate(blocks):
        if isinstance(blk, dict):
            t = blk.get("text")
            if isinstance(t, str):
                blk["text"] = t + rem
                n += 1
            continue
        t = getattr(blk, "text", None)
        if not isinstance(t, str):
            continue
        try:
            blk.text = t + rem
            n += 1
        except Exception:
            try:
                blocks[i] = blk.model_copy(update={"text": t + rem})
                n += 1
            except Exception:
                pass
    return n


def append_reminder(result, tool: Optional[str] = None):
    _STATE["n_results"] += 1
    if tool:
        _STATE["tools_seen"][tool] = _STATE["tools_seen"].get(tool, 0) + 1

    if getattr(result, "exception", None) is not None or getattr(result, "is_error", False):
        _STATE["n_error_results"] += 1
        _note(event="skip_error_result", tool=tool)
        return result

    task = task_text()
    if not task:
        _STATE["n_failed_open"] += 1
        _note(event="failed_open", tool=tool)
        return result

    rem = ASB_TEMPLATE.format(task=task)
    raw = getattr(result, "raw", None)
    n = 0
    if isinstance(raw, str):
        result.raw = raw + rem
        n = 1
    elif raw is None:
        _STATE["n_no_text_block"] += 1
        _note(event="empty_result", tool=tool)
        return result
    else:
        content = raw.get("content") if isinstance(raw, dict) else getattr(raw, "content", None)
        if isinstance(content, list):
            n = _append_to_blocks(content, rem)
        elif isinstance(content, str):
            if isinstance(raw, dict):
                raw["content"] = content + rem
            else:
                raw.content = content + rem
            n = 1
        else:
            _STATE["n_unknown_shape"] += 1
            _note(event="unknown_shape", tool=tool, shape=type(raw).__name__)
            return result

    if n:
        _STATE["n_appended"] += 1
        _STATE["chars_appended"] += len(rem) * n
        _note(event="appended", tool=tool, blocks=n, chars=len(rem))
    else:
        _STATE["n_no_text_block"] += 1
        _note(event="no_text_block", tool=tool, shape=type(raw).__name__)
    return result


def record() -> dict:
    return {
        "task_id": task_dir(),
        "task_chars": len(_STATE["task_text"] or ""),
        "n_results": _STATE["n_results"],
        "n_appended": _STATE["n_appended"],
        "n_error_results": _STATE["n_error_results"],
        "n_no_text_block": _STATE["n_no_text_block"],
        "n_unknown_shape": _STATE["n_unknown_shape"],
        "n_failed_open": _STATE["n_failed_open"],
        "chars_appended": _STATE["chars_appended"],
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
        write_method_record(tid, "sandwich", record())
    except Exception:
        pass


atexit.register(_dump_record)
