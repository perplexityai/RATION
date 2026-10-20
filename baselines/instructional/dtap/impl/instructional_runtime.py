from __future__ import annotations

import atexit
import sys
from typing import Any, List, Optional, Union

from baselines.instructional.core import ASB_TEMPLATE, is_prevented, prevent_instruction
from scripts.common.method_evidence import write_method_record

_STATE: dict[str, Any] = {
    "n_configs_seen": 0,
    "n_wrapped": 0,
    "n_turns_wrapped": 0,
    "n_already": 0,
    "n_failed_open": 0,
    "chars_before": 0,
    "chars_after": 0,
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


def _chars(x) -> int:
    if x is None:
        return 0
    if isinstance(x, (list, tuple)):
        return sum(len(str(t)) for t in x)
    return len(str(x))


def _turns(x) -> int:
    return len(x) if isinstance(x, (list, tuple)) else (1 if x else 0)


def prevent_task_instruction(
    instruction: Union[str, List[str], None],
    task_id: Optional[str] = None,
) -> Union[str, List[str], None]:
    _STATE["n_configs_seen"] += 1
    if instruction is None or not str(instruction).strip():
        _STATE["n_failed_open"] += 1
        _note(event="failed_open", reason="empty instruction", task_id=task_id)
        return instruction

    before_turns = instruction if isinstance(instruction, (list, tuple)) else [instruction]
    if all(is_prevented(str(t)) for t in before_turns if str(t).strip()):
        _STATE["n_already"] += 1
        _note(event="already_prevented", task_id=task_id)
        return instruction

    after = prevent_instruction(instruction)
    n_turns = sum(1 for a, b in zip(before_turns,
                                    after if isinstance(after, list) else [after])
                  if str(a) != str(b))
    if not n_turns:
        _STATE["n_failed_open"] += 1
        _note(event="failed_open", reason="nothing rewritten", task_id=task_id)
        return instruction

    _STATE["n_wrapped"] += 1
    _STATE["n_turns_wrapped"] += n_turns
    _STATE["chars_before"] += _chars(instruction)
    _STATE["chars_after"] += _chars(after)
    _note(event="prevented", task_id=task_id, turns=_turns(instruction),
          turns_rewritten=n_turns, chars_before=_chars(instruction), chars_after=_chars(after))
    return after


def armed() -> bool:
    return bool(_STATE["n_wrapped"])


def record() -> dict:
    return {
        "task_id": task_dir(),
        "armed": armed(),
        "n_configs_seen": _STATE["n_configs_seen"],
        "n_wrapped": _STATE["n_wrapped"],
        "n_turns_wrapped": _STATE["n_turns_wrapped"],
        "n_already": _STATE["n_already"],
        "n_failed_open": _STATE["n_failed_open"],
        "chars_before": _STATE["chars_before"],
        "chars_after": _STATE["chars_after"],
        "frame_chars": len(ASB_TEMPLATE) - len("{task}"),
        "events": _STATE["events"],
    }


def _dump_record() -> None:
    if not _STATE["n_configs_seen"]:
        return
    tid = task_dir()
    if tid is None:
        return
    try:
        write_method_record(tid, "instructional", record())
    except Exception:
        pass


atexit.register(_dump_record)
