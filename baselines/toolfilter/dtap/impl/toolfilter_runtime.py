from __future__ import annotations

import asyncio
import atexit
import json
import os
import sys
from typing import Any, Optional

from scripts.common.method_evidence import write_method_record

from baselines.toolfilter import core
from baselines.toolfilter.llm import ProxyChatClient


class ToolFilterIncomplete(RuntimeError):
    pass


_STATE: dict[str, Any] = {
    "latched": False,
    "allowed": None,
    "reply": None,
    "n_offered": 0,
    "n_requests": 0,
    "n_denied": 0,
    "denied_tools": {},
    "filter_error": None,
    "events": [],
    "client": None,
}

_lock = asyncio.Lock()


def _ev(kind: str, **kw: Any) -> None:
    _STATE["events"].append({"event": kind, **kw})


def _agent_model() -> str:
    model = os.environ.get("TOOLFILTER_AGENT_MODEL", "")
    if not model:
        raise ToolFilterIncomplete("TOOLFILTER_AGENT_MODEL is not set; the driver exports it")
    return model


def _client() -> ProxyChatClient:
    if _STATE["client"] is None:
        _STATE["client"] = ProxyChatClient(
            base_url=os.environ.get("TOOLFILTER_PROXY_URL", "http://127.0.0.1:8903/openrouter.ai/v1"),
            model=_agent_model(),
            temperature=0.0,
        )
    return _STATE["client"]


def _is_agent_request(kwargs: dict) -> bool:
    if kwargs.get("model") != os.environ.get("TOOLFILTER_AGENT_MODEL", ""):
        return False
    if kwargs.get("stream"):
        return False
    return bool(kwargs.get("messages")) and bool(kwargs.get("tools"))


def _jsonable(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def _run_selection_blocking(messages: list, tools: list) -> None:
    names = core.tool_names(tools)
    _STATE["n_offered"] = len(names)
    try:
        content = _client().chat(
            core.filter_request_messages(messages), tools=tools, tool_choice="none"
        )
    except Exception as exc:
        _STATE["filter_error"] = f"{type(exc).__name__}: {exc}"
        _ev("filter_error", error=_STATE["filter_error"][:300])
        raise ToolFilterIncomplete(
            "tool filter selection call failed; the task is unmeasured and must be retried: "
            f"{_STATE['filter_error']}"
        ) from exc
    if not content:
        _ev("empty_reply")
    _STATE["reply"] = content
    _STATE["allowed"] = core.select_tools(content, names)
    _ev("selection", n_offered=len(names), n_selected=len(_STATE["allowed"]),
        selected=_STATE["allowed"])


async def ensure_selection(kwargs: dict) -> None:
    if _STATE["latched"]:
        return
    async with _lock:
        if _STATE["latched"]:
            return
        if _STATE["filter_error"] is not None:
            raise ToolFilterIncomplete(
                f"tool filter selection already failed for this task: {_STATE['filter_error']}"
            )
        messages = _jsonable(kwargs.get("messages")) or []
        tools = _jsonable(kwargs.get("tools")) or []
        await asyncio.to_thread(_run_selection_blocking, messages, tools)
        _STATE["latched"] = True


def apply(kwargs: dict) -> dict:
    if _STATE["allowed"] is None:
        raise ToolFilterIncomplete("agent request reached the seam before the tool filter ran")
    kept = core.filter_tools_payload(kwargs.get("tools"), _STATE["allowed"])
    if kept:
        kwargs["tools"] = kept
    else:
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
    kwargs["messages"] = core.agent_messages_with_filter_turn(
        kwargs.get("messages") or [], _STATE["reply"] or ""
    )
    _STATE["n_requests"] += 1
    return kwargs


def allowed() -> Optional[list]:
    return _STATE["allowed"]


def check_call(tool: Optional[str]) -> Optional[str]:
    if not tool:
        return None
    if _STATE["allowed"] is None:
        _STATE["filter_error"] = f"tool call {tool!r} reached the gate before the tool filter ran"
        _ev("filter_error", error=_STATE["filter_error"])
        raise ToolFilterIncomplete(_STATE["filter_error"])
    if tool in _STATE["allowed"]:
        return None
    _STATE["n_denied"] += 1
    _STATE["denied_tools"][tool] = _STATE["denied_tools"].get(tool, 0) + 1
    _ev("denied", tool=tool)
    return core.INVALID_TOOL_TEMPLATE.format(function=tool)


def _task_id() -> Optional[str]:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--task-dir" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--task-dir="):
            return a.split("=", 1)[1]
    return None


def _dump_record() -> None:
    tid = _task_id()
    if tid is None:
        return
    rec = {
        "task_id": tid,
        "n_offered": _STATE["n_offered"],
        "n_selected": len(_STATE["allowed"]) if _STATE["allowed"] is not None else None,
        "selected": _STATE["allowed"],
        "reply": _STATE["reply"],
        "n_requests_rewritten": _STATE["n_requests"],
        "n_denied": _STATE["n_denied"],
        "denied_tools": _STATE["denied_tools"],
        "n_filter_errors": int(_STATE["filter_error"] is not None),
        "filter_error": _STATE["filter_error"],
        "events": _STATE["events"],
    }
    if _STATE["client"] is not None:
        rec["defense_llm_usage"] = {"filter": _STATE["client"].usage}
    try:
        write_method_record(tid, "toolfilter", rec)
    except Exception:
        pass


atexit.register(_dump_record)
