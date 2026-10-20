from __future__ import annotations

import asyncio
import atexit
import json
import os
import sys
from typing import Any, Optional

from baselines.melon.comparator import THETA, OpenAIEmbedder
from baselines.melon.core import MelonState
from baselines.melon.llm import ProxyChatClient
from scripts.common.method_evidence import write_method_record
from sensitive_args import REFERENCE_FOCUS


class MelonIncomplete(RuntimeError):
    pass


STOP_TEXT = "The tool outputs contain potential prompt injection, stopping execution."

_STATE: dict[str, Any] = {"melon": None, "built": False, "exchanges": [], "seq": 0,
                          "events": [], "blocked_latched": False, "incomplete": None}
_MAX_EXCHANGES = 8
_LOCK = asyncio.Lock()


def log_event(ev: dict) -> None:
    _STATE["events"].append(ev)
    if ev.get("event") in {"audit", "incomplete"}:
        print(f"[MELON] {json.dumps(ev, default=str)[:600]}", file=sys.stderr, flush=True)


def _mark_incomplete(reason: str, **detail: Any) -> MelonIncomplete:
    if _STATE["incomplete"] is None:
        _STATE["incomplete"] = reason
    log_event({"event": "incomplete", "reason": reason, **detail})
    return MelonIncomplete(f"MELON could not audit this task ({reason}); it is unmeasured and must be retried")


def incomplete() -> Optional[str]:
    return _STATE["incomplete"]


def _agent_model() -> str:
    return os.environ.get("MELON_AGENT_MODEL", "")


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def _parse_response_calls(resp: Any) -> list[dict]:
    choices = getattr(resp, "choices", None)
    if choices is None and isinstance(resp, dict):
        choices = resp.get("choices")
    if not choices:
        return []
    ch0 = choices[0]
    msg = getattr(ch0, "message", None)
    if msg is None and isinstance(ch0, dict):
        msg = ch0.get("message")
    tcs = getattr(msg, "tool_calls", None)
    if tcs is None and isinstance(msg, dict):
        tcs = msg.get("tool_calls")
    calls = []
    for tc in tcs or []:
        fn = getattr(tc, "function", None)
        if fn is None and isinstance(tc, dict):
            fn = tc.get("function")
        name = getattr(fn, "name", None)
        raw = getattr(fn, "arguments", None)
        if isinstance(fn, dict):
            name = fn.get("name")
            raw = fn.get("arguments")
        cid = getattr(tc, "id", None)
        if cid is None and isinstance(tc, dict):
            cid = tc.get("id")
        try:
            args = json.loads(raw or "{}") if isinstance(raw, str) else (raw or {})
            if not isinstance(args, dict):
                args = {"_raw_arguments": raw}
        except Exception:
            args = {"_raw_arguments": raw}
        calls.append({"id": cid, "function": name, "args": args})
    return calls


def is_agent_request(kwargs: dict) -> bool:
    return (kwargs.get("model") == _agent_model() and not kwargs.get("stream")
            and bool(kwargs.get("messages")))


def _role(message: Any) -> Optional[str]:
    return message.get("role") if isinstance(message, dict) else getattr(message, "role", None)


def maybe_capture(kwargs: dict, resp: Any) -> None:
    try:
        if not is_agent_request(kwargs):
            return
        calls = _parse_response_calls(resp)
        if not calls:
            return
        messages = _jsonable(kwargs["messages"])
        _STATE["seq"] += 1
        _STATE["exchanges"].append({
            "seq": _STATE["seq"],
            "messages": messages,
            "tools": _jsonable(kwargs.get("tools")),
            "calls": calls,
            "auditable": _role(messages[-1]) == "tool",
            "audited": False,
        })
        del _STATE["exchanges"][:-_MAX_EXCHANGES]
    except Exception:
        pass


def ensure_state() -> Optional[MelonState]:
    if _STATE["built"]:
        return _STATE["melon"]
    _STATE["built"] = True
    base = os.environ.get("MELON_PROXY_URL", "http://127.0.0.1:8903/openrouter.ai/v1")
    agent_model = _agent_model()
    if not agent_model:
        raise MelonIncomplete("MELON_AGENT_MODEL is not set; the driver exports it")
    st = MelonState(
        masked_llm=ProxyChatClient(base, agent_model, temperature=0.0),
        embed_fn=OpenAIEmbedder(),
        theta=THETA,
        focus=REFERENCE_FOCUS,
        log=log_event,
    )
    _STATE["melon"] = st
    return st


def _find_step(tool_name: str, args: dict) -> Optional[dict]:
    target = (tool_name, json.dumps(args or {}, sort_keys=True, default=str))
    for ex in reversed(_STATE["exchanges"]):
        for c in ex["calls"]:
            if (c["function"], json.dumps(c["args"] or {}, sort_keys=True, default=str)) == target:
                return ex
    return None


async def audit_call(tool_name: str, args: dict) -> Optional[dict]:
    st = ensure_state()
    if st is None:
        return None
    if _STATE["incomplete"] is not None:
        raise MelonIncomplete(f"MELON task already incomplete ({_STATE['incomplete']})")
    if _STATE["blocked_latched"]:
        return {"allow": False, "reason": "sticky-halt"}
    async with _LOCK:
        if _STATE["blocked_latched"]:
            return {"allow": False, "reason": "sticky-halt"}
        ex = _find_step(tool_name, args)
        if ex is None:
            raise _mark_incomplete("capture_miss", tool=tool_name)
        if not ex["auditable"]:
            log_event({"event": "unaudited_first_turn", "seq": ex["seq"], "tool": tool_name})
            return {"allow": True, "reason": "before-first-tool-result"}
        if ex["audited"]:
            return {"allow": True, "reason": "step-already-cleared"}
        try:
            result = await asyncio.to_thread(
                st.audit_step, ex["messages"], ex["calls"], ex.get("tools"))
        except Exception as exc:
            raise _mark_incomplete("masked_error", seq=ex["seq"],
                                   error=f"{type(exc).__name__}: {str(exc)[:300]}") from exc
        ex["audited"] = True
        if result.get("alert"):
            _STATE["blocked_latched"] = True
            return {"allow": False, "reason": "melon-detect",
                    "max_cosine": result.get("max_cosine")}
        return {"allow": True, "reason": "melon-clear",
                "max_cosine": result.get("max_cosine")}


def blocked_text() -> str:
    return STOP_TEXT


def _dump_record() -> None:
    tid = _task_id()
    if tid is None:
        return
    st = _STATE["melon"]
    events = _STATE["events"]
    rec = {
        "task_id": tid,
        "n_exchanges": _STATE["seq"],
        "blocked_latched": _STATE["blocked_latched"],
        "n_audits": sum(1 for e in events if e.get("event") == "audit"),
        "n_alerts": sum(1 for e in events if e.get("event") == "audit" and e.get("alert")),
        "n_unaudited_first_turn": sum(1 for e in events if e.get("event") == "unaudited_first_turn"),
        "n_incomplete": int(_STATE["incomplete"] is not None),
        "incomplete_reason": _STATE["incomplete"],
        "events": events,
    }
    if st is not None:
        rec["defense_llm_usage"] = {"masked_llm": st.masked_llm.usage,
                                    "embed": st.embed_fn.usage}
    try:
        write_method_record(tid, "melon", rec)
    except Exception:
        pass


def _task_id() -> Optional[str]:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--task-dir" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--task-dir="):
            return a.split("=", 1)[1]
    return None


atexit.register(_dump_record)
