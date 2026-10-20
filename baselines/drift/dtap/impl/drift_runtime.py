from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from baselines.drift.llm import ProxyLLMClient
from baselines.drift.state import DriftState
from scripts.dtap.trusted_request import trusted_request
from scripts.common.method_evidence import write_method_record


class DriftIncomplete(RuntimeError):
    pass


_STATE: dict[str, Any] = {"drift": None, "built": False, "tools": {}, "batches": [], "seq": 0,
                          "events": [], "incomplete": None}
_MAX_BATCHES = 8
_LOCK: Optional[asyncio.Lock] = None
_PRIVILEGE_CACHE = Path(__file__).resolve().parents[4] / "runs" / "dtap-drift-privcache"


def lock() -> asyncio.Lock:
    global _LOCK
    if _LOCK is None:
        _LOCK = asyncio.Lock()
    return _LOCK


def _agent_model() -> str:
    return os.environ.get("DRIFT_AGENT_MODEL", "")


def task_id() -> Optional[str]:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--task-dir" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--task-dir="):
            return a.split("=", 1)[1]
    return None


def log_event(ev: dict) -> None:
    _STATE["events"].append(ev)
    if ev.get("event") in {"refusal", "redaction", "plan", "deviation", "incomplete"}:
        print(f"[DRIFT] {json.dumps(ev, default=str)[:800]}", file=sys.stderr, flush=True)


def incomplete() -> Optional[str]:
    return _STATE["incomplete"]


def _mark_incomplete(reason: str, **detail: Any) -> DriftIncomplete:
    if _STATE["incomplete"] is None:
        _STATE["incomplete"] = reason
    log_event({"event": "incomplete", "reason": reason, **detail})
    return DriftIncomplete(f"DRIFT could not run for this task ({reason}); it is unmeasured and must be retried")


def register_tools(server: Optional[str], tools: Any) -> None:
    try:
        for t in tools or []:
            schema = getattr(t, "inputSchema", None)
            if hasattr(schema, "model_dump"):
                schema = schema.model_dump()
            name = getattr(t, "name", None)
            if not name:
                continue
            _STATE["tools"][name] = {"name": name, "description": getattr(t, "description", "") or "",
                                     "parameters": schema}
    except Exception:
        pass


def tools_registered() -> bool:
    return bool(_STATE["tools"])


def _cache_key(spec_json: str) -> str:
    return hashlib.sha1(f"{_agent_model()}\n{spec_json}".encode()).hexdigest()


def _cache_get(spec_json: str) -> Optional[str]:
    f = _PRIVILEGE_CACHE / (_cache_key(spec_json) + ".txt")
    try:
        return f.read_text().strip() if f.is_file() else None
    except Exception:
        return None


def _cache_put(spec_json: str, perm: str) -> None:
    try:
        _PRIVILEGE_CACHE.mkdir(parents=True, exist_ok=True)
        (_PRIVILEGE_CACHE / (_cache_key(spec_json) + ".txt")).write_text(perm)
    except Exception:
        pass


def ensure_state(force: bool = False) -> Optional[DriftState]:
    if _STATE["built"]:
        return _STATE["drift"]
    if not tools_registered() and not force:
        return None
    _STATE["built"] = True
    tid = task_id()
    query = trusted_request(tid) if tid else None
    if not query:
        raise _mark_incomplete("no_trusted_request")
    if not _agent_model():
        raise _mark_incomplete("agent_model_unset")
    st = DriftState(query, ProxyLLMClient(os.environ.get("DRIFT_PROXY_URL", "http://127.0.0.1:8903/openrouter.ai/v1"),
                                         _agent_model()), log=log_event)
    spec = list(_STATE["tools"].values())
    if not spec:
        log_event({"event": "empty_tool_listing_at_plan"})
    try:
        st.build_plan(spec)
        st.assign_privileges(cache_get=_cache_get, cache_put=_cache_put)
    except Exception as exc:
        raise _mark_incomplete("planner_error", error=f"{type(exc).__name__}: {str(exc)[:300]}") from exc
    _STATE["drift"] = st
    return st


def get_state() -> Optional[DriftState]:
    return _STATE["drift"]


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


def capture_batch(kwargs: dict, resp: Any) -> None:
    try:
        if not is_agent_request(kwargs):
            return
        calls = _parse_response_calls(resp)
        if not calls:
            return
        _STATE["seq"] += 1
        _STATE["batches"].append({"seq": _STATE["seq"], "calls": calls, "verdict": None})
        del _STATE["batches"][:-_MAX_BATCHES]
    except Exception:
        pass


def _same_args(a: Any, b: Any) -> bool:
    return json.dumps(_jsonable(a or {}), sort_keys=True) == json.dumps(_jsonable(b or {}), sort_keys=True)


def _find_batch(tool: str, args: dict) -> Optional[tuple[dict, int]]:
    for batch in reversed(_STATE["batches"]):
        for index, call in enumerate(batch["calls"]):
            if call["function"] == tool and _same_args(call["args"], args):
                return batch, index
    return None


async def audit_call(tool: str, args: dict) -> Optional[dict]:
    async with lock():
        if _STATE["incomplete"] is not None:
            raise DriftIncomplete(f"DRIFT task already incomplete ({_STATE['incomplete']})")
        st = await asyncio.to_thread(ensure_state, True)
        if st is None:
            return None
        found = _find_batch(tool, args)
        if found is None:
            raise _mark_incomplete("capture_miss", tool=tool)
        batch, index = found
        if batch["verdict"] is None:
            try:
                batch["verdict"] = await asyncio.to_thread(st.validate_batch, batch["calls"])
            except Exception as exc:
                raise _mark_incomplete("validator_error", seq=batch["seq"],
                                       error=f"{type(exc).__name__}: {str(exc)[:300]}") from exc
        verdict = batch["verdict"]
        disposition = verdict.dispositions[index]
        text = verdict.refusal_text if disposition == "refused" else ("" if disposition == "dropped" else None)
        return {"disposition": disposition, "text": text, "kind": verdict.kind}


async def isolate_result(tool: str, args: dict, text: str) -> Optional[str]:
    async with lock():
        if _STATE["incomplete"] is not None:
            raise DriftIncomplete(f"DRIFT task already incomplete ({_STATE['incomplete']})")
        st = get_state()
        found = _find_batch(tool, args)
        if st is None or found is None:
            raise _mark_incomplete("capture_miss", tool=tool, phase="post_tool_call")
        batch, index = found
        if index != len(batch["calls"]) - 1:
            return None
        call_id = batch["calls"][index].get("id")
        try:
            cleaned, _detected = await asyncio.to_thread(st.isolate, text, tool, args, call_id)
        except Exception as exc:
            raise _mark_incomplete("detector_error", seq=batch["seq"],
                                   error=f"{type(exc).__name__}: {str(exc)[:300]}") from exc
        return cleaned


def _dump_record() -> None:
    tid = task_id()
    if tid is None:
        return
    st = _STATE["drift"]
    events = _STATE["events"]
    rec = {
        "task_id": tid,
        "n_batches": _STATE["seq"],
        "n_refusals": sum(1 for e in events if e.get("event") == "refusal"),
        "n_redactions": sum(1 for e in events if e.get("event") == "redaction"),
        "n_deviations": sum(1 for e in events if e.get("event") == "deviation"),
        "n_judge_fail_open": sum(1 for e in events if e.get("event") == "judge_fail_open"),
        "n_node_check_fail_open": sum(1 for e in events if e.get("event") == "node_check_fail_open"),
        "n_incomplete": int(_STATE["incomplete"] is not None),
        "incomplete_reason": _STATE["incomplete"],
        "events": events,
    }
    if st is not None:
        rec.update({
            "initial_trajectory": st.initial_function_trajectory,
            "final_trajectory": st.function_trajectory,
            "achieved_trajectory": st.achieved_function_trajectory,
            "checklist": st.node_checklist[:4000] if isinstance(st.node_checklist, str) else st.node_checklist,
            "tool_permissions": st.tool_permissions,
            "defense_llm_usage": st.llm.usage,
        })
    try:
        write_method_record(tid, "drift", rec)
    except Exception:
        pass


atexit.register(_dump_record)
