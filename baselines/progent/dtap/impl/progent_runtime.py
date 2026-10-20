from __future__ import annotations

import asyncio
import atexit
import builtins
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from baselines.progent import llm
from scripts.dtap.trusted_request import trusted_request
from scripts.common.method_evidence import write_method_record


class ProgentIncomplete(RuntimeError):
    pass


REFUSAL_PREFIX = "An error occurred while running the tool. Please try again. Error: "
LOG_LINE_CAP = 8192
_SCHEMA_KEYWORDS = {"type", "properties", "required", "additionalProperties", "title",
                    "$schema", "description", "definitions", "$defs"}

_STATE: dict[str, Any] = {
    "tool": None,
    "built": False,
    "task": None,
    "tools": {},
    "batches": [],
    "seq": 0,
    "events": [],
    "incomplete": None,
    "update_ignored": 0,
}
_MAX_BATCHES = 8
_LOCK: Optional[asyncio.Lock] = None


def lock() -> asyncio.Lock:
    global _LOCK
    if _LOCK is None:
        _LOCK = asyncio.Lock()
    return _LOCK


def _agent_model() -> str:
    return os.environ.get("PROGENT_AGENT_MODEL", "")


def log_event(ev: dict) -> None:
    _STATE["events"].append(ev)
    if ev.get("event") in {"block", "policy", "policy_update", "update_ignored", "incomplete",
                           "allowlist_missing"}:
        print(f"[PROGENT] {json.dumps(ev, default=str)[:800]}", file=sys.stderr, flush=True)


def incomplete() -> Optional[str]:
    return _STATE["incomplete"]


def _mark_incomplete(reason: str, **detail: Any) -> ProgentIncomplete:
    if _STATE["incomplete"] is None:
        _STATE["incomplete"] = reason
    log_event({"event": "incomplete", "reason": reason, **detail})
    return ProgentIncomplete(f"Progent could not run for this task ({reason}); it is unmeasured and must be retried")


def task_id() -> Optional[str]:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--task-dir" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--task-dir="):
            return a.split("=", 1)[1]
    return None


def _args_from_properties(props: Any) -> dict:
    if not isinstance(props, dict):
        return {}
    return {k: v for k, v in props.items() if not (isinstance(v, dict) and "$ref" in v)}


def register_tools(server: Optional[str], tools: Any) -> None:
    try:
        for t in tools or []:
            name = getattr(t, "name", None)
            if not name:
                continue
            schema = getattr(t, "inputSchema", None)
            if hasattr(schema, "model_dump"):
                schema = schema.model_dump()
            props = (schema or {}).get("properties") if isinstance(schema, dict) else None
            _STATE["tools"][name] = {"name": name,
                                     "description": getattr(t, "description", "") or "",
                                     "args": _args_from_properties(props)}
    except Exception:
        pass


def register_local_tool(name: str, description: Optional[str], params_json_schema: Any) -> None:
    props = (params_json_schema or {}).get("properties") if isinstance(params_json_schema, dict) else None
    _STATE["tools"][name] = {"name": name, "description": description or "",
                             "args": _args_from_properties(props)}


def tools_registered() -> bool:
    return bool(_STATE["tools"])


def authored_allowlist() -> tuple[list[str], dict[str, Any]]:
    path = Path(__file__).resolve().parent / "always_allowed.json"
    domain = os.environ.get("PROGENT_DOMAIN", "")
    try:
        book = json.loads(path.read_text())
    except Exception as exc:
        log_event({"event": "allowlist_missing", "path": str(path), "error": repr(exc)})
        return [], {"source": "missing"}
    entry = (book.get("domains") or {}).get(domain)
    if not entry:
        log_event({"event": "allowlist_missing", "path": str(path), "domain": domain})
        return [], {"source": "missing", "domain": domain}
    names = sorted((entry.get("always_allowed") or {}).keys())
    return names, {"source": str(path), "domain": domain, "n": len(names)}


def _cap_upstream_log_lines(tool_mod: Any) -> None:
    def _print(*args: Any, **kw: Any) -> None:
        msg = kw.pop("sep", " ").join(str(a) for a in args)
        if msg.startswith("Error: ") and msg.endswith("Ignore the update error."):
            _STATE["update_ignored"] += 1
            log_event({"event": "update_ignored", "error": msg[7:-25][:300]})
        if len(msg) > LOG_LINE_CAP:
            msg = f"{msg[:LOG_LINE_CAP]}… [{len(msg) - LOG_LINE_CAP} chars omitted: DTAP reads task output with a 64 KiB line limit]"
        kw.setdefault("file", sys.stderr)
        builtins.print(msg, **kw)

    tool_mod.print = _print


def ensure_state() -> Any:
    if _STATE["built"]:
        return _STATE["tool"]
    _STATE["built"] = True
    tid = task_id()
    _STATE["task"] = tid
    query = trusted_request(tid) if tid else None
    if not query:
        raise _mark_incomplete("no_trusted_request")
    if not _agent_model():
        raise _mark_incomplete("agent_model_unset")
    proxy = os.environ.get("PROGENT_PROXY_URL", "")
    if not proxy:
        raise _mark_incomplete("proxy_url_unset")
    if not tools_registered():
        raise _mark_incomplete("empty_tool_listing")
    llm.configure(_agent_model(), proxy, allow_late=True)
    llm.install_usage_tally()
    import secagent.tool as tool

    _cap_upstream_log_lines(tool)
    _STATE["tool"] = tool
    if (tool.policy_model, tool.ignore_update_error, tool.generate_policy, tool.enable_secagent) != (_agent_model(), True, True, True):
        raise _mark_incomplete("upstream_env_misread", policy_model=tool.policy_model,
                               ignore_update_error=tool.ignore_update_error)
    tool.reset_security_policy(include_human_policy=True)
    listing = list(_STATE["tools"].values())
    tool.update_available_tools(listing)
    allow, ameta = authored_allowlist()
    if allow:
        tool.update_always_allowed_tools(allow, allow_all_no_arg_tools=False)

    prev = llm.set_role("policy_gen")
    try:
        tool.generate_security_policy(query, manual_check=False)
    except Exception as exc:
        raise _mark_incomplete("policy_error", error=f"{type(exc).__name__}: {str(exc)[:300]}") from exc
    finally:
        llm.set_role(prev)
    generated = tool.get_generated_policy()
    log_event({"event": "policy", "allowlist": ameta,
               "tools_in_listing": len(listing), "generated": generated,
               **restriction_counts(generated, _STATE["tools"]), "usage": llm.usage()})
    return tool


def restriction_counts(generated: list, tools: dict) -> dict[str, int]:
    n_args = n_inert = n_whole = 0
    for entry in generated or []:
        args = entry.get("args") if isinstance(entry, dict) else None
        if not isinstance(args, dict) or not args:
            continue
        real = set((tools.get(entry.get("name")) or {}).get("args") or {})
        for key in args:
            n_args += 1
            if key not in real:
                n_inert += 1
        if set(args) <= _SCHEMA_KEYWORDS and isinstance(args.get("properties"), dict):
            n_whole += 1
    return {"n_arg_restrictions": n_args, "n_inert_restrictions": n_inert,
            "n_whole_object_entries": n_whole}


def get_state() -> Any:
    return _STATE["tool"]


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
        try:
            args = json.loads(raw or "{}") if isinstance(raw, str) else (raw or {})
            if not isinstance(args, dict):
                args = {"_raw_arguments": raw}
        except Exception:
            args = {"_raw_arguments": raw}
        calls.append({"function": name, "args": args})
    return calls


def is_agent_request(kwargs: dict) -> bool:
    return (kwargs.get("model") == _agent_model() and not kwargs.get("stream")
            and bool(kwargs.get("messages")))


async def before_agent_request() -> None:
    async with lock():
        if _STATE["incomplete"] is not None:
            raise ProgentIncomplete(f"Progent task is incomplete ({_STATE['incomplete']}); aborting the agent run so the task is retried, never scored")
        if not _STATE["built"]:
            await asyncio.to_thread(ensure_state)


def capture_batch(kwargs: dict, resp: Any) -> None:
    try:
        if not is_agent_request(kwargs):
            return
        calls = _parse_response_calls(resp)
        if not calls:
            return
        _STATE["seq"] += 1
        _STATE["batches"].append({"seq": _STATE["seq"], "calls": calls,
                                  "results": [None] * len(calls), "updated": False})
        del _STATE["batches"][:-_MAX_BATCHES]
    except Exception:
        pass


def _same_args(a: Any, b: Any) -> bool:
    return json.dumps(_jsonable(a or {}), sort_keys=True) == json.dumps(_jsonable(b or {}), sort_keys=True)


def _find_call(tool: str, args: dict, *, unsettled: bool) -> Optional[tuple[dict, int]]:
    for batch in reversed(_STATE["batches"]):
        for index, call in enumerate(batch["calls"]):
            if call["function"] != tool or not _same_args(call["args"], args):
                continue
            if unsettled and batch["results"][index] is not None:
                continue
            return batch, index
    return None


def _check_blocking(tool_mod: Any, tool: str, args: dict) -> Optional[str]:
    try:
        tool_mod.check_tool_call(tool, dict(args or {}))
        return None
    except tool_mod.ValidationError as exc:
        return str(exc)


async def check_call(tool: str, args: dict) -> Optional[str]:
    async with lock():
        if _STATE["incomplete"] is not None:
            raise ProgentIncomplete(f"Progent task already incomplete ({_STATE['incomplete']})")
        tool_mod = _STATE["tool"]
        if tool_mod is None:
            raise _mark_incomplete("no_agent_request_seen", tool=tool)
        found = _find_call(tool, args, unsettled=True)
        if found is None:
            raise _mark_incomplete("capture_miss", tool=tool)
        batch, index = found
        try:
            message = _check_blocking(tool_mod, tool, args)
        except Exception as exc:
            raise _mark_incomplete("check_error", tool=tool, error=f"{type(exc).__name__}: {str(exc)[:300]}") from exc
        if message is None:
            return None
        log_event({"event": "block", "tool": tool, "seq": batch["seq"],
                   "args_keys": sorted((args or {}).keys()), "reason": message[:400]})
        await _deposit(batch, index, "")
        return REFUSAL_PREFIX + message


async def settle_result(tool: str, args: dict, text: Optional[str]) -> None:
    async with lock():
        if _STATE["incomplete"] is not None:
            raise ProgentIncomplete(f"Progent task already incomplete ({_STATE['incomplete']})")
        found = _find_call(tool, args, unsettled=True)
        if found is None:
            raise _mark_incomplete("capture_miss", tool=tool, phase="post_tool_call")
        batch, index = found
        await _deposit(batch, index, text if isinstance(text, str) else "")


async def _deposit(batch: dict, index: int, text: str) -> None:
    batch["results"][index] = text
    if batch["updated"] or any(r is None for r in batch["results"]):
        return
    batch["updated"] = True
    tool_mod = _STATE["tool"]
    calls = [{"name": c["function"], "args": dict(c["args"] or {})} for c in batch["calls"]]
    before = copy.deepcopy(tool_mod.get_generated_policy())
    prev = llm.set_role("update")
    try:
        await asyncio.to_thread(tool_mod.generate_update_security_policy, calls,
                                str(list(batch["results"])), False)
    except Exception as exc:
        raise _mark_incomplete("update_decide_error", seq=batch["seq"],
                               error=f"{type(exc).__name__}: {str(exc)[:300]}") from exc
    finally:
        llm.set_role(prev)
    after = tool_mod.get_generated_policy()
    if after != before:
        log_event({"event": "policy_update", "seq": batch["seq"], "calls": [c["name"] for c in calls],
                   "before": before, "after": after})


def _dump_record() -> None:
    tid = task_id()
    if tid is None:
        return
    tool_mod = _STATE["tool"]
    events = _STATE["events"]
    generated = tool_mod.get_generated_policy() if tool_mod is not None else []
    rec = {
        "task_id": tid,
        "defense_initialized": tool_mod is not None,
        "tools_registered": len(_STATE["tools"]),
        "n_batches": _STATE["seq"],
        "n_blocks": sum(1 for e in events if e.get("event") == "block"),
        "n_policy_updates": sum(1 for e in events if e.get("event") == "policy_update"),
        "n_update_ignored": _STATE["update_ignored"],
        "n_incomplete": int(_STATE["incomplete"] is not None),
        "incomplete_reason": _STATE["incomplete"],
        **restriction_counts(generated, _STATE["tools"]),
        "final_policy": tool_mod.get_current_config() if tool_mod is not None else None,
        "defense_llm_usage": llm.usage(),
        "events": events,
    }
    try:
        write_method_record(tid, "progent", rec)
    except Exception:
        pass


atexit.register(_dump_record)
