from __future__ import annotations

import asyncio
import atexit
import json
import os
import sys
from typing import Any, Optional

from openai._types import NotGiven, Omit

from baselines.attriguard.core import AttriGuardState, ReplayError, call_signature
from baselines.attriguard.llm import ProxyChatClient
from baselines.attriguard.prompts import blocked_reason
from scripts.dtap.trusted_request import trusted_request
from scripts.common.method_evidence import write_method_record

from external_tools import LOCAL_EXTERNAL_TOOLS, classification_for_domain


class AttriGuardIncomplete(BaseException):
    pass


class CaptureError(RuntimeError):
    pass


class TrustedRequestError(RuntimeError):
    pass


_STATE: dict[str, Any] = {
    "ag": None,
    "built": False,
    "exchanges": [],
    "seq": 0,
    "events": [],
    "pending_skip_after_seq": None,
    "incomplete": False,
}
_MAX_EXCHANGES = 8
_LOCK: asyncio.Lock | None = None
_LOCK_LOOP: asyncio.AbstractEventLoop | None = None


def _flag(name: str, default: str = "0") -> bool:
    return (os.environ.get(name, default) or "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def log_event(ev: dict) -> None:
    _STATE["events"].append(ev)
    if ev.get("event") in {
        "survival", "shadow_replay", "audit_skipped", "all_empty_step",
        "fully_blocked_step",
        "attenuation_error", "capture_error", "capture_miss", "replay_error", "judge_error",
        "state_error", "trusted_request_error",
    }:
        brief = json.dumps(ev, default=str)[:800]
        print(f"[ATTRIGUARD] {brief}", file=sys.stderr, flush=True)


def task_id() -> Optional[str]:
    for index, arg in enumerate(sys.argv):
        if arg == "--task-dir" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if arg.startswith("--task-dir="):
            return arg.split("=", 1)[1]
    return None


def read_trusted_request() -> Optional[str]:
    tid = task_id()
    if not tid:
        return None
    try:
        return trusted_request(tid)
    except Exception as exc:
        log_event({"event": "trusted_request_error", "error": f"{type(exc).__name__}: {exc}"})
        if isinstance(exc, TrustedRequestError):
            raise
        raise TrustedRequestError(str(exc)) from exc


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"required AttriGuard setting is missing: {name}")
    return value.strip()


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def _parse_response_calls(resp: Any) -> list[dict]:
    choices = getattr(resp, "choices", None)
    if choices is None and isinstance(resp, dict):
        choices = resp.get("choices")
    if not choices:
        return []
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is None and isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    calls: list[dict] = []
    for tool_call in tool_calls or []:
        function = getattr(tool_call, "function", None)
        if function is None and isinstance(tool_call, dict):
            function = tool_call.get("function")
        name = getattr(function, "name", None)
        raw_args = getattr(function, "arguments", None)
        if isinstance(function, dict):
            name = function.get("name")
            raw_args = function.get("arguments")
        call_id = getattr(tool_call, "id", None)
        if call_id is None and isinstance(tool_call, dict):
            call_id = tool_call.get("id")
        args = parse_tool_arguments(raw_args)
        if not isinstance(name, str) or not name:
            raise CaptureError("proposed tool call has no function name")
        calls.append({"id": call_id, "function": name, "args": args})
    return calls


def parse_tool_arguments(raw_args: Any) -> dict:
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            return {"_raw_arguments": raw_args}
        return args
    except Exception:
        return {"_raw_arguments": raw_args}


def _validate_offered_tools(tools: list) -> None:
    domain = _required_env("ATTRIGUARD_DOMAIN")
    classified, _ = classification_for_domain(domain)
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = (function or {}).get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name:
            raise CaptureError("offered tool has no function name")
        names.add(name)
    missing = sorted(names - set(classified) - set(LOCAL_EXTERNAL_TOOLS))
    if missing:
        raise CaptureError(f"unclassified tools for DTAP domain {domain!r}: {missing}")


def maybe_capture(kwargs: dict, resp: Any) -> bool:
    try:
        agent_model = _required_env("ATTRIGUARD_AGENT_MODEL")
        if kwargs.get("stream"):
            return False
        calls = _parse_response_calls(resp)
        if not calls:
            return False
        if kwargs.get("model") != agent_model:
            raise CaptureError(
                f"agent tool-call model {kwargs.get('model')!r} differs from "
                f"configured DTAP model {agent_model!r}"
            )
        tools = kwargs.get("tools")
        messages = kwargs.get("messages")
        if not isinstance(tools, list) or not tools:
            raise CaptureError("agent proposed calls without a non-empty tools list")
        if not isinstance(messages, list) or not messages:
            raise CaptureError("agent proposed calls without a non-empty message history")
        _validate_offered_tools(tools)
        step_kwargs: dict[str, Any] = {}
        for key in (
            "temperature", "top_p", "tool_choice", "parallel_tool_calls", "max_tokens",
            "reasoning_effort",
        ):
            value = kwargs.get(key)
            if value is None or isinstance(value, (Omit, NotGiven)):
                continue
            json.dumps(value)
            step_kwargs[key] = value
        _STATE["seq"] += 1
        _STATE["exchanges"].append({
            "seq": _STATE["seq"],
            "messages": _jsonable(messages),
            "tools": _jsonable(tools),
            "kwargs": step_kwargs,
            "calls": calls,
            "verdicts": None,
            "result_states": None,
        })
        del _STATE["exchanges"][:-_MAX_EXCHANGES]
        return True
    except Exception as exc:
        _STATE["incomplete"] = True
        log_event({"event": "capture_error", "error": f"{type(exc).__name__}: {exc}"})
        if isinstance(exc, CaptureError):
            raise
        raise CaptureError(str(exc)) from exc


def ensure_state() -> AttriGuardState:
    if _STATE["built"]:
        return _STATE["ag"]
    user_task = read_trusted_request()
    if not user_task and task_id() is None:
        for exchange in reversed(_STATE["exchanges"]):
            for message in exchange["messages"]:
                if message.get("role") == "user" and isinstance(message.get("content"), str):
                    user_task = message["content"]
                    break
            if user_task:
                break
    if not user_task:
        raise TrustedRequestError("no trusted request is available")
    base_url = _required_env("ATTRIGUARD_PROXY_URL")
    agent_model = _required_env("ATTRIGUARD_AGENT_MODEL")
    aux_model = _required_env("ATTRIGUARD_MODEL_ID")
    judge_model = _required_env("ATTRIGUARD_JUDGE_MODEL_ID")
    if len({agent_model, aux_model, judge_model}) != 1:
        raise RuntimeError("every AttriGuard LLM role must use the configured DTAP model")
    temperature = float(_required_env("ATTRIGUARD_TEMPERATURE"))
    reasoning_effort = _required_env("ATTRIGUARD_REASONING_EFFORT")
    cap_text = os.environ.get("ATTRIGUARD_MAX_OUTPUT_TOKENS", "").strip()
    max_tokens = int(cap_text) if cap_text else None
    domain = _required_env("ATTRIGUARD_DOMAIN")
    _, external = classification_for_domain(domain)
    client_kwargs = {
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
    }
    state = AttriGuardState(
        user_task=user_task,
        shadow_llm=ProxyChatClient(base_url, agent_model, **client_kwargs),
        attenuation_llm=ProxyChatClient(base_url, aux_model, **client_kwargs),
        judge_llm=ProxyChatClient(base_url, judge_model, **client_kwargs),
        external_tools=set(external | LOCAL_EXTERNAL_TOOLS),
        attenuation_level=int(_required_env("ATTRIGUARD_LEVEL")),
        survival_mode=_required_env("ATTRIGUARD_SURVIVAL"),
        skip_empty_tool_results_audit=_flag("ATTRIGUARD_SKIP_EMPTY_AUDIT", "1"),
        log=log_event,
    )
    _STATE["ag"] = state
    _STATE["built"] = True
    return state


def get_state() -> Optional[AttriGuardState]:
    return _STATE["ag"]


def _audit_lock() -> asyncio.Lock:
    global _LOCK, _LOCK_LOOP
    loop = asyncio.get_running_loop()
    if _LOCK is None or _LOCK_LOOP is not loop:
        _LOCK = asyncio.Lock()
        _LOCK_LOOP = loop
    return _LOCK


async def audit_call(tool_name: str, args: dict) -> Optional[dict]:
    async with _audit_lock():
        try:
            state = await asyncio.to_thread(ensure_state)
            signature = call_signature(tool_name, args)
            for exchange in reversed(_STATE["exchanges"]):
                if not any(
                    call_signature(call["function"], call["args"]) == signature
                    for call in exchange["calls"]
                ):
                    continue
                if exchange["verdicts"] is None:
                    skip_after = _STATE.get("pending_skip_after_seq")
                    force_skip = isinstance(skip_after, int) and exchange["seq"] > skip_after
                    if force_skip:
                        _STATE["pending_skip_after_seq"] = None
                    try:
                        exchange["verdicts"] = await asyncio.to_thread(
                            state.audit_step,
                            {
                                "messages": exchange["messages"],
                                "tools": exchange["tools"],
                                "kwargs": exchange["kwargs"],
                            },
                            exchange["calls"],
                            force_skip=force_skip,
                        )
                    except ReplayError as exc:
                        log_event({
                            "event": "replay_error", "error": str(exc)[:300],
                            "seq": exchange["seq"],
                        })
                        raise
                    decisions = [
                        exchange["verdicts"].get(
                            call_signature(call["function"], call["args"])
                        )
                        for call in exchange["calls"]
                    ]
                    exchange["result_states"] = [
                        "blocked" if verdict is not None and not verdict.get("allow") else None
                        for verdict in decisions
                    ]
                    _update_skip_state(exchange)
                verdict = exchange["verdicts"].get(signature)
                if verdict is None:
                    raise CaptureError(
                        f"captured exchange {exchange['seq']} has no verdict for {tool_name}"
                    )
                return verdict
            log_event({"event": "capture_miss", "tool": tool_name, "args": args})
            raise CaptureError(f"no captured exchange contains {tool_name}{args}")
        except asyncio.CancelledError:
            raise
        except AttriGuardIncomplete:
            raise
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            _STATE["incomplete"] = True
            if not isinstance(exc, (ReplayError, CaptureError, TrustedRequestError)):
                log_event({"event": "state_error", "error": f"{type(exc).__name__}: {exc}"})
            raise AttriGuardIncomplete(
                "AttriGuard task is incomplete and must be retried: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def _result_is_empty(raw: Any) -> bool:
    content = getattr(raw, "content", None)
    if content is None and isinstance(raw, dict) and "content" in raw:
        content = raw.get("content")
    if content is None:
        content = raw
    if isinstance(content, str):
        return not content.strip()
    if content is None:
        return True
    if not isinstance(content, list):
        return False
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text", block.get("content"))
        if isinstance(text, str):
            if text.strip():
                return False
        else:
            return False
    return True


def _update_skip_state(exchange: dict) -> None:
    states = exchange.get("result_states")
    if not isinstance(states, list) or not states or any(state is None for state in states):
        return
    if all(state in {"blocked", "empty"} for state in states):
        _STATE["pending_skip_after_seq"] = exchange["seq"]
        fully_blocked = all(state == "blocked" for state in states)
        log_event({
            "event": "fully_blocked_step" if fully_blocked else "all_empty_step",
            "seq": exchange["seq"],
            "n_calls": len(states),
            "n_blocked": sum(state == "blocked" for state in states),
        })


def record_tool_result(tool_name: str, args: dict, raw: Any) -> None:
    signature = call_signature(tool_name, args)
    for exchange in reversed(_STATE["exchanges"]):
        states = exchange.get("result_states")
        if not isinstance(states, list):
            continue
        for index, call in enumerate(exchange["calls"]):
            if states[index] is not None:
                continue
            if call_signature(call["function"], call["args"]) == signature:
                states[index] = "empty" if _result_is_empty(raw) else "nonempty"
                _update_skip_state(exchange)
                return
    _STATE["incomplete"] = True
    log_event({"event": "capture_miss", "phase": "post_tool_call",
               "tool": tool_name, "args": args})
    raise AttriGuardIncomplete(
        f"AttriGuard could not associate the result for {tool_name}{args}"
    )


def blocked_text(tool_name: str, args: dict) -> str:
    state = get_state()
    user_task = state.user_task if state is not None else (read_trusted_request() or "")
    return blocked_reason(tool_name, args, user_task)


def _dump_record() -> None:
    tid = task_id()
    if tid is None:
        return
    state = _STATE["ag"]
    events = _STATE["events"]
    record = {
        "task_id": tid,
        "n_exchanges": _STATE["seq"],
        "n_blocked": sum(
            1 for event in events
            if event.get("event") == "survival" and not event.get("allow")
        ),
        "n_audited_calls": sum(1 for event in events if event.get("event") == "survival"),
        "n_audit_skipped": sum(1 for event in events if event.get("event") == "audit_skipped"),
        "n_capture_miss": sum(1 for event in events if event.get("event") == "capture_miss"),
        "n_replay_error": sum(1 for event in events if event.get("event") == "replay_error"),
        "n_judge_error": sum(1 for event in events if event.get("event") == "judge_error"),
        "n_attenuation_error": sum(
            1 for event in events if event.get("event") == "attenuation_error"
        ),
        "n_incomplete": int(bool(_STATE.get("incomplete"))),
        "events": events,
    }
    if state is not None:
        record["defense_llm_usage"] = {
            "shadow": state.shadow_llm.usage,
            "attenuation": state.attenuation_llm.usage,
            "judge": state.judge_llm.usage,
        }
    try:
        write_method_record(tid, "attriguard", record)
    except Exception:
        pass


atexit.register(_dump_record)
