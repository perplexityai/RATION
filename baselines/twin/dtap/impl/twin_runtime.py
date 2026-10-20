from __future__ import annotations

import atexit
import json
import os
import sys
import time
from typing import Any, Optional

from baselines.twin import core as C
from baselines.twin.llm import ProxyChatClient
from baselines.twin.prompts import DETECTOR_PROMPT
from scripts.common.method_evidence import write_method_record


class TwinIncomplete(RuntimeError):
    pass


BUDGET = 100
OVER_BUDGET = "truncate"
EXPLORE_REPROMPTS = 3

_S: dict[str, Any] = {
    "hints": {},
    "decided": set(),
    "events": [],
    "stats": C.TwinStats(),
    "explore": None,
    "detector": None,
    "incomplete": None,
    "written": False,
}


def _agent_model() -> str:
    return os.environ.get("TWIN_AGENT_MODEL", "")


def log_event(ev: dict) -> None:
    _S["events"].append(ev)
    if ev.get("event") in {"leak", "incomplete", "detector_unparsable", "hint_blocked",
                           "explore_exhausted"}:
        print(f"[TWIN] {json.dumps(ev, default=str)[:600]}", file=sys.stderr, flush=True)


def incomplete() -> Optional[str]:
    return _S["incomplete"]


def _mark_incomplete(reason: str, **detail: Any) -> TwinIncomplete:
    if _S["incomplete"] is None:
        _S["incomplete"] = reason
    log_event({"event": "incomplete", "reason": reason, **detail})
    return TwinIncomplete(f"Twin could not run for this task ({reason}); it is unmeasured and must be retried")


def task_id() -> Optional[str]:
    for i, a in enumerate(sys.argv):
        if a == "--task-dir" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--task-dir="):
            return a.split("=", 1)[1]
    return None


def _client(role: str) -> ProxyChatClient:
    if _S[role] is None:
        proxy = os.environ.get("TWIN_PROXY_URL", "")
        if not proxy or not _agent_model():
            raise _mark_incomplete("proxy_or_model_unset", role=role)
        _S[role] = ProxyChatClient(base_url=proxy, model=_agent_model())
    return _S[role]


def applies(kwargs: dict) -> bool:
    if kwargs.get("stream"):
        return False
    if not _agent_model() or kwargs.get("model") != _agent_model():
        return False
    msgs = kwargs.get("messages")
    return isinstance(msgs, list) and bool(msgs)


def _tool_names(messages: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in messages:
        if not isinstance(m, dict):
            continue
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict) and tc.get("id"):
                out[tc["id"]] = ((tc.get("function") or {}).get("name")) or "tool"
    return out


def _run_explore(messages: list[dict], items: list[C.UntrustedItem],
                 new_items: list[C.UntrustedItem], tools: Any) -> Optional[str]:
    st: C.TwinStats = _S["stats"]
    safe_view = C.rewrite_for_safe(messages, _S["hints"])
    names = _tool_names(messages)
    new_keys = {C.item_key(it) for it in new_items}
    labelled = []
    for k, it in enumerate(items, 1):
        tag = " (new this turn)" if C.item_key(it) in new_keys else ""
        labelled.append(f"[tool output {k}: {names.get(it.call_id or '', 'tool')}{tag}]\n{it.text}")
    ex_msgs = C.build_explore_messages(safe_view, labelled, BUDGET)
    t0 = time.time()
    content, rep = "", None
    for attempt in range(EXPLORE_REPROMPTS + 1):
        try:
            content, _calls, _lp = _client("explore").chat(ex_msgs, tools=tools or None,
                                                           extra={"tool_choice": "none"} if tools else None)
            st.explore_calls += 1
        except TwinIncomplete:
            raise
        except Exception as exc:
            raise _mark_incomplete("explore_transport", error=f"{type(exc).__name__}: {str(exc)[:300]}") from exc
        rep = C.parse_explore_reply(content)
        if content.strip() and not rep.parse_error:
            break
        if not content.strip():
            st.explore_empty += 1
        if attempt < EXPLORE_REPROMPTS:
            st.explore_retries += 1
            log_event({"event": "explore_retry", "attempt": attempt + 1,
                       "reason": "empty" if not content.strip() else rep.parse_error})
    if rep.parse_error:
        st.parse_errors += 1
        st.explore_exhausted += 1
        log_event({"event": "explore_exhausted", "detail": rep.parse_error, "raw": content[:300]})
    hint, hit = C.enforce_budget(rep.hint, BUDGET, OVER_BUDGET)
    if hit:
        st.budget_hits += 1
    if hint is None:
        st.null_hints += 1
        log_event({"event": "hint", "hint": None, "budget_hit": hit,
                   "next_action": rep.next_action, "latency_s": round(time.time() - t0, 2)})
        return None
    st.detector_calls += 1
    try:
        out, _c, _l = _client("detector").chat([{"role": "system", "content": DETECTOR_PROMPT},
                                                {"role": "user", "content": C.detector_input(hint)}])
    except TwinIncomplete:
        raise
    except Exception as exc:
        raise _mark_incomplete("detector_transport", error=f"{type(exc).__name__}: {str(exc)[:300]}") from exc
    susp, raw = C.parse_detector_verdict(out)
    if susp is None:
        st.detector_unparsable += 1
        log_event({"event": "detector_unparsable", "raw": raw[:120]})
        susp = True
    if susp:
        st.detector_blocks += 1
        log_event({"event": "hint_blocked", "by": "detector", "hint": hint, "chars": len(hint)})
        return None
    st.hints += 1
    st.hint_chars.append(len(hint))
    log_event({"event": "hint", "hint": hint, "chars": len(hint), "budget_hit": hit,
               "next_action": rep.next_action, "n_new_outputs": len(new_items),
               "n_view_outputs": len(items), "latency_s": round(time.time() - t0, 2)})
    return hint


def transform_request(kwargs: dict) -> dict:
    if _S["incomplete"] is not None:
        raise TwinIncomplete(f"Twin task already incomplete ({_S['incomplete']})")
    st: C.TwinStats = _S["stats"]
    msgs: list[dict] = kwargs.get("messages") or []
    items = C.untrusted_items(msgs)
    if items:
        st.steps += 1
    st.untrusted_msgs_withheld = len(items)
    new = [it for it in items if C.item_key(it) not in _S["decided"]]
    if new:
        hint = _run_explore(msgs, items, new, kwargs.get("tools"))
        _S["hints"][C.item_key(new[-1])] = hint
    for it in new:
        _S["decided"].add(C.item_key(it))
        _S["hints"].setdefault(C.item_key(it), None)
    safe_msgs = C.rewrite_for_safe(msgs, _S["hints"])
    allowed = [h for h in _S["hints"].values() if h]
    leaks = C.leak_scan(msgs, safe_msgs, allowed)
    if leaks:
        st.leaks += len(leaks)
        log_event({"event": "leak", "n": len(leaks), "sample": leaks[0][:120]})
    out = dict(kwargs)
    out["messages"] = safe_msgs
    return out


def _record() -> dict:
    st: C.TwinStats = _S["stats"]
    ex = _S["explore"].usage if _S["explore"] else {}
    de = _S["detector"].usage if _S["detector"] else {}
    hc = st.hint_chars
    return {
        "task_dir": task_id(),
        "n_steps": st.steps, "n_explore_calls": st.explore_calls,
        "n_hints": st.hints, "n_null_hints": st.null_hints,
        "n_budget_hits": st.budget_hits,
        "n_detector_calls": st.detector_calls, "n_detector_blocks": st.detector_blocks,
        "n_detector_unparsable": st.detector_unparsable,
        "n_explore_retries": st.explore_retries, "n_explore_empty": st.explore_empty,
        "n_explore_exhausted": st.explore_exhausted,
        "n_leaks": st.leaks,
        "n_incomplete": int(_S["incomplete"] is not None),
        "incomplete_reason": _S["incomplete"],
        "hint_chars_mean": round(sum(hc) / len(hc), 1) if hc else None,
        "hint_chars_max": max(hc) if hc else None,
        "defense_tokens": {
            "explore_prompt": ex.get("prompt_tokens", 0), "explore_completion": ex.get("completion_tokens", 0),
            "detector_prompt": de.get("prompt_tokens", 0), "detector_completion": de.get("completion_tokens", 0),
            "explore_calls": ex.get("calls", 0), "detector_calls": de.get("calls", 0),
            "transport_errors": (ex.get("errors", 0) + de.get("errors", 0)),
        },
        "events": _S["events"],
    }


def write_record() -> None:
    if _S["written"]:
        return
    _S["written"] = True
    try:
        tid = task_id()
        if not tid:
            return
        write_method_record(tid, "twin", _record())
    except Exception:
        pass


atexit.register(write_record)
