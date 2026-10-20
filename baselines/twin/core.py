from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from baselines.twin.prompts import (DETECTOR_INPUT_TEMPLATE, EXPLORE_BLOCK, HINT_LINE,
                                    SAFE_BLOCK, WITHHELD_NOTICE)


UNTRUSTED_ROLES = ("tool", "function")

UNTRUSTED_BLOCK_TYPES = ("tool_result", "function_call_output")


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                if isinstance(b.get("text"), str):
                    out.append(b["text"])
                elif b.get("type") in UNTRUSTED_BLOCK_TYPES:
                    out.append(json.dumps(b.get("content", ""), default=str))
                elif b.get("type") in ("image_url", "image"):
                    out.append("[image]")
                else:
                    out.append(json.dumps(b, default=str))
            else:
                out.append(str(b))
        return "\n".join(out)
    return json.dumps(content, default=str)


def is_untrusted(msg: dict) -> bool:
    if str(msg.get("role", "")).lower() in UNTRUSTED_ROLES:
        return True
    content = msg.get("content")
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") in UNTRUSTED_BLOCK_TYPES:
                return True
    return False


@dataclass
class UntrustedItem:
    index: int
    call_id: Optional[str]
    text: str


def untrusted_items(messages: list[dict]) -> list[UntrustedItem]:
    items = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or not is_untrusted(m):
            continue
        cid = m.get("tool_call_id") or m.get("id") or m.get("call_id")
        items.append(UntrustedItem(index=i, call_id=cid, text=_content_text(m.get("content"))))
    return items


def item_key(it: UntrustedItem) -> str:
    return it.call_id if it.call_id else f"#idx{it.index}"


def rewrite_for_safe(messages: list[dict], hints: dict[str, Optional[str]]) -> list[dict]:
    out: list[dict] = []
    system_done = False
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            out.append(m)
            continue
        m2 = dict(m)
        role = str(m.get("role", "")).lower()
        if role in ("system", "developer") and not system_done:
            base = _content_text(m.get("content"))
            m2["content"] = f"{base}\n\n{SAFE_BLOCK}" if base else SAFE_BLOCK
            system_done = True
        elif is_untrusted(m):
            cid = m.get("tool_call_id") or m.get("id") or m.get("call_id") or f"#idx{i}"
            hint = hints.get(cid)
            text = WITHHELD_NOTICE
            if hint:
                text = f"{WITHHELD_NOTICE}\n{HINT_LINE.format(hint=hint)}"
            m2["content"] = text
        out.append(m2)
    if not system_done:
        out.insert(0, {"role": "system", "content": SAFE_BLOCK})
    return out


UNTRUSTED_HANDOFF = ("TOOL OUTPUTS (untrusted; visible only to you, NOT to the safe agent). "
                     "The safe agent's trace above shows these outputs as withheld. This is the "
                     "residual it is missing:\n\n{payload}")


def build_explore_messages(safe_messages: list[dict], untrusted: list[str],
                           budget: int) -> list[dict]:
    block = EXPLORE_BLOCK.format(max_hint_length=budget)
    out: list[dict] = []
    system_done = False
    for m in safe_messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        m2 = dict(m)
        role = str(m.get("role", "")).lower()
        if role in ("system", "developer") and not system_done:
            base = _content_text(m.get("content"))

            base = base.replace(SAFE_BLOCK, "").strip()
            m2["content"] = f"{base}\n\n{block}" if base else block
            system_done = True
        out.append(m2)
    if not system_done:
        out.insert(0, {"role": "system", "content": block})
    payload = "\n\n---\n\n".join(t for t in untrusted if t)
    if payload:
        out.append({"role": "user", "content": UNTRUSTED_HANDOFF.format(payload=payload)})
    return out


_NULLS = {"", "null", "none", "n/a", "na", "nil", "no hint", "(null)", "<null>"}


@dataclass
class ExploreReply:
    hint: Optional[str]
    next_action: Optional[str]
    raw: str = ""
    parse_error: Optional[str] = None


def _strip_wrappers(s: str) -> str:
    s = s.strip()
    for a, b in (("`", "`"), ('"', '"'), ("'", "'"), ("<", ">"), ("[", "]")):
        while len(s) >= 2 and s.startswith(a) and s.endswith(b):
            s = s[1:-1].strip()
    return s.rstrip(".").strip() if s.lower().rstrip(".") in _NULLS else s


def parse_explore_reply(text: str) -> ExploreReply:
    raw = text or ""
    hint: Optional[str] = None
    action: Optional[str] = None
    err = None
    m = re.search(r"^[\s>*_#\-]*HINT[\s*_]*:[\s*_]*(.*?)\s*$", raw, re.M | re.I)
    if m:
        cand = _strip_wrappers(m.group(1))
        hint = None if cand.lower().rstrip(".") in _NULLS else cand
    else:
        err = "no_hint_line"
    a = re.search(r"^[\s>*_#\-]*NEXT_?ACTION[\s*_]*:[\s*_]*(.*?)\s*$", raw, re.M | re.I)
    if a:
        cand = re.split(r"THOUGHT\s*:|HINT\s*:|undefined\s+to=", a.group(1), 1, re.I)[0]
        cand = _strip_wrappers(cand)
        action = None if cand.lower().rstrip(".") in _NULLS else cand
    return ExploreReply(hint=hint, next_action=action, raw=raw, parse_error=err)


def enforce_budget(hint: Optional[str], budget: int, policy: str = "truncate") -> tuple[Optional[str], bool]:
    if hint is None or budget <= 0:
        return (None if budget <= 0 else hint), bool(hint) and budget <= 0
    if len(hint) <= budget:
        return hint, False
    if policy == "block":
        return None, True
    return hint[:budget], True


def detector_input(hint: str) -> str:
    return DETECTOR_INPUT_TEMPLATE.format(hint=hint)


def parse_detector_verdict(text: str) -> tuple[Optional[bool], str]:
    raw = (text or "").strip()
    m = re.search(r"[01]", raw[:16])
    if not m:
        return None, raw
    return (m.group(0) == "1"), raw


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _windows(s: str, size: int, step: int) -> Iterable[str]:
    s = _norm(s)
    if len(s) <= size:
        if s:
            yield s
        return
    for i in range(0, len(s) - size + 1, step):
        yield s[i:i + size]


def leak_scan(orig_messages: list[dict], safe_messages: list[dict],
              allowed: Iterable[str] = (), *, window: int = 40, step: int = 10) -> list[str]:
    untrusted_text = " \n ".join(it.text for it in untrusted_items(orig_messages))
    trusted_text = " \n ".join(
        _content_text(m.get("content")) + " " + json.dumps(m.get("tool_calls") or "", default=str)
        for m in orig_messages if isinstance(m, dict) and not is_untrusted(m))
    trusted_norm = _norm(trusted_text)
    allowed_norm = _norm(" \n ".join(a for a in allowed if a))
    safe_norm = _norm(" \n ".join(
        _content_text(m.get("content")) + " " + json.dumps(m.get("tool_calls") or "", default=str)
        for m in safe_messages if isinstance(m, dict)))
    for a in allowed:
        an = _norm(a)
        if an:
            safe_norm = safe_norm.replace(an, " ")
    leaks = []
    for w in _windows(untrusted_text, window, step):
        if w in trusted_norm or (allowed_norm and w in allowed_norm):
            continue
        if w in safe_norm:
            leaks.append(w)
    return leaks


@dataclass
class TwinStats:
    steps: int = 0
    explore_calls: int = 0
    hints: int = 0
    null_hints: int = 0
    budget_hits: int = 0
    detector_calls: int = 0
    detector_blocks: int = 0
    detector_unparsable: int = 0
    parse_errors: int = 0
    explore_retries: int = 0
    explore_empty: int = 0
    explore_exhausted: int = 0
    leaks: int = 0
    untrusted_msgs_withheld: int = 0
    hint_chars: list[int] = field(default_factory=list)
