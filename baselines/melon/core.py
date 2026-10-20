from __future__ import annotations

import copy
import json
from typing import Any, Callable, Optional

from baselines.melon.prompts import (
    FEWSHOT_MESSAGES,
    MASKED_READ_CALL_ID,
    TASK_NEUTRAL_PROMPT,
    fence_tool_output,
)


FocusMap = dict[str, list[str]]


def transform_tool_call(function: Optional[str], args: Any, focus: FocusMap) -> str:
    if function is None:
        return "No tool calls"
    if not isinstance(args, dict):
        args = {}
    keep = focus.get(function)
    text = function + "("
    if args:
        items = ([(k, v) for k, v in args.items() if k in keep] if keep is not None
                 else list(args.items()))
        for name, value in items:
            text += f"{name} = {value}, "
        if items:
            text = text[:-2]
    return text + ")"


class MelonState:

    def __init__(
        self,
        masked_llm,
        embed_fn: Callable[[list[str]], list[list[float]]],
        theta: float,
        focus: Optional[FocusMap] = None,
        log: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.masked_llm = masked_llm
        self.embed_fn = embed_fn
        self.theta = float(theta)
        self.focus = dict(focus or {})
        self.log = log or (lambda ev: None)

        self._bank: dict[str, list[float]] = {}

    def build_masked_messages(self, messages: list[dict]) -> list[dict]:
        id2name = _tool_call_id_to_name(messages)
        tool_contents = ""
        masked: list[dict] = []
        for m in messages:
            if m.get("role") == "system":
                masked.append(m)
            elif m.get("role") == "tool":
                fn = id2name.get(m.get("tool_call_id")) or _tool_call_function(m)
                tool_contents += fence_tool_output(fn, _content_text(m))
        masked += copy.deepcopy(FEWSHOT_MESSAGES)
        masked += [
            {"role": "user", "content": TASK_NEUTRAL_PROMPT},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": MASKED_READ_CALL_ID, "type": "function",
                             "function": {"name": "read_file",
                                          "arguments": json.dumps({"name": "random.txt"})}}]},
            {"role": "tool", "tool_call_id": MASKED_READ_CALL_ID, "content": tool_contents},
        ]
        for m in masked:
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") if isinstance(tc, dict) else None
                if isinstance(fn, dict) and isinstance(fn.get("arguments"), (dict, list)):
                    fn["arguments"] = json.dumps(fn["arguments"])
        return masked

    def audit_step(self, messages: list[dict], original_calls: list[dict],
                   tools: Optional[list] = None) -> dict:
        masked_messages = self.build_masked_messages(messages)
        _content, masked_calls, _ = self.masked_llm.chat(masked_messages, tools=tools)
        fresh = []
        for c in masked_calls or []:
            t = transform_tool_call(c.get("function"), c.get("args"), self.focus)
            if t not in self._bank:
                fresh.append(t)
        if fresh:
            vecs = self.embed_fn(fresh)
            for t, v in zip(fresh, vecs):
                self._bank[t] = v
        if not original_calls:
            self.log({"event": "audit", "alert": False, "reason": "no-original-calls",
                      "n_masked": len(self._bank)})
            return {"alert": False, "max_cosine": -1.0, "n_masked_calls": len(self._bank)}
        orig_texts = [transform_tool_call(c.get("function"), c.get("args"), self.focus)
                      for c in original_calls]
        orig_vecs = self.embed_fn(orig_texts) if orig_texts else []
        alert = False
        max_cos = -1.0
        for bank_vec in self._bank.values():
            if alert:
                break
            for ov in orig_vecs:
                c = _cosine(ov, bank_vec)
                if c > max_cos:
                    max_cos = c
                if c > self.theta:
                    alert = True
                    break
        self.log({"event": "audit", "alert": alert, "max_cosine": round(max_cos, 4),
                  "n_masked": len(self._bank), "n_original": len(orig_texts),
                  "theta": self.theta})
        return {"alert": alert, "max_cosine": max_cos, "n_masked_calls": len(self._bank)}


def _content_text(message: dict) -> str:
    c = message.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for b in c:
            if isinstance(b, dict):
                t = b.get("text", b.get("content"))
                if isinstance(t, str):
                    out.append(t)
        return "\n".join(out)
    return "" if c is None else str(c)


def _tool_call_id_to_name(messages: list[dict]) -> dict:
    m: dict = {}
    for msg in messages:
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") if isinstance(tc, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else None
            cid = tc.get("id") if isinstance(tc, dict) else None
            if cid and name:
                m[cid] = name
    return m


def _tool_call_function(tool_message: dict) -> str:
    tc = tool_message.get("tool_call")
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return fn["name"]
        if isinstance(fn, str):
            return fn
    return tool_message.get("name") or "tool"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)
