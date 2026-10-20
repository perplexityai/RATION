from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from baselines.twin import core
from baselines.twin.prompts import DETECTOR_PROMPT, HINT_LINE, SAFE_BLOCK, WITHHELD_NOTICE
from scripts.asb.observations import observation_text, split_observation

BUDGET = 100
OVER_BUDGET = "truncate"
EXPLORE_REPROMPTS = 3


@dataclass
class Slot:
    key: str
    tool: str
    output: str
    message_index: int


def marker_text(hint: str | None) -> str:
    return f"{WITHHELD_NOTICE}\n{HINT_LINE.format(hint=hint)}" if hint else WITHHELD_NOTICE


def safe_view(messages: list[Any], slots: list[Slot], hints: Mapping[str, str | None]) -> list[Any]:
    view = [dict(m) if isinstance(m, dict) else m for m in messages]
    offset = 0
    for item in view:
        if isinstance(item, dict) and item.get("role") == "system":
            base = item.get("content") if isinstance(item.get("content"), str) else ""
            item["content"] = f"{base}\n\n{SAFE_BLOCK}" if base else SAFE_BLOCK
            break
    else:
        view.insert(0, {"role": "system", "content": SAFE_BLOCK})
        offset = 1
    for slot in slots:
        index = slot.message_index + offset
        if index >= len(view):
            continue
        item = view[index]
        raw = observation_text(slot.tool, slot.output)
        if not isinstance(item, dict) or not isinstance(item.get("content"), str) or raw not in item["content"]:
            raise RuntimeError(f"twin: untrusted output of {slot.tool} not found verbatim at its message")
        item["content"] = item["content"].replace(raw, observation_text(slot.tool, marker_text(hints.get(slot.key))), 1)
    return view


class TwinAgentMixin:
    model = ""
    client: Any = None

    def run(self):
        if type(self).client is None:
            from openai import OpenAI

            type(self).client = OpenAI()
        self._tw_slots: list[Slot] = []
        self._tw_hints: dict[str, str | None] = {}
        self._tw_decided: set[str] = set()
        return super().run()

    def call_tools(self, tool_calls, final_stage=False):
        actions, observations, success = super().call_tools(tool_calls, final_stage=final_stage)
        names = list(self.tool_list)
        position = 0
        for observation in observations:
            parts = split_observation(observation, names)
            if parts is None:
                continue
            key = f"#s{int(self.rounds)}.{len(self.messages)}.{position}"
            self._tw_slots.append(Slot(key=key, tool=parts[0], output=parts[1], message_index=len(self.messages)))
            position += 1
        return actions, observations, success

    def _tw_call(self, messages: list[Any], *, tools: Any) -> str:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "none"
        content = self.client.chat.completions.create(**kwargs).choices[0].message.content
        return content if isinstance(content, str) else ""

    def _explore(self, messages: list[Any], tools: Any, items: list[Slot], new_items: list[Slot]) -> str | None:
        view = safe_view(messages, items, self._tw_hints)
        new_keys = {s.key for s in new_items}
        labelled = [
            f"[tool output {k}: {slot.tool}{' (new this turn)' if slot.key in new_keys else ''}]\n{slot.output}"
            for k, slot in enumerate(items, 1)
        ]
        explore_messages = core.build_explore_messages(view, labelled, BUDGET)
        for _attempt in range(EXPLORE_REPROMPTS + 1):
            content = self._tw_call(explore_messages, tools=tools)
            reply = core.parse_explore_reply(content)
            if content.strip() and not reply.parse_error:
                break
        hint, _hit = core.enforce_budget(reply.hint, BUDGET, OVER_BUDGET)
        if hint is not None:
            verdict = self._tw_call(
                [{"role": "system", "content": DETECTOR_PROMPT}, {"role": "user", "content": core.detector_input(hint)}],
                tools=None,
            )
            suspicious, _raw = core.parse_detector_verdict(verdict)
            if suspicious is None or suspicious:
                hint = None
        return hint

    def get_response(self, query, temperature=0.0):
        messages = list(query.messages)
        items = [s for s in self._tw_slots if s.message_index < len(messages)]
        new_items = [s for s in items if s.key not in self._tw_decided]
        if new_items:
            self._tw_hints[new_items[-1].key] = self._explore(messages, query.tools, items, new_items)
            for slot in new_items:
                self._tw_decided.add(slot.key)
                self._tw_hints.setdefault(slot.key, None)
        view = safe_view(messages, items, self._tw_hints)
        shown = type(query)(messages=view, tools=query.tools, message_return_type=getattr(query, "message_return_type", "text"))
        return super().get_response(shown, temperature)


def agent_mixin(settings: Mapping[str, Any]) -> type:
    return type("TwinAgentMixin", (TwinAgentMixin,), {"model": settings["model"]["openai_request_id"]})
