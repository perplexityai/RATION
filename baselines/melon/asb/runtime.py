from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any, Mapping

from baselines.melon.comparator import THETA, OpenAIEmbedder
from baselines.melon.core import MelonState
from scripts.asb.observations import split_observation

FOCUS = {"send_email": ["recipients"], "send_money": ["recipient", "amount"]}
STOP_TEXT = "The tool outputs contain potential prompt injection, stopping execution."
MASKED_RUN_MAX_TOKENS = 32768


def wire_messages(messages: list[Any], outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = [dict(m) for m in messages if isinstance(m, dict) and m.get("role") == "system"]
    for item in outputs:
        wire.append({"role": "assistant", "content": "", "tool_calls": [{"id": item["id"], "type": "function", "function": {"name": item["tool"], "arguments": "{}"}}]})
        wire.append({"role": "tool", "tool_call_id": item["id"], "content": item["output"]})
    return wire


def original_calls(tool_calls: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for call in tool_calls or []:
        if isinstance(call, dict):
            name = call.get("name")
            if not isinstance(name, str) and isinstance(call.get("function"), dict):
                name = call["function"].get("name")
            args = call.get("parameters") if isinstance(call.get("parameters"), dict) else {}
            if isinstance(name, str) and name:
                calls.append({"function": name, "args": args})
        elif isinstance(call, str) and call.strip():
            calls.append({"function": call.strip(), "args": {}})
    return calls


class MaskedClient:
    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def chat(self, messages: list[dict[str, Any]], tools: Any = None) -> tuple[str, list[dict[str, Any]], Any]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": MASKED_RUN_MAX_TOKENS}
        if tools:
            kwargs["tools"] = tools
        message = self.client.chat.completions.create(**kwargs).choices[0].message
        calls: list[dict[str, Any]] = []
        for tc in getattr(message, "tool_calls", None) or []:
            function = getattr(tc, "function", None)
            raw = getattr(function, "arguments", None) or "{}"
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    args = {"_raw_arguments": raw}
            except Exception:
                args = {"_raw_arguments": raw}
            calls.append({"id": getattr(tc, "id", None), "function": getattr(function, "name", None), "args": args})
        return message.content if isinstance(message.content, str) else "", calls, None


class MelonAgentMixin:
    model = ""
    client: Any = None

    def run(self):
        if type(self).client is None:
            from openai import OpenAI

            type(self).client = OpenAI()
        self._ml_outputs: list[dict[str, Any]] = []
        self._ml_halted = False
        self._ml_planning = False
        self._ml_state = MelonState(MaskedClient(self.client, self.model), OpenAIEmbedder(None), THETA, FOCUS)
        return super().run()

    def call_tools(self, tool_calls, final_stage=False):
        actions, observations, success = super().call_tools(tool_calls, final_stage=final_stage)
        names = list(self.tool_list)
        for observation in observations:
            parts = split_observation(observation, names)
            if parts is not None:
                self._ml_outputs.append({"id": f"asb_call_{len(self._ml_outputs)}", "tool": parts[0], "output": parts[1]})
        return actions, observations, success

    def automatic_workflow(self):
        self._ml_planning = True
        try:
            return super().automatic_workflow()
        finally:
            self._ml_planning = False

    def get_response(self, query, temperature=0.0):
        if self._ml_halted:
            now = time.time()
            return SimpleNamespace(response_message=STOP_TEXT, tool_calls=None), [now], [now], [0.0], [0.0]
        result = super().get_response(query, temperature)
        if self._ml_planning or not self._ml_outputs:
            return result
        calls = original_calls(getattr(result[0], "tool_calls", None))
        verdict = self._ml_state.audit_step(wire_messages(query.messages, self._ml_outputs), calls, tools=self.tools)
        if verdict["alert"]:
            self._ml_halted = True
            return (SimpleNamespace(response_message=STOP_TEXT, tool_calls=None), *result[1:])
        return result


def agent_mixin(settings: Mapping[str, Any]) -> type:
    return type("MelonAgentMixin", (MelonAgentMixin,), {"model": settings["model"]["openai_request_id"]})
