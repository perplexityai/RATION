from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str

from baselines.twin import core
from baselines.twin.agentdojo.wire import overlay, system_message_of, to_wire, tool_names, user_message
from baselines.twin.prompts import DETECTOR_PROMPT
from scripts.agentdojo.openai_llm import (
    assistant_message_from_response,
    function_to_tool_dict,
    messages_to_dialogue,
)
from scripts.common.llm_client import LLMClient

BUDGET = 100
OVER_BUDGET = "truncate"
EXPLORE_REPROMPTS = 3


class TwinIncomplete(RuntimeError):
    pass


@dataclass
class UnitState:
    hints: dict[str, str | None] = field(default_factory=dict)
    decided: set[str] = field(default_factory=set)


@dataclass
class RuntimeState:
    model: str
    unit: UnitState = field(default_factory=UnitState)


class TwinSafeLLM(BasePipelineElement):

    def __init__(self, llm: Any, state: RuntimeState) -> None:
        self.llm = llm
        self.state = state

    def _client_text(self, messages: list[ChatMessage], tool_dicts: list[dict[str, Any]], *, tool_choice: str | None, role: str) -> str:
        system_prompt, dialogue = messages_to_dialogue(messages)
        try:
            response = LLMClient(self.state.model).call(system_prompt, dialogue, tool_dicts, tool_choice=tool_choice)
        except Exception as exc:
            raise TwinIncomplete(f"Twin {role} call failed; the unit is unmeasured: {type(exc).__name__}: {exc}") from exc
        return get_text_content_as_str(assistant_message_from_response(response)["content"] or [])

    def _explore(self, messages: Sequence[ChatMessage], wire: list[dict[str, Any]], items: list[core.UntrustedItem], new_items: list[core.UntrustedItem], runtime: FunctionsRuntime) -> str | None:
        unit = self.state.unit
        safe_view = core.rewrite_for_safe(wire, unit.hints)
        names = tool_names(wire)
        new_keys = {core.item_key(it) for it in new_items}
        labelled = []
        for k, it in enumerate(items, 1):
            tag = " (new this turn)" if core.item_key(it) in new_keys else ""
            labelled.append(f"[tool output {k}: {names.get(it.call_id or '', 'tool')}{tag}]\n{it.text}")
        explore_wire = core.build_explore_messages(safe_view, labelled, BUDGET)
        explore_messages = overlay(messages, explore_wire[: len(wire)])
        if len(explore_wire) == len(wire) + 1:
            explore_messages.append(user_message(explore_wire[-1]["content"]))
        tool_dicts = [function_to_tool_dict(tool) for tool in runtime.functions.values()]
        for _ in range(EXPLORE_REPROMPTS + 1):
            content = self._client_text(explore_messages, tool_dicts, tool_choice="none" if tool_dicts else None, role="explore")
            reply = core.parse_explore_reply(content)
            if content.strip() and not reply.parse_error:
                break
        hint, _ = core.enforce_budget(reply.hint, BUDGET, OVER_BUDGET)
        if hint is not None:
            detector_messages = [system_message_of(DETECTOR_PROMPT), user_message(core.detector_input(hint))]
            verdict_text = self._client_text(detector_messages, [], tool_choice=None, role="detector")
            suspicious, _ = core.parse_detector_verdict(verdict_text)
            if suspicious is None or suspicious:
                hint = None
        return hint

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        unit = self.state.unit
        messages = list(messages)
        wire = to_wire(messages)
        items = core.untrusted_items(wire)
        new_items = [it for it in items if core.item_key(it) not in unit.decided]
        if new_items:
            hint = self._explore(messages, wire, items, new_items, runtime)
            unit.hints[core.item_key(new_items[-1])] = hint
            for it in new_items:
                unit.decided.add(core.item_key(it))
                unit.hints.setdefault(core.item_key(it), None)
        safe_messages = overlay(messages, core.rewrite_for_safe(wire, unit.hints))
        _, _, _, safe_out, updated_extra = self.llm.query(query, runtime, env, safe_messages, extra_args)
        return query, runtime, env, [*messages, safe_out[-1]], updated_extra
