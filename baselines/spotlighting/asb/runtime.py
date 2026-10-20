from __future__ import annotations

from typing import Any, Mapping

from baselines.spotlighting.core import spotlight, system_prompt_with_suffix
from scripts.asb.observations import observation_text, split_observation


class SpotlightingAgentMixin:
    def build_system_instruction(self):
        super().build_system_instruction()
        persona = str(self.prefix)
        for index, item in enumerate(self.messages):
            if isinstance(item, dict) and item.get("role") == "system" and item.get("content") == persona:
                self.messages[index] = {**item, "content": system_prompt_with_suffix(persona)}
                break

    def call_tools(self, tool_calls, final_stage=False):
        actions, observations, success = super().call_tools(tool_calls, final_stage=final_stage)
        names = list(self.tool_list)
        rewritten = []
        for observation in observations:
            parts = split_observation(observation, names)
            rewritten.append(observation if parts is None else observation_text(parts[0], spotlight(parts[1])))
        return actions, rewritten, success


def agent_mixin(settings: Mapping[str, Any]) -> type:
    return SpotlightingAgentMixin
