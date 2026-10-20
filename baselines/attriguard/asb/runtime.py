from __future__ import annotations

import json
import os
from typing import Any, Mapping

from baselines.attriguard.asb.official import attriguard as official

ATTENUATION_LEVEL = 2
SURVIVAL_MODE = "hyper_fuzzy"
MAX_ITERS = 15
STEP_PROMPT_TEMPLATE = "At step {round}, you need to {message} "
CLIENT_OVERRIDES = (
    "ATTRIGUARD_BASE_URL",
    "ATTRIGUARD_API_KEY",
    "ATTRIGUARD_MAX_TOKENS",
    "ATTRIGUARD_TEMPERATURE",
    "ATTRIGUARD_TOP_P",
    "ATTRIGUARD_SEED",
    "ATTRIGUARD_REASONING_EFFORT",
)


def step_message(prompt: str, rounds: int) -> str:
    prefix = STEP_PROMPT_TEMPLATE.format(round=rounds + 1, message="")
    return prompt[len(prefix) - 1 : -1]


def _signature(call: Mapping[str, Any]) -> tuple[Any, str]:
    name = call.get("name") or call.get("function")
    params = call.get("parameters") or call.get("args") or {}
    try:
        key = json.dumps(params, ensure_ascii=True, sort_keys=True)
    except Exception:
        key = "{}"
    return name, key


def _query(messages: list[dict[str, Any]], tools: Any) -> Any:
    from pyopenagi.utils.chat_template import Query

    return Query(messages=messages, tools=tools)


class AttriGuardAgentMixin:
    def run(self):
        self._attriguard_in_shadow = False
        self._attriguard_planning = False
        self._attriguard_blocked: list[dict[str, Any]] = []
        self._attriguard_user_task: str | None = None
        self.attriguard = official.AttriGuard(
            shadow_request_fn=self._attriguard_shadow_request,
            attenuation_client=official.build_attenuation_client_from_env(),
            attenuation_level=ATTENUATION_LEVEL,
            survival_mode=SURVIVAL_MODE,
            max_iters=MAX_ITERS,
        )
        return super().run()

    def automatic_workflow(self):
        self._attriguard_planning = True
        try:
            return super().automatic_workflow()
        finally:
            self._attriguard_planning = False

    def _attriguard_shadow_request(self, messages: list[dict[str, Any]], tools: Any) -> Any:
        self._attriguard_in_shadow = True
        try:
            return self.get_response(query=_query(messages, tools))[0]
        finally:
            self._attriguard_in_shadow = False

    def get_response(self, query: Any, temperature: float = 0.0):
        if self._attriguard_in_shadow or self._attriguard_planning:
            return super().get_response(query, temperature)
        guard = self.attriguard
        self._attriguard_blocked = []
        self._attriguard_user_task = None
        guard.ingest_messages(self.messages)
        result = super().get_response(query, temperature)
        tool_calls = getattr(result[0], "tool_calls", None)
        if tool_calls:
            step_context = f"{self.task_input}\nCurrent step: {step_message(self.messages[-1].get('content'), self.rounds)}"
            _executed, blocked = guard.filter_tool_calls(step_context, self.messages, tool_calls, query.tools)
            self._attriguard_blocked = list(blocked)
            self._attriguard_user_task = step_context
        return result

    def call_tools(self, tool_calls: Any, final_stage: bool = False):
        normalized = self.normalize_tool_calls(tool_calls)
        valid = set(self.tool_list.keys())
        executable = [call for call in normalized if call.get("name") and call["name"] in valid]
        blocked_signatures = {_signature(call) for call in self._attriguard_blocked}
        plan = [(call, _signature(call) in blocked_signatures) for call in executable]
        if not any(is_blocked for _call, is_blocked in plan):
            return super().call_tools(tool_calls, final_stage=final_stage)
        user_task = self._attriguard_user_task or self.task_input
        unknown = sorted({call["name"] for call in normalized if call.get("name") and call["name"] not in valid})
        if any(not is_blocked for _call, is_blocked in plan):
            passthrough = [
                call for call in normalized
                if call.get("name") not in valid or _signature(call) not in blocked_signatures
            ]
            actions, observations, success = super().call_tools(passthrough, final_stage=final_stage)
            lead = observations[:1] if unknown else []
            per_call = observations[len(lead):]
        else:
            self.last_tool_call_non_retryable = False
            actions, per_call, success = [], [], True
            lead = [f"Unknown tool name skipped: {', '.join(unknown)}"] if unknown else []
        merged_actions: list[str] = []
        merged_observations: list[str] = list(lead)
        cursor = 0
        for call, is_blocked in plan:
            if is_blocked:
                merged_actions.append(f"I will call the {call['name']} with the params as None")
                merged_observations.append(official.blocked_observation(call, user_task))
            else:
                merged_actions.append(actions[cursor])
                merged_observations.append(per_call[cursor])
                cursor += 1
        return merged_actions, merged_observations, success


def agent_mixin(settings: Mapping[str, Any]) -> type:
    for name in CLIENT_OVERRIDES:
        os.environ.pop(name, None)
    os.environ["ATTRIGUARD_MODEL"] = settings["model"]["openai_request_id"]
    return AttriGuardAgentMixin
