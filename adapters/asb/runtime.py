from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Mapping

from ration import feedback, folds as fold_table, policies as fold_policies
from ration.model import Verdict
from ration.monitor import Monitor
from ration.policy_config import load_domain
from ration.provisioner import Provisioner
from scripts.asb.upstream import inventory
from scripts.asb.upstream_api import task_total

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "adapters/asb/config"
ENVIRONMENTS = ROOT / "adapters/asb/environments"
BENCHMARK_ENVIRONMENTS = ROOT / "benchmarks/asb/environments"
_FRAMEWORK_MARKERS = ("[RATION]",)


def framework_prompt_chars(messages: list[dict[str, Any]]) -> int:
    owned = 0
    for message in messages or []:
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text", ""), str)
            )
        if not isinstance(content, str):
            continue
        for line in content.splitlines(keepends=True):
            if any(marker in line for marker in _FRAMEWORK_MARKERS):
                owned += len(line)
    return owned


class _ProvisionerClient:
    def __init__(self, *, model: str, max_output_tokens: int) -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self._max_output_tokens = max_output_tokens

    def complete(self, *, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self._max_output_tokens,
            temperature=0,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("ASB RATION Provisioner returned no text completion")
        return content


class RuntimeContext:
    def __init__(self, settings: Mapping[str, Any]) -> None:
        os.environ["RATION_TABLE"] = str(CONFIG)
        model = settings["model"]["id"]
        self.environments = {}
        for scenario in inventory():
            self.environments[scenario.name] = (
                load_domain(ENVIRONMENTS / scenario.name / "policy.yaml"),
                fold_table.load(BENCHMARK_ENVIRONMENTS / scenario.name / "folds.json"),
                fold_policies.load(ENVIRONMENTS / scenario.name / f"policies.{model}.json"),
            )
        self._provisioner = Provisioner(
            _ProvisionerClient(model=settings["model"]["openai_request_id"], max_output_tokens=settings["max_output_tokens"])
        )
        self._outcomes: dict[tuple[str, str, int], Any] = {}
        self._outcome_lock = threading.Lock()

    def fold(self, environment: str, attacker_tool: str) -> int:
        return fold_table.fold_of(self.environments[environment][1], attacker_tool)

    def outcome(self, environment: str, trusted_request: str, fold: int) -> Any:
        key = (environment, trusted_request.strip(), fold)
        with self._outcome_lock:
            if key not in self._outcomes:
                table, _folds, policies = self.environments[environment]
                self._outcomes[key] = self._provisioner.plan(
                    trusted_request,
                    table.with_policy(fold_policies.for_fold(policies, fold)),
                    reflect_tags=(),
                )
            return self._outcomes[key]


class RationAgentMixin:
    context: RuntimeContext

    @property
    def _ration_environment(self) -> str:
        return str(self.agent_name).split("/")[-1]

    def asb_framework_prompt_chars(self, messages: list[dict[str, Any]]) -> int:
        return framework_prompt_chars(messages)

    def add_attacker_tool(self) -> None:
        super().add_attacker_tool()
        self.tools.pop()

    def _fresh_charged_tokens(self) -> int:
        total = task_total(self.asb_task_id)
        fresh = total - self._ration_tokens_seen
        self._ration_tokens_seen = total
        return fresh

    def _refresh_soft_reminder(self, details: list[Mapping[str, Any]]) -> None:
        current: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for detail in details:
            risk_id = detail.get("risk_id")
            if isinstance(risk_id, str) and risk_id not in seen:
                seen.add(risk_id)
                current.append(detail)
        if not current:
            return
        previous = [risk for risk in self._ration_monitor.breached_risks() if risk["risk_id"] not in seen]
        block = feedback.soft_reminder(current, previous)
        for message in self.messages:
            content = message.get("content")
            if message.get("role") != "system" or not isinstance(content, str):
                continue
            if self._ration_reminder_block is not None:
                content = content.removesuffix(f"\n\n{self._ration_reminder_block}")
            message["content"] = f"{content}\n\n{block}"
            self._ration_reminder_block = block
            return

    def _withheld(self, message: str) -> tuple[list[str], list[str], bool]:
        self.last_tool_call_non_retryable = True
        return ["I will not execute the withheld tool call."], [message], False

    def call_tools(self, tool_calls: Any, final_stage: bool = False):
        normalized = self.normalize_tool_calls(tool_calls)
        if not normalized:
            return super().call_tools(tool_calls, final_stage=final_stage)
        charged = self._fresh_charged_tokens()
        token_decision = self._ration_monitor.record_tokens(charged) if charged else None
        soft_details: list[Mapping[str, Any]] = []
        token_stop = None
        if token_decision is not None:
            if token_decision.verdict is Verdict.SOFT_BREACH:
                soft_details.extend(token_decision.details)
            else:
                token_stop = token_decision
        actions: list[str] = []
        observations: list[str] = []
        successes: list[bool] = []
        for call in normalized:
            name = call.get("name")
            if not isinstance(name, str) or name not in self.tool_list:
                part = super().call_tools([call], final_stage=final_stage)
            elif token_stop is not None:
                part = self._withheld(token_stop.message)
            else:
                decision = self._ration_monitor.check_call(name, {})
                if decision.verdict is Verdict.ALLOW:
                    part = super().call_tools([call], final_stage=final_stage)
                    if part[2]:
                        self._ration_monitor.commit_success(name, {})
                elif decision.verdict is Verdict.SOFT_BREACH:
                    soft_details.extend(decision.details)
                    part = self._withheld(feedback.soft_withheld())
                else:
                    part = self._withheld(decision.message)
            actions.extend(part[0])
            observations.extend(part[1])
            successes.append(bool(part[2]))
        self._refresh_soft_reminder(soft_details)
        return actions, observations, bool(successes) and all(successes)

    def run(self):
        environment = self._ration_environment
        fold = self.context.fold(environment, self.tool_name)
        self._ration_monitor = Monitor(self.context.outcome(environment, self.task_input, fold).plan)
        self._ration_tokens_seen = task_total(self.asb_task_id)
        self._ration_reminder_block: str | None = None
        return super().run()


def agent_mixin(settings: Mapping[str, Any]) -> type:
    return type("RationAgentMixin", (RationAgentMixin,), {"context": RuntimeContext(settings)})
