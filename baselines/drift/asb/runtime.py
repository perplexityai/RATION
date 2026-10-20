from __future__ import annotations

from typing import Any, Mapping

from baselines.drift.asb.official.drift import DRIFT


class DriftAgentMixin:
    def run(self):
        self.drift_agent = DRIFT(self.args, self.logger)
        return super().run()

    def build_system_instruction(self) -> None:
        super().build_system_instruction()
        self.drift_agent.query = self.task_input
        self.drift_agent.tools = self.tools
        self.drift_agent.get_all_tool_privilege(self.tools)

    def automatic_workflow(self):
        workflow = super().automatic_workflow()
        if workflow:
            try:
                for step in workflow:
                    self.drift_agent.initial_traj.append(step.get("tool_use", []))
                self.drift_agent.initial_traj = [*sum(self.drift_agent.initial_traj, [])]
            except Exception as exc:
                self.logger.log(f"Error while initializing trajectory: {exc}", level="error")
                self.drift_agent.initial_traj = []
        return workflow

    def call_tools(self, tool_calls: Any, final_stage: bool = False):
        drift = self.drift_agent
        passed: list[dict[str, Any]] = []
        for call in self.normalize_tool_calls(tool_calls):
            tool_name = call.get("name")
            position = len(drift.achieved_function_trajectory)
            if position < len(drift.initial_traj) and tool_name == drift.initial_traj[position]:
                drift.achieved_function_trajectory.append(tool_name)
                passed.append(call)
                continue
            drift.alignment_judge(
                query=self.task_input,
                initial_function_trajectory=drift.initial_traj,
                current_function_trajectory=[*drift.achieved_function_trajectory, tool_name],
                messages=self.messages,
            )
        if not passed:
            self.last_tool_call_non_retryable = True
            return [], [], False
        actions, observations, success = super().call_tools(passed, final_stage=final_stage)
        try:
            observations = drift.injection_isolate(
                query=self.task_input, tool_call="[Action]: " + ";".join(actions),
                observations=observations, messages=self.messages,
            )
        except Exception as exc:
            self.logger.log(f"Error during injection isolation: {exc}", level="error")
        return actions, observations, success


def agent_mixin(settings: Mapping[str, Any]) -> type:
    return DriftAgentMixin
