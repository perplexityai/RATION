from __future__ import annotations

import copy
import os
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
OVERLAY_ROOT = ROOT / ".method-deps/progent-asb"
TASK_TYPE = "asb"
ENV_MUST_BE_UNSET = ("SECAGENT_ONLY_ALLOW_NARROW", "SECAGENT_UPDATE", "SECAGENT_IGNORE_UPDATE_ERROR", "ENABLE_SECAGENT")

_SECAGENT_LOCK = threading.RLock()


def apply_environment_of_record(policy_model: str) -> None:
    os.environ["SECAGENT_POLICY_MODEL"] = policy_model
    os.environ["SECAGENT_GENERATE"] = "True"
    os.environ["SECAGENT_TASK_TYPE"] = TASK_TYPE
    for name in ENV_MUST_BE_UNSET:
        os.environ.pop(name, None)


class _GuardedTool:
    def __init__(self, name: str, tool: Any, agent: Any) -> None:
        self._name = name
        self._tool = tool
        self._agent = agent

    def run(self, params: Any) -> Any:
        self._agent._progent_enforce(self._name)
        return self._tool.run(params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)


class ProgentAgentMixin:
    secagent: Any = None

    def run(self):
        self._progent_policy_config: dict[str, Any] | None = None
        return super().run()

    def build_system_instruction(self) -> None:
        super().build_system_instruction()
        for name, tool in list(self.tool_list.items()):
            if not isinstance(tool, _GuardedTool):
                self.tool_list[name] = _GuardedTool(name, tool, self)

    def automatic_workflow(self):
        workflow = super().automatic_workflow()
        pkg = self.secagent
        available = [
            pkg.Tool(name=tool["function"]["name"], description=tool["function"]["description"], args={})
            for tool in self.tools
        ]
        with _SECAGENT_LOCK:
            pkg.update_available_tools(available)
            os.environ["SECAGENT_TASK_TYPE"] = TASK_TYPE
            pkg.generate_security_policy(self.task_input)
            self._progent_policy_config = copy.deepcopy(pkg.get_current_config())
        return workflow

    def _progent_enforce(self, tool_name: str) -> None:
        if self._progent_policy_config is None:
            raise RuntimeError("Progent policy check reached before the episode's policy was generated")
        with _SECAGENT_LOCK:
            self.secagent.update_security_policy(copy.deepcopy(self._progent_policy_config))
            self.secagent.check_tool_call(tool_name, {})


def agent_mixin(settings: Mapping[str, Any]) -> type:
    apply_environment_of_record(settings["model"]["openai_request_id"])
    if str(OVERLAY_ROOT) not in sys.path:
        sys.path.append(str(OVERLAY_ROOT))
    import secagent

    return type("ProgentAgentMixin", (ProgentAgentMixin,), {"secagent": secagent})
