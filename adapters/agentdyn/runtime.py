from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop
from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
from agentdojo.types import (
    ChatSystemMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

from adapters.agentdyn.ration_elements import RationInit, RationToolsExecutor
from ration import folds as fold_table, policies as fold_policies
from ration.policy_config import load_domain
from ration.provisioner import Provisioner
from scripts.agentdojo.openai_llm import ChatLLM

ROOT = Path(__file__).resolve().parents[2]
TOOL_LOOP_MAX_ITERS = 50


class ProvisionerLLMClient:

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def complete(self, *, system: str, user: str) -> str:
        messages = [
            ChatSystemMessage(role="system", content=[text_content_block_from_string(system)]),
            ChatUserMessage(role="user", content=[text_content_block_from_string(user)]),
        ]
        _, _, _, output, _ = self._llm.query(user, FunctionsRuntime([]), EmptyEnv(), messages, {})
        return get_text_content_as_str(output[-1]["content"])


def build_pipeline(args: Any, config: dict[str, Any]) -> tuple[AgentPipeline, RationInit]:
    table = ROOT / "adapters/agentdyn/environments" / args.suite / "policy.yaml"
    os.environ["RATION_TABLE"] = str(table)
    domain = load_domain(table)
    model = config["models"]["agent"]["id"]
    folds = fold_table.load(ROOT / "benchmarks/agentdyn/environments" / args.suite / "folds.json")
    policies = fold_policies.load(ROOT / "adapters/agentdyn/environments" / args.suite / f"policies.{model}.json")
    llm = ChatLLM(model)
    provisioner = Provisioner(ProvisionerLLMClient(ChatLLM(config["models"]["provisioner"]["id"])))
    init = RationInit(domain, provisioner, folds, policies)
    system = SystemMessage(load_system_message(config["runtime"]["system_message"]))
    pipeline = AgentPipeline(
        [
            system,
            InitQuery(),
            init,
            llm,
            ToolsExecutionLoop([RationToolsExecutor(), llm], max_iters=TOOL_LOOP_MAX_ITERS),
        ]
    )
    return pipeline, init
