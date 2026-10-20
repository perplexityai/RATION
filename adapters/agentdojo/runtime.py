from __future__ import annotations

import os
from pathlib import Path

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop
from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
from agentdojo.types import (
    ChatSystemMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

from adapters.agentdojo.ration_elements import RationInit, RationToolsExecutor
from ration import folds as fold_table, policies as fold_policies
from ration.policy_config import load_domain
from ration.provisioner import Provisioner
from scripts.agentdojo.openai_llm import ChatLLM

ROOT = Path(__file__).resolve().parents[2]


class ProvisionerLLMClient:

    def __init__(self, llm) -> None:
        self._llm = llm

    def complete(self, *, system: str, user: str) -> str:
        messages = [
            ChatSystemMessage(role="system", content=[text_content_block_from_string(system)]),
            ChatUserMessage(role="user", content=[text_content_block_from_string(user)]),
        ]
        _, _, _, out, _ = self._llm.query(user, FunctionsRuntime([]), EmptyEnv(), messages, {})
        return get_text_content_as_str(out[-1]["content"])


def build_pipeline(args, llm, system_message: SystemMessage):
    table = ROOT / "adapters/agentdojo/environments" / args.suite / "policy.yaml"
    os.environ["RATION_TABLE"] = str(table)
    domain_table = load_domain(table)
    folds = fold_table.load(ROOT / "benchmarks/agentdojo/environments" / args.suite / "folds.json")
    policies = fold_policies.load(ROOT / "adapters/agentdojo/environments" / args.suite / f"policies.{args.model}.json")
    provisioner = Provisioner(ProvisionerLLMClient(ChatLLM(args.provisioner_model)))
    init = RationInit(domain_table, provisioner, folds, policies)
    pipeline = AgentPipeline(
        [system_message, InitQuery(), init, llm, ToolsExecutionLoop([RationToolsExecutor(), llm])]
    )
    return pipeline, init
