from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.pi_detector import TransformersBasedPIDetector
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor

from baselines.pidetector.agentdojo.checkpoint import resolve_snapshot
from baselines.pidetector.core import SAFE_LABEL, THRESHOLD
from scripts.agentdojo.openai_llm import ChatLLM


def build_pipeline(args, config):
    llm = ChatLLM(config["models"]["agent"]["id"])
    detector = TransformersBasedPIDetector(
        model_name=str(resolve_snapshot()), safe_label=SAFE_LABEL, threshold=THRESHOLD
    )
    pipeline = AgentPipeline(
        [
            SystemMessage(load_system_message(config["runtime"]["system_message"])),
            InitQuery(),
            llm,
            ToolsExecutionLoop([ToolsExecutor(), detector, llm]),
        ]
    )
    return pipeline, None
