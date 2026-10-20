from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor

from baselines.datasentinel import core
from baselines.datasentinel.agentdojo.runtime import DataSentinelDetector
from scripts.agentdojo.openai_llm import ChatLLM


def build_pipeline(args, config):
    llm = ChatLLM(config["models"]["agent"]["id"])
    detector = DataSentinelDetector(core.DetectorClient())
    pipeline = AgentPipeline(
        [
            SystemMessage(load_system_message(config["runtime"]["system_message"])),
            InitQuery(),
            llm,
            ToolsExecutionLoop([ToolsExecutor(), detector, llm], max_iters=15),
        ]
    )
    return pipeline, detector
