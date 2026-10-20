from __future__ import annotations

from agentdojo.agent_pipeline import (
    AgentPipeline,
    InitQuery,
    SystemMessage,
    ToolsExecutionLoop,
    ToolsExecutor,
)
from agentdojo.agent_pipeline.agent_pipeline import load_system_message

from baselines.instructional.core import prevent
from scripts.agentdojo.openai_llm import ChatLLM


class InstructionalInitQuery(InitQuery):

    def query(self, query, runtime, *args, **kwargs):
        return super().query(prevent(query), runtime, *args, **kwargs)


def build_pipeline(args, config):
    llm = ChatLLM(config["models"]["agent"]["id"])
    pipeline = AgentPipeline(
        [
            SystemMessage(load_system_message(config["runtime"]["system_message"])),
            InstructionalInitQuery(),
            llm,
            ToolsExecutionLoop([ToolsExecutor(), llm]),
        ]
    )
    return pipeline, None
