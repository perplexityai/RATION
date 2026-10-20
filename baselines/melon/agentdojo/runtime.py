from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, ToolsExecutionLoop, ToolsExecutor

from baselines.melon.agentdojo_family import MelonElement, MelonInitQuery, RuntimeState
from baselines.melon.comparator import OpenAIEmbedder


def build_pipeline(args, llm, system_message):
    state = RuntimeState(embedder=OpenAIEmbedder())
    pipeline = AgentPipeline(
        [
            system_message,
            MelonInitQuery(state),
            llm,
            ToolsExecutionLoop([ToolsExecutor(), MelonElement(llm, state)]),
        ]
    )
    return pipeline, state
