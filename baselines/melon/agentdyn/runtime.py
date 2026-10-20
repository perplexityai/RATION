from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, SystemMessage, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.agent_pipeline import load_system_message

from baselines.melon.agentdojo.config import MAX_ITERS
from baselines.melon.agentdojo_family import MelonElement, MelonInitQuery, RuntimeState
from baselines.melon.comparator import OpenAIEmbedder
from scripts.agentdojo.openai_llm import ChatLLM


def build_pipeline(args, config):
    llm = ChatLLM(config["models"]["agent"]["id"])
    state = RuntimeState(embedder=OpenAIEmbedder())
    pipeline = AgentPipeline(
        [
            SystemMessage(load_system_message(config["runtime"]["system_message"])),
            MelonInitQuery(state),
            llm,
            ToolsExecutionLoop([ToolsExecutor(), MelonElement(llm, state)], max_iters=MAX_ITERS),
        ]
    )
    return pipeline, state
