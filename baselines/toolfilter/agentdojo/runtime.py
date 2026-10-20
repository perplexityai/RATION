from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, InitQuery
from agentdojo.agent_pipeline.agent_pipeline import TOOL_FILTER_PROMPT
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLMToolFilter
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor

from baselines.toolfilter.agentdojo.transport import FilterTransport


def build_pipeline(args, llm, system_message):
    transport = FilterTransport.start()
    tool_filter = OpenAILLMToolFilter(TOOL_FILTER_PROMPT, transport.client, args.model, temperature=0.0)
    pipeline = AgentPipeline(
        [
            system_message,
            InitQuery(),
            tool_filter,
            llm,
            ToolsExecutionLoop([ToolsExecutor(), llm]),
        ]
    )
    return pipeline, transport


def finish_taskset(transport: FilterTransport) -> None:
    transport.close()
