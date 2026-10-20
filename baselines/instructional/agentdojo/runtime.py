from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, ToolsExecutionLoop, ToolsExecutor

from baselines.instructional.core import prevent


class InstructionalInitQuery(InitQuery):

    def query(self, query, runtime, *args, **kwargs):
        return super().query(prevent(query), runtime, *args, **kwargs)


def build_pipeline(args, llm, system_message):
    pipeline = AgentPipeline(
        [system_message, InstructionalInitQuery(), llm, ToolsExecutionLoop([ToolsExecutor(), llm])]
    )
    return pipeline, None
