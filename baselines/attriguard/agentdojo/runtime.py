from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, InitQuery

from baselines.attriguard.agentdojo.config import (
    ATTENUATION_LEVEL,
    MAX_ITERS,
    SKIP_EMPTY_TOOL_RESULTS_AUDIT,
    SURVIVAL_MODE,
)
from baselines.attriguard.agentdojo.official import AttriGuardExecutionLoop


def build_pipeline(args, llm, system_message):
    loop = AttriGuardExecutionLoop(
        llm,
        judge_llm=llm,
        attenuation_llm=llm,
        attenuation_level=ATTENUATION_LEVEL,
        survival_mode=SURVIVAL_MODE,
        max_iters=MAX_ITERS,
        skip_empty_tool_results_audit=SKIP_EMPTY_TOOL_RESULTS_AUDIT,
    )
    return AgentPipeline([system_message, InitQuery(), llm, loop]), None
