from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, InitQuery
from agentdojo.agent_pipeline.pi_detector import TransformersBasedPIDetector
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor

from baselines.pidetector.agentdojo.checkpoint import resolve_snapshot
from baselines.pidetector.core import SAFE_LABEL, THRESHOLD


def build_pipeline(args, llm, system_message):
    detector = TransformersBasedPIDetector(
        model_name=str(resolve_snapshot()), safe_label=SAFE_LABEL, threshold=THRESHOLD
    )
    pipeline = AgentPipeline(
        [system_message, InitQuery(), llm, ToolsExecutionLoop([ToolsExecutor(), detector, llm])]
    )
    return pipeline, None
