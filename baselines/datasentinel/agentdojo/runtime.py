from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, InitQuery
from agentdojo.agent_pipeline.pi_detector import PromptInjectionDetector
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor

from baselines.datasentinel import core


class DataSentinelDetector(PromptInjectionDetector):

    def __init__(self, client: core.DetectorClient) -> None:
        super().__init__(mode="message", raise_on_injection=False)
        self._client = client

    def detect(self, tool_output: str) -> bool:
        if tool_output == "":
            return False
        return self._client.detect(tool_output, max_prompt_tokens=None).contaminated


def build_pipeline(args, llm, system_message):
    detector = DataSentinelDetector(core.DetectorClient())
    pipeline = AgentPipeline(
        [
            system_message,
            InitQuery(),
            llm,
            ToolsExecutionLoop([ToolsExecutor(), detector, llm]),
        ]
    )
    return pipeline, detector
