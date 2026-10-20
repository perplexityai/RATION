from __future__ import annotations

from collections.abc import Sequence

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage

from baselines.twin.agentdojo_family import RuntimeState, TwinSafeLLM, UnitState
from scripts.agentdojo.openai_llm import ChatLLM


class UnitReset(BasePipelineElement):

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        self.state.unit = UnitState()
        return query, runtime, env, messages, extra_args


def build_pipeline(args, config):
    model = config["models"]["agent"]["id"]
    llm = ChatLLM(model)
    state = RuntimeState(model=model)
    safe = TwinSafeLLM(llm, state)
    pipeline = AgentPipeline(
        [
            UnitReset(state),
            SystemMessage(load_system_message(config["runtime"]["system_message"])),
            InitQuery(),
            safe,
            ToolsExecutionLoop([ToolsExecutor(), safe]),
        ]
    )
    return pipeline, state
