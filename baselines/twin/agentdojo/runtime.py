from __future__ import annotations

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, ToolsExecutionLoop, ToolsExecutor

from baselines.twin.agentdojo_family import RuntimeState, TwinSafeLLM, UnitState


def build_pipeline(args, llm, system_message):
    state = RuntimeState(model=args.model)
    safe = TwinSafeLLM(llm, state)
    pipeline = AgentPipeline(
        [
            system_message,
            InitQuery(),
            safe,
            ToolsExecutionLoop([ToolsExecutor(), safe]),
        ]
    )
    return pipeline, state


def prepare_unit(args, state: RuntimeState, suite, user_task: str, injection_task: str | None) -> None:
    state.unit = UnitState()
