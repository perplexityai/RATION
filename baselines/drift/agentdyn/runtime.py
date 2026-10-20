from __future__ import annotations

import atexit
from pathlib import Path

import agentdojo
from agentdojo.agent_pipeline import AgentPipeline, InitQuery, ToolsExecutor
from agentdojo.agent_pipeline.agent_pipeline import DriftMessageAdapter

from baselines.drift.agentdojo import runtime as drift


class FailureCountingClient(drift.FailureCountingClient):

    def agent_run(self, messages, *args, **kwargs):
        result = self.delegate.agent_run(messages, *args, **kwargs)
        if isinstance(result, list) and result and result[0] == drift.FAILED_GENERATION:
            self.failures += 1
        return result


def build_pipeline(args, config):
    model = config["models"]["agent"]["id"]
    modules = drift.load_official(Path(agentdojo.__file__).resolve().parent / "defenses/drift")
    logger = drift.official_logger()
    server, previous_env = drift.start_proxy(model)
    client = FailureCountingClient(modules["client"].OpenAIModel(model=model, logger=logger))
    drift_llm = modules["DRIFTLLM"].DRIFTLLM(drift.DRIFT_ARGS, client=client, model=model, logger=logger)
    adapter = DriftMessageAdapter(drift_llm)
    loop = modules["DRIFTToolsExecutionLoop"].DRIFTToolsExecutionLoop([ToolsExecutor(), adapter])
    state = drift.RuntimeState(client, server, previous_env)
    atexit.register(drift.finish_taskset, state)
    return AgentPipeline([InitQuery(), adapter, loop]), state


finish_unit = drift.finish_unit
