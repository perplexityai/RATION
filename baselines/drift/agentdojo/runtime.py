from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, ToolsExecutor
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv
from agentdojo.types import text_content_block_from_string

from scripts.common.openai_proxy import Handler

OFFICIAL = Path(__file__).with_name("official")
OFFICIAL_MODULES = ("prompts", "import_lib", "client", "DRIFTLLM", "DRIFTToolsExecutionLoop")
FAILED_GENERATION = "FAILED GENERATION."
PROXY_ENV = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "PROXY_FORCE_MODEL")
DRIFT_ARGS = SimpleNamespace(build_constraints=True, injection_isolation=True, dynamic_validation=True)


class FailureCountingClient:

    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.failures = 0

    def llm_run(self, *args, **kwargs):
        result = self.delegate.llm_run(*args, **kwargs)
        if result == FAILED_GENERATION:
            self.failures += 1
        return result

    def __getattr__(self, item):
        return getattr(self.delegate, item)


@dataclass
class RuntimeState:
    client: FailureCountingClient
    server: ThreadingHTTPServer
    previous_env: dict[str, str | None]


class ScoringWrap(BasePipelineElement):

    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args=None):
        messages = list(messages)
        last = messages[-1] if messages else {}
        if last.get("role") == "assistant" and isinstance(last.get("content"), str):
            messages[-1] = {**last, "content": [text_content_block_from_string(last["content"])]}
        return query, runtime, env, messages, dict(extra_args or {})


def load_official(directory: Path) -> dict[str, ModuleType]:
    modules = {}
    for name in OFFICIAL_MODULES:
        spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def official_logger() -> logging.Logger:
    logger = logging.getLogger("drift.official")
    logger.disabled = True
    return logger


def start_proxy(model: str) -> tuple[ThreadingHTTPServer, dict[str, str | None]]:
    previous = {name: os.environ.get(name) for name in PROXY_ENV}
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    os.environ.setdefault("OPENAI_API_KEY", "proxy")
    os.environ["PROXY_FORCE_MODEL"] = model
    return server, previous


def build_pipeline(args, llm, system_message):
    modules = load_official(OFFICIAL)
    logger = official_logger()
    server, previous_env = start_proxy(args.model)
    client = FailureCountingClient(modules["client"].OpenAIModel(model=args.model, logger=logger))
    drift_llm = modules["DRIFTLLM"].DRIFTLLM(DRIFT_ARGS, client, logger=logger)
    loop = modules["DRIFTToolsExecutionLoop"].DRIFTToolsExecutionLoop([ToolsExecutor(), drift_llm])
    state = RuntimeState(client, server, previous_env)
    return AgentPipeline([InitQuery(), drift_llm, loop, ScoringWrap()]), state


def finish_unit(state: RuntimeState) -> None:
    failures, state.client.failures = state.client.failures, 0
    if failures:
        raise RuntimeError(f"DRIFT auxiliary LLM call failed at the provider {failures} time(s)")


def finish_taskset(state: RuntimeState) -> None:
    state.server.shutdown()
    state.server.server_close()
    for name, value in state.previous_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
