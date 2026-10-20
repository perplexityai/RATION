from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Sequence

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import EMPTY_FUNCTION_NAME
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionsRuntime, make_function
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import ChatMessage, get_text_content_as_str

from baselines.progent.agentdojo.config import ALWAYS_ALLOWED, SECAGENT_ENV, SECAGENT_UNSET
from baselines.progent.agentdojo.transport import StatusRecordingServer, start_proxy

PROXY_ENV = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "PROXY_FORCE_MODEL")


@dataclass
class RuntimeState:
    package: Any
    server: StatusRecordingServer
    server_thread: threading.Thread
    previous_env: dict[str, str | None]


def load_secagent() -> Any:
    for name, value in SECAGENT_ENV.items():
        os.environ[name] = value
    for name in SECAGENT_UNSET:
        os.environ.pop(name, None)
    import secagent

    return secagent


def secured_tools(package: Any, suite) -> dict[str, Function]:
    package.reset_security_policy(include_human_policy=True)
    package.update_available_tools([])
    wrapped = {
        function.name: make_function(package.apply_secure_tool_wrapper(function.run))
        for function in suite.tools
    }
    names, allow_all = ALWAYS_ALLOWED[suite.name]
    package.update_always_allowed_tools(list(names), allow_all_no_arg_tools=allow_all)
    return wrapped


class SecuredRuntime(BasePipelineElement):

    def __init__(self, wrapped: dict[str, Function]) -> None:
        self.wrapped = wrapped

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        runtime.update_functions(dict(self.wrapped))
        return query, runtime, env, messages, extra_args


class ProgentInitQuery(InitQuery):

    def __init__(self, package: Any) -> None:
        super().__init__()
        self.package = package

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        self.package.generate_security_policy(query)
        return super().query(query, runtime, env, messages, extra_args)


class ProgentToolsExecutor(ToolsExecutor):

    def __init__(self, package: Any) -> None:
        super().__init__()
        self.package = package

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ):
        before = len(messages)
        had_calls = bool(messages) and messages[-1]["role"] == "assistant" and bool(messages[-1]["tool_calls"])
        result = super().query(query, runtime, env, messages, extra_args)
        if not had_calls:
            return result
        offered = {tool.name for tool in runtime.functions.values()}
        intermediate_calls: list[dict[str, Any]] = []
        intermediate_results: list[str] = []
        for call, tool_message in zip(messages[-1]["tool_calls"], result[3][before:]):
            if call.function == EMPTY_FUNCTION_NAME or call.function not in offered:
                continue
            intermediate_calls.append({"name": call.function, "args": call.args})
            intermediate_results.append(get_text_content_as_str(tool_message["content"]))
        if os.getenv("SECAGENT_UPDATE", "False") == "True":
            self.package.generate_update_security_policy(
                intermediate_calls, str(intermediate_results), manual_check=False
            )
        return result


def build_pipeline(args, llm, system_message):
    previous_env = {name: os.environ.get(name) for name in (*PROXY_ENV, *SECAGENT_ENV, *SECAGENT_UNSET)}
    package = load_secagent()
    server, server_thread = start_proxy(args.model)
    wrapped = secured_tools(package, get_suite(args.benchmark_version, args.suite))
    pipeline = AgentPipeline(
        [
            system_message,
            SecuredRuntime(wrapped),
            ProgentInitQuery(package),
            llm,
            ToolsExecutionLoop([ProgentToolsExecutor(package), llm]),
        ]
    )
    return pipeline, RuntimeState(package, server, server_thread, previous_env)


def prepare_unit(args, state: RuntimeState, suite, user_task: str, injection_task: str | None) -> None:
    state.package.reset_security_policy()
    state.server.failed_statuses.clear()


def finish_unit(state: RuntimeState) -> None:
    failed = state.server.failed_statuses
    if failed:
        raise RuntimeError(
            f"Progent policy call failed at the provider {len(failed)} time(s) "
            f"(HTTP {', '.join(map(str, failed))}); unit left unmeasured"
        )


def finish_taskset(state: RuntimeState) -> None:
    state.server.shutdown()
    state.server.server_close()
    state.server_thread.join(timeout=5)
    for name, value in state.previous_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
