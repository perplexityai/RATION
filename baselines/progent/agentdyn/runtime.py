from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

from agentdojo.agent_pipeline import AgentPipeline, BasePipelineElement, PipelineConfig
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage

from baselines.progent.agentdojo.transport import StatusRecordingServer, start_proxy

SECAGENT_ENV = {
    "SECAGENT_POLICY_MODEL": "gpt-4.1-2025-04-14",
    "SECAGENT_GENERATE": "True",
    "SECAGENT_UPDATE": "False",
    "SECAGENT_IGNORE_UPDATE_ERROR": "False",
}
SECAGENT_UNSET = ("SECAGENT_POLICY_MAX_TOKENS",)
SUITE_ALLOWLISTS: dict[str, tuple[str, ...]] = {
    "shopping": (
        "get_balance",
        "get_shopping_account_user_information",
        "search_product",
        "view_order_history",
        "get_invoice",
        "view_cart",
        "cart_total",
        "view_available_discounts",
        "get_shopping_linked_unread_emails",
        "get_shopping_linked_received_emails",
        "list_directory",
        "read_file",
        "get_current_day",
        "get_day_calendar_events",
    ),
    "github": (
        "get_received_emails",
        "get_sent_emails",
        "get_github_linked_unread_emails",
        "get_logged_in_github_user_information",
        "get_github_account_user_information",
        "git_get_linked_ssh_keys",
        "list_directory",
        "read_file",
        "get_github_repository_information",
        "get_current_day",
        "get_day_calendar_events",
    ),
    "dailylife": (
        "get_balance",
        "verify_transaction",
        "get_unread_emails",
        "get_received_emails",
        "get_sent_emails",
        "search_emails",
        "list_directory",
        "read_file",
        "get_current_day",
        "get_day_calendar_events",
    ),
}


@dataclass
class RuntimeState:
    package: Any
    server: StatusRecordingServer
    suite: str
    tools_initialized: bool = False
    guards: list[Any] = field(default_factory=list)


def load_secagent() -> Any:
    import agentdojo

    for name, value in SECAGENT_ENV.items():
        os.environ[name] = value
    for name in SECAGENT_UNSET:
        os.environ.pop(name, None)
    directory = str(Path(agentdojo.__file__).resolve().parent / "defenses/progent")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import secagent

    return secagent


class ProgentBootstrap(BasePipelineElement):

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
        state = self.state
        package = state.package
        if not state.tools_initialized:
            package.update_available_tools(
                [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "args": tool.parameters.model_json_schema().get("properties", {}),
                    }
                    for tool in runtime.functions.values()
                ]
            )
            package.update_always_allowed_tools(list(SUITE_ALLOWLISTS[state.suite]))
            state.tools_initialized = True
        package.generate_security_policy(query)
        if getattr(runtime.run_function, "__func__", None) not in state.guards:
            original_run_function = runtime.run_function
            update_policy_after_calls = os.getenv("SECAGENT_UPDATE", "False").lower() == "true"

            def guarded_run_function(self_runtime, call_env: Env, function_name: str, args: dict):
                try:
                    package.check_tool_call(function_name, args)
                except Exception as e:
                    return "", str(e)
                result, error = original_run_function(call_env, function_name, args)
                if update_policy_after_calls and error is None:
                    package.generate_update_security_policy(
                        [{"name": function_name, "args": args}],
                        str([str(result)]),
                        manual_check=False,
                    )
                return result, error

            runtime.run_function = MethodType(guarded_run_function, runtime)
            state.guards.append(guarded_run_function)
        return query, runtime, env, messages, extra_args


def build_pipeline(args: Any, config: dict[str, Any]) -> tuple[AgentPipeline, RuntimeState]:
    from scripts.agentdojo.openai_llm import ChatLLM

    model = config["models"]["agent"]["id"]
    official = AgentPipeline.from_config(
        PipelineConfig(
            llm=ChatLLM(model),
            model_id=None,
            defense=None,
            tool_delimiter="tool",
            system_message_name=None,
            system_message=load_system_message(config["runtime"]["system_message"]),
            tool_output_format="yaml",
        )
    )
    system_message, init_query, llm, tools_loop = official.elements
    package = load_secagent()
    server, _thread = start_proxy(model)
    state = RuntimeState(package=package, server=server, suite=args.suite)
    pipeline = AgentPipeline([system_message, init_query, ProgentBootstrap(state), llm, tools_loop])
    return pipeline, state


def finish_unit(state: RuntimeState) -> None:
    failed = list(state.server.failed_statuses)
    state.server.failed_statuses.clear()
    if failed:
        raise RuntimeError(
            f"Progent policy call failed at the provider {len(failed)} time(s) "
            f"(HTTP {', '.join(map(str, failed))}); unit left unmeasured"
        )
