from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.logging import Logger
from agentdojo.types import (
    ChatMessage,
    ChatToolResultMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)
from ast import literal_eval

from ration import feedback, folds as fold_table, policies as fold_policies
from ration.provisioner import Provisioner, ProvisionerOutcome
from ration.model import Verdict
from ration.monitor import Monitor
from ration.policy_config import DomainTable

EMPTY_FUNCTION_NAME = "<empty-function-name>"

MONITOR_KEY = "ration_monitor"
USAGE_KEY = "llm_usage"
USAGE_CURSOR_KEY = "ration_usage_cursor"


class RationInit(BasePipelineElement):

    def __init__(
        self,
        domain_table: DomainTable,
        provisioner: Provisioner,
        folds: Mapping[str, int],
        policies: Mapping[str, Any],
    ):
        self.domain_table = domain_table
        self.provisioner = provisioner
        self.folds = folds
        self.policies = policies
        self._outcome_cache: dict[tuple[str, int], ProvisionerOutcome] = {}

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if extra_args is None:
            extra_args = {}
        extra_args[USAGE_CURSOR_KEY] = len(extra_args.get(USAGE_KEY, []))
        user_task = getattr(Logger.get(), "context", {}).get("user_task_id")
        fold = fold_table.fold_of(self.folds, str(user_task))
        key = (query.strip(), fold)
        outcome = self._outcome_cache.get(key)
        if outcome is None:
            table = self.domain_table.with_policy(fold_policies.for_fold(self.policies, fold))
            outcome = self.provisioner.plan(query, table)
            self._outcome_cache[key] = outcome
        monitor = Monitor(outcome.plan)
        monitor.confirm_destructive = True
        extra_args[MONITOR_KEY] = monitor
        out_messages = list(messages)
        if out_messages and out_messages[0].get("role") == "system":
            out_messages[0] = _append_system_block(out_messages[0], feedback.request_wrapper(query))
        return query, runtime, env, out_messages, extra_args


def _error_result(tool_call, message: str) -> ChatToolResultMessage:
    return ChatToolResultMessage(
        role="tool",
        content=[text_content_block_from_string("")],
        tool_call_id=tool_call.id,
        tool_call=tool_call,
        error=message,
    )


def _append_system_block(system_msg: ChatMessage, text: str) -> ChatMessage:
    content = [*system_msg["content"], text_content_block_from_string("\n\n" + text)]
    return {**system_msg, "content": content}


def _is_reminder_block(block) -> bool:
    try:
        return get_text_content_as_str([block]).lstrip().startswith(feedback.prefix())
    except Exception:
        return False


def _set_system_reminder(system_msg: ChatMessage, reminder_text: str) -> ChatMessage:
    stripped = {**system_msg, "content": [b for b in system_msg["content"] if not _is_reminder_block(b)]}
    return _append_system_block(stripped, reminder_text)


class RationToolsExecutor(BasePipelineElement):

    def __init__(self, tool_output_formatter=tool_result_to_str) -> None:
        self.output_formatter = tool_output_formatter

    def _abort(self, message: str, messages: list[ChatMessage], env: Env) -> None:
        try:
            Logger().get().log(messages)
        except Exception:
            pass
        raise AbortAgentError(message, messages, env)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if extra_args is None:
            extra_args = {}
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args
        if messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        if messages[-1]["tool_calls"] is None or len(messages[-1]["tool_calls"]) == 0:
            return query, runtime, env, messages, extra_args

        monitor: Monitor = extra_args[MONITOR_KEY]
        usage = extra_args.get(USAGE_KEY, [])
        fresh = usage[extra_args.get(USAGE_CURSOR_KEY, 0):]
        extra_args[USAGE_CURSOR_KEY] = len(usage)
        soft_details: list[dict] = []
        n = sum(record["input_tokens"] + record["output_tokens"] for record in fresh)
        token_decision = monitor.record_tokens(n) if n else None
        if token_decision is not None:
            if token_decision.terminal:
                results = [_error_result(tc, token_decision.message) for tc in messages[-1]["tool_calls"]]
                self._abort(token_decision.message, [*messages, *results], env)
            soft_details.extend(token_decision.details)
        tool_call_results: list[ChatMessage] = []
        for tool_call in messages[-1]["tool_calls"]:
            if tool_call.function == EMPTY_FUNCTION_NAME:
                tool_call_results.append(
                    _error_result(tool_call, "Empty function name provided. Provide a valid function name.")
                )
                continue
            for arg_k, arg_v in tool_call.args.items():
                if isinstance(arg_v, str) and is_string_list(arg_v):
                    tool_call.args[arg_k] = literal_eval(arg_v)

            decision = monitor.check_call(tool_call.function, tool_call.args)
            if decision.verdict in (Verdict.REFUSE, Verdict.SOFT_BREACH):
                message = decision.message
                if decision.verdict is Verdict.SOFT_BREACH:
                    soft_details.extend(decision.details)
                    message = feedback.soft_withheld()
                tool_call_results.append(_error_result(tool_call, message))
                continue
            if decision.terminal:
                tool_call_results.append(_error_result(tool_call, decision.message))
                self._abort(decision.message, [*messages, *tool_call_results], env)

            if tool_call.function not in (tool.name for tool in runtime.functions.values()):
                tool_call_results.append(_error_result(tool_call, f"Invalid tool {tool_call.function} provided."))
                continue

            tool_call_result, error = runtime.run_function(env, tool_call.function, tool_call.args)
            if error is None:
                monitor.commit_success(tool_call.function, tool_call.args)
            formatted = self.output_formatter(tool_call_result)
            if error is None:
                formatted = feedback.wrap_tool_output(formatted, check=True)
            tool_call_results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(formatted)],
                    tool_call_id=tool_call.id,
                    tool_call=tool_call,
                    error=error,
                )
            )

        head = list(messages)
        if soft_details and head and head[0].get("role") == "system":
            current, seen = [], set()
            for d in soft_details:
                if d["risk_id"] not in seen:
                    seen.add(d["risk_id"])
                    current.append(d)
            previous = [r for r in monitor.breached_risks() if r["risk_id"] not in seen]
            head[0] = _set_system_reminder(head[0], feedback.soft_reminder(current, previous))

        return query, runtime, env, [*head, *tool_call_results], extra_args
