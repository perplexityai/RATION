#!/usr/bin/env python3

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from typing import Any

import openai


def _emit(payload: dict[str, Any]) -> None:
    with open(os.environ["CAMEL_ASB_RESPONSE"], "w") as handle:
        json.dump(payload, handle)


def _fail(message: str) -> None:
    _emit({"ok": False, "error": message})


def _make_tool(name: str, description: str, achievement: str, injection: str | None, calls: list[dict[str, Any]]):
    def tool() -> str:
        text = achievement if injection is None else f"{achievement}; {injection}"
        calls.append({"tool": name, "returned": text})
        return text

    tool.__name__ = name
    tool.__qualname__ = name
    tool.__doc__ = description
    return tool


def _message_to_plain(message: Any) -> dict[str, Any]:
    role = message.get("role")
    content = message.get("content")
    if isinstance(content, list):
        text = "".join(
            block.get("content", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    elif content is None:
        text = None
    else:
        text = str(content)
    plain: dict[str, Any] = {"role": role, "content": text}
    tool_calls = message.get("tool_calls")
    if tool_calls:
        plain["tool_calls"] = [
            {"function": getattr(call, "function", None) or call.get("function"), "args": _plain_args(call)}
            for call in tool_calls
        ]
    tool_call = message.get("tool_call")
    if tool_call is not None:
        plain["tool_call"] = {"function": tool_call.function, "args": _plain_args(tool_call)}
    if message.get("error") is not None:
        plain["error"] = str(message.get("error"))
    return plain


def _plain_args(call: Any) -> Any:
    args = getattr(call, "args", None)
    if args is None and isinstance(call, dict):
        args = call.get("args")
    try:
        json.dumps(args)
        return args
    except TypeError:
        return {str(k): str(v) for k, v in (args or {}).items()}


def _install_runnability_patch() -> None:
    from camel.interpreter import interpreter as interp
    from camel.interpreter import result

    original = interp._eval_constant

    def _eval_constant(node, namespace, tool_calls_chain, dependencies, eval_args):
        try:
            return original(node, namespace, tool_calls_chain, dependencies, eval_args)
        except AttributeError as exc:
            if "__type__" not in str(exc):
                raise
            return interp.EvalResult(
                result.Error(
                    interp.CaMeLException(
                        NotImplementedError(f"unsupported constant type {type(node.value).__name__}"),
                        (node,),
                        (),
                    )
                ),
                namespace,
                tool_calls_chain,
                dependencies,
            )

    interp._eval_constant = _eval_constant


def main() -> int:
    request = json.load(sys.stdin)
    try:
        importlib.import_module("agentdojo.task_suite.load_suites")
        from agentdojo.agent_pipeline import OpenAILLM
        from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
        from camel.interpreter.interpreter import MetadataEvalMode
        from camel.pipeline_elements.privileged_llm import PrivilegedLLM
        from camel.pipeline_elements.security_policies import ADNoSecurityPolicyEngine
    except Exception as exc:
        _fail(f"import failure: {type(exc).__name__}: {exc}")
        return 3

    _install_runnability_patch()
    model = str(request["request_model"])
    calls: list[dict[str, Any]] = []
    runtime = FunctionsRuntime()
    injection = request.get("injection")
    for spec in request["tools"]:
        runtime.register_function(
            _make_tool(spec["name"], spec["description"], spec["returns"], injection, calls)
        )

    client = openai.OpenAI()
    llm = OpenAILLM(client, model, None, temperature=0.0)
    eval_mode = MetadataEvalMode[request["eval_mode"]]
    element = PrivilegedLLM(
        llm,
        ADNoSecurityPolicyEngine,
        f"openai:{model}",
        eval_mode=eval_mode,
        quarantined_llm_retries=int(request["quarantined_retries"]),
        max_attempts=int(request["max_attempts"]),
    )
    try:
        _query, _runtime, _env, messages, _extra = element.query(str(request["task"]), runtime, EmptyEnv())
    except Exception as exc:
        _fail(f"camel raised {type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}")
        return 4
    plain_messages = [_message_to_plain(message) for message in messages]
    programs = [
        m["content"]
        for m in plain_messages
        if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].lstrip().startswith("```")
    ]
    _emit({"ok": True, "messages": plain_messages, "tool_calls": calls, "programs": programs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
