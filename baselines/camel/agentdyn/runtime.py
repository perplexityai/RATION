from __future__ import annotations

import atexit
import os
import re
import threading
from http.server import ThreadingHTTPServer
from typing import Any, Sequence

from agentdojo.agent_pipeline import AgentPipeline, InitQuery
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str

from scripts.agentdojo.openai_llm import ChatLLM
from scripts.common.openai_proxy import Handler

QUARANTINED_MODEL = "openai:gpt-4.1"
QUARANTINED_RETRIES = 10
MAX_ATTEMPTS = 10
PROXY_ENV = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "PROXY_FORCE_MODEL")
ERROR_PREFIX = "Running the code gave the following error:"
TRANSPORT_FAILURE_CLASSES = frozenset(
    {
        "ModelHTTPError",
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "InternalServerError",
        "RateLimitError",
        "APIError",
    }
)
_EXCEPTION_LINE = re.compile(r"^([A-Za-z_][\w.]*): ", re.MULTILINE)


def install_runnability_patch() -> None:
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


def _message_text(message: ChatMessage) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return get_text_content_as_str(content)
    return ""


def transport_failures(messages: Sequence[ChatMessage]) -> list[str]:
    failures = []
    for index, message in enumerate(messages):
        text = _message_text(message)
        if message.get("role") != "user" or index == 0 or not text.startswith(ERROR_PREFIX):
            continue
        head, _, _tail = text.partition("Provide the new code with the error fixed")
        matches = _EXCEPTION_LINE.findall(head)
        if matches and matches[-1] in TRANSPORT_FAILURE_CLASSES:
            failures.append(matches[-1])
    return failures


class TransportFailureCheck(BasePipelineElement):

    def __init__(self) -> None:
        self.failures: list[str] = []

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict | None = None,
    ):
        self.failures.extend(transport_failures(messages))
        return query, runtime, env, messages, dict(extra_args or {})


def _start_proxy(model: str) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, name="camel-agentdyn-openai-proxy", daemon=True)
    thread.start()
    os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    os.environ.setdefault("OPENAI_API_KEY", "proxy")
    os.environ["PROXY_FORCE_MODEL"] = model
    return server, thread


def _close_transport(server: ThreadingHTTPServer, thread: threading.Thread, previous_env: dict[str, str | None]) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    for name, value in previous_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def build_pipeline(args: Any, config: dict[str, Any]) -> tuple[AgentPipeline, TransportFailureCheck]:
    from camel.interpreter.interpreter import MetadataEvalMode
    from camel.pipeline_elements.privileged_llm import PrivilegedLLM
    from camel.pipeline_elements.security_policies import ADNoSecurityPolicyEngine

    model = config["models"]["agent"]["id"]
    previous_env = {name: os.environ.get(name) for name in PROXY_ENV}
    server, thread = _start_proxy(model)
    install_runnability_patch()
    element = PrivilegedLLM(
        ChatLLM(model),
        ADNoSecurityPolicyEngine,
        QUARANTINED_MODEL,
        eval_mode=MetadataEvalMode.NORMAL,
        quarantined_llm_retries=QUARANTINED_RETRIES,
        max_attempts=MAX_ATTEMPTS,
    )
    check = TransportFailureCheck()
    pipeline = AgentPipeline([InitQuery(), element, check])
    atexit.register(_close_transport, server, thread, previous_env)
    return pipeline, check


def finish_unit(check: TransportFailureCheck) -> None:
    failures, check.failures = check.failures, []
    if failures:
        raise RuntimeError(
            f"CaMeL quarantined-LLM transport failed {len(failures)} time(s) "
            f"({', '.join(sorted(set(failures)))}); unit left unmeasured"
        )
