from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    MessageContentBlock,
    get_text_content_as_str,
    text_content_block_from_string,
)

from scripts.common.llm_client import LLMClient, usage_from_response

logger = logging.getLogger(__name__)

DialogueBlock = dict[str, Any]
DialogueEntry = dict[str, Any]

_VALID_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_TRUNCATE_RAW_ARGS_AT = 400


def _assistant_message_to_blocks(message: ChatAssistantMessage) -> list[DialogueBlock]:
    blocks: list[DialogueBlock] = []
    for content_block in message["content"] or []:
        if content_block["type"] == "text":
            if content_block["content"]:
                blocks.append({"type": "text", "text": content_block["content"]})
        else:
            logger.debug("Dropping assistant content block of type %r", content_block["type"])
    for tool_call in message["tool_calls"] or []:
        if tool_call.id is None:
            raise ValueError("Tool call ID is required for the OpenAI API")
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function,
                "input": json.dumps(tool_call.args),
            }
        )
    return blocks


def _message_to_dialogue_entry(message: ChatMessage) -> tuple[str, list[DialogueBlock]]:
    match message["role"]:
        case "user":
            blocks = [
                {"type": "text", "text": content_block["content"]}
                for content_block in message["content"]
                if content_block["type"] == "text" and content_block["content"]
            ]
            return "user", blocks
        case "assistant":
            return "assistant", _assistant_message_to_blocks(message)
        case "tool":
            if message["tool_call_id"] is None:
                raise ValueError("`tool_call_id` is required for the OpenAI API")
            if message["error"] is not None:
                result_text = message["error"]
            else:
                result_text = get_text_content_as_str(message["content"])
            tool_result: DialogueBlock = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": [{"type": "text", "text": result_text}] if result_text else [],
                "is_error": message["error"] is not None,
            }
            return "user", [tool_result]
        case other:
            raise ValueError(f"Invalid message role for the OpenAI API: {other}")


def messages_to_dialogue(messages: Sequence[ChatMessage]) -> tuple[str | None, list[DialogueEntry]]:
    body = list(messages)
    system_prompt: str | None = None
    if body and body[0]["role"] == "system":
        system_text = get_text_content_as_str(body[0]["content"])
        system_prompt = system_text if system_text else None
        body = body[1:]

    dialogue: list[DialogueEntry] = []
    for message in body:
        role, blocks = _message_to_dialogue_entry(message)
        if not blocks:
            continue
        if dialogue and dialogue[-1]["role"] == role:
            dialogue[-1]["content"].extend(blocks)
        else:
            dialogue.append({"role": role, "content": list(blocks)})
    return system_prompt, dialogue


def function_to_tool_dict(function: Function) -> dict[str, Any]:
    return {
        "name": function.name,
        "description": function.description,
        "input_schema": function.parameters.model_json_schema(),
    }


def _truncate(text: str, limit: int = _TRUNCATE_RAW_ARGS_AT) -> str:
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def parse_tool_args(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8", errors="replace")
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"tool arguments were not valid JSON ({exc}); raw input: {_truncate(raw)!r}"
            ) from exc
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(
            "tool arguments JSON decoded to "
            f"{type(parsed).__name__}, expected an object; raw input: {_truncate(raw)!r}"
        )
    raise TypeError(f"tool arguments had unsupported type {type(raw).__name__}")


def _generate_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


PROVIDER_CUT_FINISH_REASONS = frozenset({"content_filter"})
PROVIDER_CUT_MARKER = (
    "[provider finish_reason={reason}: the completion was ended by the provider before {n} tool call(s) "
    "were fully specified; the truncated call(s) were dropped and not executed]"
)


def finish_reason_of(response: Any) -> str:
    reason = getattr(response, "finish_reason", None)
    if reason is None:
        return ""
    return str(getattr(reason, "value", reason)).lower()


def assistant_message_from_response(response: Any) -> ChatAssistantMessage:
    content: list[MessageContentBlock] = []
    tool_calls: list[FunctionCall] = []
    reason = finish_reason_of(response)
    provider_cut = reason in PROVIDER_CUT_FINISH_REASONS
    dropped = 0
    for block in getattr(response, "content", None) or []:
        block_type = str(getattr(block, "type", "") or "")
        if block_type == "text":
            text = getattr(block, "text", "") or ""
            if text:
                content.append(text_content_block_from_string(text))
        elif block_type == "tool_use":
            name = getattr(block, "name", "") or ""
            call_id = getattr(block, "id", "") or _generate_tool_call_id()
            try:
                args = parse_tool_args(getattr(block, "input", None))
            except ValueError:
                if not provider_cut:
                    raise
                dropped += 1
                logger.warning(
                    "provider finish_reason=%s cut tool call %s(%s) mid-arguments; dropping it", reason, name, call_id
                )
                continue
            tool_calls.append(FunctionCall(function=name, args=args, id=call_id))
        else:
            logger.debug("Ignoring unsupported response block type %r", block_type)
    if dropped:
        content.append(text_content_block_from_string(PROVIDER_CUT_MARKER.format(reason=reason, n=dropped)))
    return ChatAssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls if tool_calls else None,
    )


class ChatLLM(BasePipelineElement):

    def __init__(self, model: str, *, timeout: float | None = None, max_retries: int = 4) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))

    @property
    def name(self) -> str:
        return self.model

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
        system_prompt, dialogue = messages_to_dialogue(messages)
        tool_dicts = [function_to_tool_dict(tool) for tool in runtime.functions.values()]
        response = self._call(system_prompt, dialogue, tool_dicts)
        usage = usage_from_response(response, self.model)
        output = assistant_message_from_response(response)
        if output["tool_calls"] is not None:
            output["tool_calls"] = [
                tool_call
                for tool_call in output["tool_calls"]
                if _VALID_TOOL_NAME_PATTERN.match(tool_call.function) is not None
            ]
        extra_args = dict(extra_args)
        extra_args["llm_usage"] = [*extra_args.get("llm_usage", []), usage]
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args

    def _call(
        self,
        system_prompt: str | None,
        dialogue: Sequence[DialogueEntry],
        tool_dicts: Sequence[Mapping[str, Any]],
    ) -> Any:
        return LLMClient(
            self.model, timeout=self.timeout, max_retries=self.max_retries
        ).call(system_prompt, dialogue, tool_dicts)
