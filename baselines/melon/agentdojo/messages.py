from __future__ import annotations

import json
from typing import Any, Sequence

from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

from baselines.melon.prompts import (
    FEWSHOT_MESSAGES,
    MASKED_READ_CALL_ID,
    TASK_NEUTRAL_PROMPT,
    fence_tool_output,
)


def _function_call(wire_call: dict[str, Any]) -> FunctionCall:
    function = wire_call["function"]
    arguments = function.get("arguments")
    args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
    return FunctionCall(function=function["name"], args=args, id=wire_call["id"], placeholder_args=None)


def wire_to_agentdojo(messages: Sequence[dict[str, Any]]) -> list[ChatMessage]:
    rendered: list[ChatMessage] = []
    pending_calls: dict[str, FunctionCall] = {}
    for message in messages:
        role = message["role"]
        text = message.get("content") or ""
        if role == "user":
            rendered.append(ChatUserMessage(role="user", content=[text_content_block_from_string(text)]))
        elif role == "assistant":
            calls = [_function_call(call) for call in message.get("tool_calls") or []]
            for call in calls:
                pending_calls[str(call.id)] = call
            rendered.append(
                ChatAssistantMessage(
                    role="assistant",
                    content=[text_content_block_from_string(text)],
                    tool_calls=calls or None,
                )
            )
        elif role == "tool":
            call_id = message["tool_call_id"]
            rendered.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(text)],
                    tool_call_id=call_id,
                    tool_call=pending_calls[call_id],
                    error=None,
                )
            )
        else:
            raise ValueError(f"unexpected prompt message role {role!r}")
    return rendered


def fewshot_messages() -> list[ChatMessage]:
    return wire_to_agentdojo(FEWSHOT_MESSAGES)


def tool_message_text(message: ChatMessage) -> str:
    return get_text_content_as_str(message.get("content") or [])


def build_masked_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    tool_contents = ""
    masked: list[ChatMessage] = []
    for message in messages:
        if message["role"] == "system":
            masked.append(message)
        elif message["role"] == "tool":
            call = message.get("tool_call")
            function = getattr(call, "function", None)
            if function is None:
                raise ValueError("tool result without its tool_call cannot be fenced (pi_detector.py:324)")
            tool_contents += fence_tool_output(function, tool_message_text(message))
    masked += fewshot_messages()
    read_call = FunctionCall(function="read_file", args={"name": "random.txt"}, id=MASKED_READ_CALL_ID, placeholder_args=None)
    masked += [
        ChatUserMessage(role="user", content=[text_content_block_from_string(TASK_NEUTRAL_PROMPT)]),
        ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("")], tool_calls=[read_call]),
        ChatToolResultMessage(
            role="tool",
            content=[text_content_block_from_string(tool_contents)],
            tool_call_id=MASKED_READ_CALL_ID,
            tool_call=read_call,
            error=None,
        ),
    ]
    return masked
