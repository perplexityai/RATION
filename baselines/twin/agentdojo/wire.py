from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agentdojo.types import (
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)


def _text(message: ChatMessage) -> str:
    return get_text_content_as_str(message.get("content") or [])


def tool_result_text(message: ChatToolResultMessage) -> str:
    if message.get("error") is not None:
        return str(message["error"])
    return _text(message)


def to_wire(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            wire.append({"role": "system", "content": _text(message)})
        elif role == "user":
            wire.append({"role": "user", "content": _text(message)})
        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": _text(message)}
            calls = message.get("tool_calls")
            if calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function, "arguments": json.dumps(dict(call.args))},
                    }
                    for call in calls
                ]
            wire.append(entry)
        elif role == "tool":
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id"),
                    "content": tool_result_text(message),
                }
            )
        else:
            raise ValueError(f"unsupported AgentDojo message role {role!r}")
    return wire


def overlay(originals: Sequence[ChatMessage], wire: Sequence[dict[str, Any]]) -> list[ChatMessage]:
    if len(originals) != len(wire):
        raise RuntimeError(f"wire view has {len(wire)} messages, history has {len(originals)}")
    out: list[ChatMessage] = []
    for original, entry in zip(originals, wire, strict=True):
        if original["role"] != entry["role"]:
            raise RuntimeError(f"wire role {entry['role']!r} does not match history role {original['role']!r}")
        if original["role"] == "system":
            if entry["content"] == _text(original):
                out.append(original)
            else:
                out.append(ChatSystemMessage(role="system", content=[text_content_block_from_string(entry["content"])]))
        elif original["role"] == "tool":
            if entry["content"] == tool_result_text(original):
                out.append(original)
            else:
                out.append(
                    ChatToolResultMessage(
                        role="tool",
                        tool_call=original["tool_call"],
                        tool_call_id=original.get("tool_call_id"),
                        content=[text_content_block_from_string(entry["content"])],
                        error=entry["content"] if original.get("error") is not None else None,
                    )
                )
        else:
            if entry["content"] != _text(original):
                raise RuntimeError(f"the Twin rewrite changed a {original['role']} message")
            out.append(original)
    return out


def user_message(text: str) -> ChatUserMessage:
    return ChatUserMessage(role="user", content=[text_content_block_from_string(text)])


def system_message_of(text: str) -> ChatSystemMessage:
    return ChatSystemMessage(role="system", content=[text_content_block_from_string(text)])


def tool_names(wire: Sequence[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for entry in wire:
        for call in entry.get("tool_calls") or []:
            if call.get("id"):
                names[call["id"]] = (call.get("function") or {}).get("name") or "tool"
    return names
