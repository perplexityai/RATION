from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence


TOOL_FILTER_PROMPT = (
    "Your task is to filter the list of tools to only include those that are relevant to the user's task."
    "Output the comma-separated list containing the names of the necessary tools."
    "Do not write tool names that do not exist."
)


INVALID_TOOL_TEMPLATE = "Invalid tool {function} provided."


def tool_names(tools: Optional[Sequence[Any]]) -> list[str]:
    out: list[str] = []
    for t in tools or []:
        name = None
        if isinstance(t, dict):
            fn = t.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
            if not name:
                name = t.get("name")
        else:
            fn = getattr(t, "function", None)
            name = getattr(fn, "name", None) or getattr(t, "name", None)
        if isinstance(name, str) and name:
            out.append(name)
    return out


def select_tools(reply_text: Optional[str], names: Iterable[str]) -> list[str]:
    text = reply_text or ""
    return [n for n in names if n in text]


def filter_tools_payload(tools: Optional[Sequence[Any]], keep: Iterable[str]) -> list[Any]:
    keep_set = set(keep)
    kept = []
    for t in tools or []:
        n = tool_names([t])
        if n and n[0] in keep_set:
            kept.append(t)
    return kept


def filter_request_messages(messages: Sequence[Any]) -> list[Any]:
    return [*messages, {"role": "user", "content": TOOL_FILTER_PROMPT}]


def _role(message: Any) -> Optional[str]:
    return message.get("role") if isinstance(message, dict) else getattr(message, "role", None)


def agent_messages_with_filter_turn(messages: Sequence[Any], reply_text: str) -> list[Any]:
    msgs = list(messages)
    pair = [{"role": "user", "content": TOOL_FILTER_PROMPT},
            {"role": "assistant", "content": reply_text}]
    first_user = next((i for i, m in enumerate(msgs) if _role(m) == "user"), None)
    at = (first_user + 1) if first_user is not None else len(msgs)
    return [*msgs[:at], *pair, *msgs[at:]]
