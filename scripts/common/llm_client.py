from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DialogueEntry = dict[str, Any]


OPENAI_MODEL_IDS: dict[str, str] = {
    "gpt_4_1": "gpt-4.1",
    "gpt_4_1_mini": "gpt-4.1-mini",
    "gpt_4o": "gpt-4o",
    "gpt_4o_mini": "gpt-4o-mini",
    "gpt_5_4": "gpt-5.4",
    "gpt_5_4_mini": "gpt-5.4-mini",
}


def openai_model_id(model: str) -> str:
    if model in OPENAI_MODEL_IDS:
        return OPENAI_MODEL_IDS[model]
    if "-" in model or "." in model:
        return model
    raise ValueError(f"unknown model id {model!r}; register it in OPENAI_MODEL_IDS")


@dataclass
class Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: str = "{}"


@dataclass
class Usage:
    total_input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Response:
    content: list[Block] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""


def _coerce_int(value: Any) -> int:
    try:
        return 0 if value is None else int(value)
    except (TypeError, ValueError):
        return 0


def usage_from_response(response: Any, model: str) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": _coerce_int(getattr(usage, "total_input_tokens", None)),
        "output_tokens": _coerce_int(getattr(usage, "output_tokens", None)),
        "model": model,
    }


def usage_from_openai(usage: Any) -> Usage:

    def get(obj: Any, name: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, Mapping):
            return obj.get(name)
        return getattr(obj, name, None)

    return Usage(
        total_input_tokens=_coerce_int(get(usage, "prompt_tokens")),
        output_tokens=_coerce_int(get(usage, "completion_tokens")),
    )


def _image_part(block: Mapping[str, Any]) -> dict[str, Any]:
    if block.get("url"):
        url = str(block["url"])
    else:
        media = block.get("media_type") or "image/png"
        url = f"data:{media};base64,{block.get('data') or ''}"
    return {"type": "image_url", "image_url": {"url": url}}


def _content_parts(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for block in blocks:
        if block["type"] == "text":
            if block.get("text"):
                parts.append({"type": "text", "text": block["text"]})
        elif block["type"] == "image":
            parts.append(_image_part(block))
        else:
            raise ValueError(f"unsupported content block type: {block['type']!r}")
    return parts


def _tool_input_json(raw: Any) -> str:
    if isinstance(raw, Mapping):
        return json.dumps(dict(raw))
    if raw is None:
        return "{}"
    return str(raw)


def build_messages(
    system_prompt: str | None, dialogue: Sequence[DialogueEntry]
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    for entry in dialogue:
        blocks = list(entry["content"])
        if entry["role"] == "assistant":
            text_parts = [b["text"] for b in blocks if b["type"] == "text" and b.get("text")]
            tool_calls = [
                {
                    "id": b.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": b.get("name", ""), "arguments": _tool_input_json(b.get("input"))},
                }
                for b in blocks
                if b["type"] == "tool_use"
            ]
            message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            if message["content"] is None and not tool_calls:
                continue
            messages.append(message)
            continue
        results = [b for b in blocks if b["type"] == "tool_result"]
        rest = [b for b in blocks if b["type"] in ("text", "image")]
        for result in results:
            inner = list(result.get("content") or [])
            text = "".join(str(b.get("text", "")) for b in inner if b["type"] == "text")
            messages.append(
                {"role": "tool", "tool_call_id": result["tool_use_id"], "content": text}
            )
            images = [b for b in inner if b["type"] == "image"]
            if images:

                rest = [*images, *rest]
        parts = _content_parts(rest)
        if parts:
            if all(part["type"] == "text" for part in parts):
                messages.append({"role": "user", "content": "".join(p["text"] for p in parts)})
            else:
                messages.append({"role": "user", "content": parts})
    return messages


def build_tools(tool_dicts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tools = []
    for tool in tool_dicts:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or f"Tool '{tool['name']}'.",
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def build_tool_choice(tool_choice: str | Mapping[str, Any] | None) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, Mapping):
        if tool_choice.get("type") == "function":
            return dict(tool_choice)
        name = tool_choice.get("name") or (tool_choice.get("function") or {}).get("name")
        return {"type": "function", "function": {"name": name}}
    if tool_choice in ("auto", "none", "required"):
        return tool_choice
    if tool_choice == "any":
        return "required"
    return {"type": "function", "function": {"name": str(tool_choice)}}


def response_from_openai(completion: Any) -> Response:

    def get(obj: Any, name: str) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(name)
        return getattr(obj, name, None)

    choices = get(completion, "choices") or []
    if not choices:
        raise RuntimeError("OpenAI response carries no choices")
    choice = choices[0]
    message = get(choice, "message")
    blocks: list[Block] = []
    text = get(message, "content")
    if isinstance(text, list):
        text = "".join(str(get(p, "text") or "") for p in text)
    if text:
        blocks.append(Block(type="text", text=str(text)))
    for call in get(message, "tool_calls") or []:
        function = get(call, "function")
        blocks.append(
            Block(
                type="tool_use",
                id=str(get(call, "id") or ""),
                name=str(get(function, "name") or ""),
                input=str(get(function, "arguments") or "{}"),
            )
        )
    return Response(
        content=blocks,
        usage=usage_from_openai(get(completion, "usage")),
        finish_reason=str(get(choice, "finish_reason") or ""),
    )


def _env_float(name: str, default: float) -> float:
    try:
        value = os.environ.get(name)
        return default if value in (None, "") else float(value)
    except ValueError:
        return default


def sampling_from_environment() -> dict[str, Any]:
    sampling: dict[str, Any] = {}
    temperature = (os.environ.get("LLM_TEMPERATURE") or "").strip()
    if temperature:
        sampling["temperature"] = float(temperature)
    cap = (os.environ.get("LLM_MAX_TOKENS") or "").strip().lower()
    if cap and cap not in {"0", "none", "unset"}:
        sampling["max_completion_tokens"] = int(cap)
    effort = (os.environ.get("LLM_REASONING_EFFORT") or "").strip().lower()
    if effort and effort not in {"none", "default", "provider_default"}:
        sampling["reasoning_effort"] = effort
    return sampling


_CLIENT_LOCK = threading.Lock()
_CLIENT: Any = None


def openai_client() -> Any:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            import openai

            _CLIENT = openai.OpenAI(
                timeout=_env_float("LLM_REQUEST_TIMEOUT", 600.0), max_retries=0
            )
    return _CLIENT


def is_retryable(exc: BaseException) -> bool:
    try:
        import openai
    except ImportError:
        return False
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                        openai.InternalServerError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


def retry_after_seconds(exc: BaseException, attempt: int, base: float = 2.0) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        value = headers.get("retry-after") if hasattr(headers, "get") else None
        if value:
            return max(0.5, float(value))
    except (TypeError, ValueError):
        pass
    return min(base * 2**attempt, 60.0)


class LLMClient:

    def __init__(
        self,
        model: str,
        *,
        timeout: float | None = None,
        max_retries: int = 4,
    ):
        self.model = model
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))

    def request(
        self,
        system_prompt: str | None,
        dialogue: Sequence[DialogueEntry],
        tool_dicts: Sequence[Mapping[str, Any]],
        tool_choice: str | Mapping[str, Any] | None = "auto",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": openai_model_id(self.model),
            "messages": build_messages(system_prompt, dialogue),
            **sampling_from_environment(),
        }
        if tool_dicts:
            body["tools"] = build_tools(tool_dicts)
            choice = build_tool_choice(tool_choice)
            if choice is not None:
                body["tool_choice"] = choice
        return body

    def call(
        self,
        system_prompt: str | None,
        dialogue: Sequence[DialogueEntry],
        tool_dicts: Sequence[Mapping[str, Any]],
        *,
        tool_choice: str | Mapping[str, Any] | None = "auto",
    ) -> Response:
        body = self.request(system_prompt, dialogue, tool_dicts, tool_choice)
        client = openai_client()
        timeout = self.timeout or _env_float("LLM_REQUEST_TIMEOUT", 600.0)
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                completion = client.chat.completions.create(**body, timeout=timeout)
            except Exception as exc:
                if not is_retryable(exc):
                    raise
                last = exc
                if attempt + 1 < self.max_retries:
                    delay = retry_after_seconds(exc, attempt)
                    logger.warning(
                        "transient OpenAI failure (attempt %d/%d): %s: %s; retrying in %.1fs",
                        attempt + 1, self.max_retries, type(exc).__name__, exc, delay,
                    )
                    time.sleep(delay)
                continue
            return response_from_openai(completion)
        raise last
