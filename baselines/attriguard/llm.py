from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional


class ProxyChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 480.0,
        attempts: int = 3,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.attempts = attempts
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": 0}

    def chat(self, messages: list[dict], tools: Optional[list] = None,
             logprobs: bool = False, top_logprobs: int = 5,
             extra: Optional[dict] = None) -> tuple[str, list[dict], Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if tools:
            body["tools"] = tools
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        for k, v in (extra or {}).items():
            if v is not None:
                body[k] = v
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        last_exc: Exception | None = None
        attempted = 0
        for attempt in range(self.attempts):
            attempted += 1
            req = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=data,
                headers={"content-type": "application/json",
                         "authorization": "Bearer " + (os.environ.get("OPENAI_API_KEY") or "proxy")},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.load(r)
                self.usage["calls"] += 1
                u = resp.get("usage") or {}
                self.usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                self.usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                choice = (resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                content = msg.get("content")
                if content is None:
                    content = ""
                elif not isinstance(content, str):
                    content = str(content)
                calls = []
                for tc in msg.get("tool_calls") or []:
                    fn = (tc.get("function") or {})
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args)
                        if not isinstance(args, dict):
                            args = {"_raw_arguments": raw_args}
                    except Exception:
                        args = {"_raw_arguments": raw_args}
                    calls.append({"id": tc.get("id"), "function": fn.get("name"), "args": args})
                return content, calls, choice.get("logprobs")
            except Exception as exc:
                last_exc = exc
                self.usage["errors"] += 1
                if isinstance(exc, urllib.error.HTTPError) and exc.code in {400, 422}:
                    break
                if attempt + 1 < self.attempts:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"attriguard llm call failed after {attempted} attempt(s): {last_exc}")
