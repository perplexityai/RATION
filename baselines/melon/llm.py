from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional


class ProxyChatClient:
    def __init__(self, base_url: str, model: str, timeout: float = 480.0, retries: int = 3,
                 temperature: Optional[float] = None, top_p: Optional[float] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.top_p = top_p
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": 0}

    def chat(self, messages: list[dict], tools: Optional[list] = None,
             logprobs: bool = False, top_logprobs: int = 5,
             extra: Optional[dict] = None) -> tuple[str, list[dict], Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if tools:
            body["tools"] = tools
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        for k, v in (extra or {}).items():
            if v is not None:
                body[k] = v
        data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
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
                content = (msg.get("content") or "").strip()
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
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", "replace")[:400]
                except Exception:
                    pass
                last_exc = f"{exc} | body: {body}"
                self.usage["errors"] += 1
                if attempt < self.retries:
                    time.sleep(2 * (attempt + 1))
            except Exception as exc:
                last_exc = exc
                self.usage["errors"] += 1
                if attempt < self.retries:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"melon masked-run llm call failed after {self.retries + 1} attempts: {last_exc}")
