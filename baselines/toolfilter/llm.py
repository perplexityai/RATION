from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional


UPSTREAM_CLIENT_RETRIES = 2
UPSTREAM_CLIENT_TIMEOUT_S = 600.0


class ProxyChatClient:
    def __init__(self, base_url: str, model: str, *, temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": 0}

    def chat(self, messages: list[dict], tools: Optional[list] = None,
             tool_choice: Optional[str] = None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        last_exc: Any = None
        for attempt in range(UPSTREAM_CLIENT_RETRIES + 1):
            req = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=data,
                headers={"content-type": "application/json",
                         "authorization": "Bearer " + (os.environ.get("OPENAI_API_KEY") or "proxy")},
            )
            try:
                with urllib.request.urlopen(req, timeout=UPSTREAM_CLIENT_TIMEOUT_S) as r:
                    resp = json.load(r)
                self.usage["calls"] += 1
                u = resp.get("usage") or {}
                self.usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                self.usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                choice = (resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                return msg.get("content") or ""
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:400]
                except Exception:
                    pass
                last_exc = f"{exc} | body: {detail}"
                self.usage["errors"] += 1
            except Exception as exc:
                last_exc = exc
                self.usage["errors"] += 1
            if attempt < UPSTREAM_CLIENT_RETRIES:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"tool filter selection call failed after {UPSTREAM_CLIENT_RETRIES + 1} attempts: "
            f"{last_exc}")
