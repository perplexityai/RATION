from __future__ import annotations

import json
import os
import time
import urllib.request


class ProxyLLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 240.0, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": 0}

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }).encode("utf-8")
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=body,
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
                return (resp["choices"][0]["message"].get("content") or "").strip()
            except Exception as exc:
                last_exc = exc
                self.usage["errors"] += 1
                if attempt < self.retries:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"drift llm call failed after {self.retries + 1} attempts: {last_exc}")
