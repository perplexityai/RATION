from __future__ import annotations

import os
import time
from typing import Optional

EMBED_MODEL = "text-embedding-3-large"
OPENAI_API_URL = "https://api.openai.com/v1"
THETA = 0.8
_RETRIES = 4
_TIMEOUT_S = 120.0


class OpenAIEmbedder:

    def __init__(self, url: Optional[str] = None):
        self.url = url or os.environ.get("MELON_EMBED_URL") or OPENAI_API_URL
        self.model = EMBED_MODEL
        self.usage = {"requests": 0, "inputs": 0, "errors": 0}
        self._client = None

    def _openai(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                base_url=self.url.removesuffix("/embeddings"), timeout=_TIMEOUT_S, max_retries=0
            )
        return self._client

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        last: Exception | None = None
        for attempt in range(_RETRIES + 1):
            try:
                response = self._openai().embeddings.create(model=self.model, input=list(texts))
                self.usage["requests"] += 1
                self.usage["inputs"] += len(texts)
                rows = sorted(response.data, key=lambda row: row.index)
                return [list(row.embedding) for row in rows]
            except Exception as exc:
                last = exc
                self.usage["errors"] += 1
                if attempt < _RETRIES:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"melon embeddings failed after {_RETRIES + 1} attempts: {last}")
