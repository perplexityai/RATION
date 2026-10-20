from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Callable

_ROLES = ("policy_gen", "update", "unattributed")
_lock = threading.Lock()
_usage: dict[str, dict[str, float]] = {}
_role = "unattributed"
_installed = False
_orig_sync_openai: Any = None


def _blank() -> dict[str, float]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "errors": 0}


def reset_usage() -> None:
    with _lock:
        _usage.clear()
        for r in _ROLES:
            _usage[r] = _blank()


def usage() -> dict[str, dict[str, float]]:
    with _lock:
        return {r: dict(v) for r, v in _usage.items() if v["calls"] or v["errors"]}


def set_role(role: str) -> str:
    global _role
    prev = _role
    _role = role if role in _ROLES else "unattributed"
    return prev


def _record(kind: str, latency: float, resp: Any = None) -> None:
    with _lock:
        slot = _usage.setdefault(_role, _blank())
        slot["latency_s"] += latency
        if kind == "error":
            slot["errors"] += 1
            return
        slot["calls"] += 1
        u = getattr(resp, "usage", None)
        slot["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
        slot["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)


def configure(model: str, base_url: str, *, allow_late: bool = False) -> dict[str, str]:
    if not model or not base_url:
        raise ValueError("Progent needs the agent model and the proxy URL")
    if not allow_late and "secagent.tool" in sys.modules:
        raise RuntimeError(
            "secagent.tool is already imported; SECAGENT_* env is read at import time "
            "(tool.py:41-45). Call configure() before importing secagent."
        )
    env = {
        "SECAGENT_POLICY_MODEL": model,
        "SECAGENT_GENERATE": "True",
        "SECAGENT_UPDATE": "True",
        "SECAGENT_IGNORE_UPDATE_ERROR": "True",
        "SECAGENT_TASK_TYPE": "general",
        "ENABLE_SECAGENT": "True",
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "proxy",
    }
    os.environ.pop("SECAGENT_ONLY_ALLOW_NARROW", None)
    os.environ.update(env)
    reset_usage()
    return env


def install_usage_tally(factory: Callable[..., Any] | None = None) -> None:
    global _installed, _orig_sync_openai
    if _installed:
        return
    import openai

    _orig_sync_openai = factory or openai.OpenAI

    class _Completions:
        def __init__(self, real: Any) -> None:
            self._real = real

        def create(self, *args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic()
            try:
                resp = self._real.create(*args, **kwargs)
            except Exception:
                _record("error", time.monotonic() - t0)
                raise
            _record("ok", time.monotonic() - t0, resp)
            return resp

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

    class _Chat:
        def __init__(self, real: Any) -> None:
            self._real = real
            self.completions = _Completions(real.completions)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

    class TallyingOpenAI:

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._real = _orig_sync_openai(*args, **kwargs)
            self.chat = _Chat(self._real.chat)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

    openai.OpenAI = TallyingOpenAI
    _installed = True
    reset_usage()


reset_usage()
