from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Optional

from adapters.dtap import ration_runtime as rr
from ration.model import Verdict

_PROVISION_LOCK = threading.Lock()
_RESOLVE_LOCK = asyncio.Lock()


class RationBlocked(RuntimeError):
    pass


async def _provision_async():
    def run():
        with _PROVISION_LOCK:
            return rr.ensure_provisioned()

    return await asyncio.to_thread(run)


def _content_of(raw: Any):
    content = raw.get("content") if isinstance(raw, dict) else getattr(raw, "content", None)
    return content if isinstance(content, list) else None


def _block_text(block: Any) -> Optional[str]:
    text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
    return text if isinstance(text, str) else None


def _graceful_error_text(raw: Any) -> Optional[str]:
    if isinstance(raw, dict) and "content" not in raw and isinstance(raw.get("error"), str):
        return raw["error"]
    return None


def result_is_error(raw: Any) -> bool:
    if _graceful_error_text(raw) is not None:
        return True
    if isinstance(raw, dict):
        return bool(raw.get("isError") or raw.get("is_error"))
    return bool(getattr(raw, "isError", False) or getattr(raw, "is_error", False))


def _result_text(raw: Any) -> Optional[str]:
    graceful = _graceful_error_text(raw)
    if graceful is not None:
        return graceful
    content = _content_of(raw)
    if content is not None:
        parts = [text for text in (_block_text(block) for block in content) if text is not None]
        if parts:
            return "\n".join(parts)
    if isinstance(raw, str):
        return raw
    return None


def _dig(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _unhooked_call_tool(server):
    for cls in type(server).__mro__:
        if cls.__name__ == "_HookedCallToolMixin":
            return super(cls, server).call_tool
    return server.call_tool


async def _eager_resolve(monitor) -> None:
    async with _RESOLVE_LOCK:
        if getattr(monitor, "_eager_done", False):
            return
        for tool, key_param, key_value, yields_param, yields_field in monitor.plan.pending_resolutions():
            server = rr.server_for_tool(tool)
            if server is None:
                raise RuntimeError(f"declared resolver tool {tool!r} has no connected MCP server")
            result = await _unhooked_call_tool(server)(tool.split(".", 1)[-1], {key_param: key_value})
            text = _result_text(result)
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"declared resolver {tool!r} returned no textual result")
            values = _dig(json.loads(text), yields_field)
            if values is None:
                raise RuntimeError(f"declared resolver {tool!r} result lacks field {yields_field!r}")
            for value in (values if isinstance(values, (list, tuple)) else [values]):
                monitor.seed_resolved(yields_param, value)
        monitor._eager_done = True


class RationHook:
    async def on_pre_tool_call(self, ctx):
        monitor = await _provision_async()
        if monitor.plan.resolvers and not getattr(monitor, "_eager_done", False):
            await _eager_resolve(monitor)
        delta = rr.poll_token_delta()
        if delta:
            decision = monitor.record_tokens(delta)
            if decision is not None and (decision.terminal or decision.verdict is Verdict.SOFT_BREACH):
                raise RationBlocked(decision.message or "[RATION] token budget exceeded")
        decision = monitor.check_call(ctx.tool_name, dict(ctx.arguments or {}), ctx.server)
        if decision.verdict in (Verdict.REFUSE, Verdict.SOFT_BREACH) or decision.terminal:
            raise RationBlocked(decision.message or f"[RATION] blocked {ctx.tool_name}")
        rr.mark_dispatched()
        return None

    async def on_post_tool_call(self, ctx, result):
        if not result.is_error and not result_is_error(result.raw):
            rr.get_monitor().commit_success(ctx.tool_name, dict(ctx.arguments or {}), ctx.server)
        return result
