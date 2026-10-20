from __future__ import annotations

from typing import Any

import camel_runtime as cr


def result_text(raw: Any) -> str:
    content = getattr(raw, "content", None)
    if isinstance(content, list):
        parts = [getattr(b, "text", None) for b in content if isinstance(getattr(b, "text", None), str)]
        if parts:
            return "\n".join(parts)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return str(raw)


class CaMeLHook:
    async def on_pre_tool_call(self, ctx):
        tool = getattr(ctx, "tool_name", None) or ""
        args = dict(getattr(ctx, "arguments", None) or {})
        block = await cr.gate(tool, args, getattr(ctx, "trace_id", None))
        if block is not None:
            raise cr.CaMeLBlocked(block)
        return None

    async def on_post_tool_call(self, ctx, result):
        if getattr(result, "is_error", False):
            return None
        tool = getattr(ctx, "tool_name", None) or ""
        cr.record_result(tool, getattr(ctx, "trace_id", None), result_text(getattr(result, "raw", None)))
        return None
