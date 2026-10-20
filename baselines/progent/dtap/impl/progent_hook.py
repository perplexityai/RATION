from __future__ import annotations

from typing import Any, Optional

import progent_runtime as pr


class ProgentBlocked(RuntimeError):
    pass


def result_text(raw: Any) -> Optional[str]:
    content = getattr(raw, "content", None)
    if isinstance(content, list):
        parts = [getattr(b, "text", None) for b in content if isinstance(getattr(b, "text", None), str)]
        if parts:
            return "\n".join(parts)
    if isinstance(raw, str):
        return raw
    return None


class ProgentHook:
    async def on_pre_tool_call(self, ctx):
        tool = getattr(ctx, "tool_name", None)
        args = dict(getattr(ctx, "arguments", None) or {})
        refusal = await pr.check_call(tool, args)
        if refusal is not None:
            raise ProgentBlocked(refusal)
        return None

    async def on_post_tool_call(self, ctx, result):
        tool = getattr(ctx, "tool_name", None)
        args = dict(getattr(ctx, "arguments", None) or {})
        text = None if getattr(result, "is_error", False) else result_text(getattr(result, "raw", None))
        await pr.settle_result(tool, args, text if text is not None else "")
        return result
