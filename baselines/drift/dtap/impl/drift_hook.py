from __future__ import annotations

from typing import Any, Optional

import drift_runtime as dr


class DriftBlocked(RuntimeError):
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


def set_result_text(raw: Any, text: str) -> bool:
    content = getattr(raw, "content", None)
    if isinstance(content, list):
        for b in content:
            if isinstance(getattr(b, "text", None), str):
                b.text = text
                return True
    return False


class DriftHook:
    async def on_pre_tool_call(self, ctx):
        tool = getattr(ctx, "tool_name", None)
        args = dict(getattr(ctx, "arguments", None) or {})
        verdict = await dr.audit_call(tool, args)
        if verdict is None:
            return None
        if verdict["disposition"] != "allow":
            raise DriftBlocked(verdict["text"])
        return None

    async def on_post_tool_call(self, ctx, result):
        tool = getattr(ctx, "tool_name", None)
        args = dict(getattr(ctx, "arguments", None) or {})
        text = result_text(getattr(result, "raw", None))
        if text is None:
            return result
        cleaned = await dr.isolate_result(tool, args, text)
        if cleaned is not None and cleaned != text:
            set_result_text(result.raw, cleaned)
        return result
