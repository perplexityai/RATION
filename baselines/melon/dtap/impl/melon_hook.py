from __future__ import annotations

import melon_runtime as mr


class MelonBlocked(RuntimeError):
    pass


class MelonHook:
    async def on_pre_tool_call(self, ctx):
        tool = getattr(ctx, "tool_name", None)
        args = dict(getattr(ctx, "arguments", None) or {})
        verdict = await mr.audit_call(tool, args)
        if verdict is not None and not verdict.get("allow"):
            raise MelonBlocked(mr.blocked_text())
        return None

    async def on_post_tool_call(self, ctx, result):
        return result
