from __future__ import annotations

import sandwich_runtime as sr


class SandwichHook:
    async def on_post_tool_call(self, ctx, result):
        return sr.append_reminder(result, tool=getattr(ctx, "tool_name", None))
