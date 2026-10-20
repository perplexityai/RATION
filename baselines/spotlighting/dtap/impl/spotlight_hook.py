from __future__ import annotations

import spotlight_runtime as sr


class SpotlightHook:
    async def on_post_tool_call(self, ctx, result):
        return sr.wrap_result(result, tool=getattr(ctx, "tool_name", None))
