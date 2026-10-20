from __future__ import annotations

import pidetector_runtime as dr


class PIDetectorHook:
    async def on_post_tool_call(self, ctx, result):
        return await dr.inspect_result(result, tool=getattr(ctx, "tool_name", None))
