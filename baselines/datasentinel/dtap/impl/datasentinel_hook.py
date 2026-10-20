from __future__ import annotations

import datasentinel_runtime as dsr


class DataSentinelHook:
    async def on_post_tool_call(self, ctx, result):
        return await dsr.inspect_result(result, tool=getattr(ctx, "tool_name", None))
