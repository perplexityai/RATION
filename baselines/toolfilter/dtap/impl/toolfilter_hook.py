from __future__ import annotations

import toolfilter_runtime as tr


class ToolFilterBlocked(RuntimeError):
    pass


class ToolFilterHook:
    async def on_pre_tool_call(self, ctx):
        tool = getattr(ctx, "tool_name", None)
        refusal = tr.check_call(tool)
        if refusal:
            raise ToolFilterBlocked(refusal)
        return None

    async def on_post_tool_call(self, ctx, result):
        return result
