from __future__ import annotations

import promptguard_runtime as pr


class PromptGuardHook:
    async def on_post_tool_call(self, ctx, result):
        return await pr.inspect_result(result, tool=getattr(ctx, "tool_name", None))
