from __future__ import annotations

import attriguard_runtime as ar


class AttriGuardBlocked(RuntimeError):
    pass


def blocked_tool_result(message: str):
    from mcp.types import CallToolResult, TextContent

    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
    )


def install_block_delivery(manager_class) -> None:
    if getattr(manager_class, "_attriguard_block_delivery", False):
        return
    original = manager_class.wrap

    async def wrap(self, ctx, call):
        try:
            return await original(self, ctx, call)
        except AttriGuardBlocked as blocked:
            return blocked_tool_result(str(blocked))

    manager_class.wrap = wrap
    manager_class._attriguard_block_delivery = True


class AttriGuardHook:
    async def on_pre_tool_call(self, ctx):
        tool = getattr(ctx, "tool_name", None)
        args = dict(getattr(ctx, "arguments", None) or {})
        verdict = await ar.audit_call(tool, args)
        if verdict is not None and not verdict.get("allow"):
            raise AttriGuardBlocked(ar.blocked_text(tool, args))
        return None

    async def on_post_tool_call(self, ctx, result):
        tool = getattr(ctx, "tool_name", None)
        args = dict(getattr(ctx, "arguments", None) or {})
        ar.record_tool_result(tool, args, getattr(result, "raw", result))
        return result
