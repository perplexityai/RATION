from pathlib import Path


def _patch_hook_config_path() -> None:
    import dt_arena.src.types.hooks as H

    H._HOOKS_CONFIG_PATH = Path(__file__).resolve().parent / "hooks.json"


def _patch_camel_block_delivery() -> None:
    from dt_arena.src.types.hooks import HookManager
    from mcp.types import CallToolResult, TextContent
    import camel_runtime

    orig_wrap = HookManager.wrap

    async def wrap(self, ctx, call):
        try:
            return await orig_wrap(self, ctx, call)
        except camel_runtime.CaMeLBlocked as blk:
            return CallToolResult(content=[TextContent(type="text", text=str(blk))], isError=False)

    HookManager.wrap = wrap


_patch_hook_config_path()
_patch_camel_block_delivery()
