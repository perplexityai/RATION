from pathlib import Path

import dt_arena.src.types.hooks as H
from dt_arena.src.types.hooks import HookManager
from mcp.types import CallToolResult, TextContent
from openai.resources.chat import completions as _cc


def _patch_hook_config_path() -> None:
    ours = Path(__file__).resolve().parent / "hooks.json"
    if not ours.is_file():
        raise FileNotFoundError(f"MELON hooks file is missing: {ours}")
    H._HOOKS_CONFIG_PATH = ours


def _patch_agent_call_capture() -> None:
    _orig_create = _cc.AsyncCompletions.create

    async def create(self, *a, **kw):
        import melon_runtime as ar
        if ar.is_agent_request(kw) and ar.incomplete() is not None:
            raise ar.MelonIncomplete(
                f"MELON task is incomplete ({ar.incomplete()}); aborting the agent run "
                "so the task is retried, never scored"
            )
        resp = await _orig_create(self, *a, **kw)
        ar.maybe_capture(kw, resp)
        return resp

    _cc.AsyncCompletions.create = create


def _patch_melon_block_delivery() -> None:
    import melon_hook

    orig_wrap = HookManager.wrap

    async def wrap(self, ctx, call):
        try:
            return await orig_wrap(self, ctx, call)
        except melon_hook.MelonBlocked as blk:
            return CallToolResult(content=[TextContent(type="text", text=str(blk))],
                                  isError=False)

    HookManager.wrap = wrap


_patch_hook_config_path()
_patch_agent_call_capture()
_patch_melon_block_delivery()
