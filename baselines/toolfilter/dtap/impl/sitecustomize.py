from pathlib import Path

import dt_arena.src.types.hooks as H
from agents.tool import FunctionTool
from dt_arena.src.types.hooks import HookManager
from mcp.types import CallToolResult, TextContent
from openai.resources.chat import completions as _cc


def _patch_hook_config_path() -> None:
    ours = Path(__file__).resolve().parent / "hooks.json"
    if not ours.is_file():
        raise FileNotFoundError(f"Tool Filter hooks file is missing: {ours}")
    H._HOOKS_CONFIG_PATH = ours


def _patch_agent_request_filter() -> None:
    _orig_create = _cc.AsyncCompletions.create

    async def create(self, *a, **kw):
        import toolfilter_runtime as tr
        if tr._is_agent_request(kw):
            await tr.ensure_selection(kw)
            tr.apply(kw)
        return await _orig_create(self, *a, **kw)

    _cc.AsyncCompletions.create = create


def _patch_toolfilter_block_delivery() -> None:
    import toolfilter_hook

    orig_wrap = HookManager.wrap

    async def wrap(self, ctx, call):
        try:
            return await orig_wrap(self, ctx, call)
        except toolfilter_hook.ToolFilterBlocked as blk:
            return CallToolResult(content=[TextContent(type="text", text=str(blk))],
                                  isError=False)

    HookManager.wrap = wrap


def _patch_local_skill_tool() -> None:
    original_post_init = FunctionTool.__post_init__

    def post_init(self):
        original_post_init(self)
        if getattr(self, "name", None) != "load_skill":
            return
        original_invoke = self.on_invoke_tool

        async def invoke(context, raw_arguments):
            import toolfilter_runtime as tr

            refusal = tr.check_call("load_skill")
            if refusal:
                return refusal
            return await original_invoke(context, raw_arguments)

        self.on_invoke_tool = invoke

    FunctionTool.__post_init__ = post_init


_patch_hook_config_path()
_patch_agent_request_filter()
_patch_toolfilter_block_delivery()
_patch_local_skill_tool()
