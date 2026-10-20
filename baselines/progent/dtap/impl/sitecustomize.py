import json
from pathlib import Path

import dt_arena.src.types.hooks as H
from agent.openaisdk.src import mcp_wrapper as mw
from agents.tool import FunctionTool
from dt_arena.src.types.hooks import HookManager
from mcp.types import CallToolResult, TextContent
from openai.resources.chat import completions as _cc


def _patch_hook_config_path() -> None:
    ours = Path(__file__).resolve().parent / "hooks.json"
    if not ours.is_file():
        raise FileNotFoundError(f"Progent hooks file is missing: {ours}")
    H._HOOKS_CONFIG_PATH = ours


def _patch_tool_spec_capture() -> None:
    def _wrap(cls):
        orig = cls.list_tools

        async def list_tools(self, run_context=None, agent=None):
            tools = await orig(self, run_context, agent)
            import progent_runtime as pr
            pr.register_tools(getattr(self, "name", None), tools)
            return tools

        cls.list_tools = list_tools

    for c in (mw.MCPServerSse, mw.MCPServerStreamableHttp, mw.MCPServerStdio):
        _wrap(c)


def _patch_agent_call_capture() -> None:
    _orig_create = _cc.AsyncCompletions.create

    async def create(self, *a, **kw):
        import progent_runtime as pr
        if pr.is_agent_request(kw):
            await pr.before_agent_request()
        resp = await _orig_create(self, *a, **kw)
        pr.capture_batch(kw, resp)
        return resp

    _cc.AsyncCompletions.create = create


def _patch_progent_block_delivery() -> None:
    import progent_hook

    orig_wrap = HookManager.wrap

    async def wrap(self, ctx, call):
        try:
            return await orig_wrap(self, ctx, call)
        except progent_hook.ProgentBlocked as blk:
            return CallToolResult(content=[TextContent(type="text", text=str(blk))], isError=False)

    HookManager.wrap = wrap


def _patch_local_skill_tool() -> None:
    original_post_init = FunctionTool.__post_init__

    def post_init(self):
        original_post_init(self)
        if getattr(self, "name", None) != "load_skill":
            return
        import progent_runtime as pr

        pr.register_local_tool(self.name, getattr(self, "description", ""),
                               getattr(self, "params_json_schema", None))
        original_invoke = self.on_invoke_tool

        async def invoke(context, raw_arguments):
            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except Exception as exc:
                raise ValueError(f"Invalid JSON input for tool load_skill: {raw_arguments}") from exc
            if not isinstance(arguments, dict):
                raise ValueError(f"Invalid JSON input for tool load_skill: {raw_arguments}")
            refusal = await pr.check_call("load_skill", arguments)
            if refusal is not None:
                return refusal
            result = await original_invoke(context, raw_arguments)
            await pr.settle_result("load_skill", arguments, result if isinstance(result, str) else str(result))
            return result

        self.on_invoke_tool = invoke

    FunctionTool.__post_init__ = post_init


_patch_hook_config_path()
_patch_tool_spec_capture()
_patch_agent_call_capture()
_patch_progent_block_delivery()
_patch_local_skill_tool()
