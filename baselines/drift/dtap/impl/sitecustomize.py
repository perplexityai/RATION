from pathlib import Path


def _patch_hook_config_path() -> None:
    import dt_arena.src.types.hooks as H

    H._HOOKS_CONFIG_PATH = Path(__file__).resolve().parent / "hooks.json"


def _patch_tool_spec_capture() -> None:
    from agent.openaisdk.src import mcp_wrapper as mw

    def _wrap(cls):
        orig = cls.list_tools

        async def list_tools(self, run_context=None, agent=None):
            tools = await orig(self, run_context, agent)
            import drift_runtime as dr
            dr.register_tools(getattr(self, "name", None), tools)
            return tools

        cls.list_tools = list_tools

    for c in (mw.MCPServerSse, mw.MCPServerStreamableHttp, mw.MCPServerStdio):
        _wrap(c)


def _patch_openaisdk_instructions() -> None:
    from agent.openaisdk.src.agent import OpenAISDKAgent

    _orig = OpenAISDKAgent._build_instructions_for_skill

    def _build_instructions_for_skill(self):
        base = _orig(self)

        def _instructions(_ctx, _agent):
            import drift_runtime as dr
            st = dr.ensure_state()
            if st is not None:
                g = st.guidelines()
                if g:
                    return base + "\n\n" + g
            return base

        return _instructions

    OpenAISDKAgent._build_instructions_for_skill = _build_instructions_for_skill


def _patch_drift_block_delivery() -> None:
    from dt_arena.src.types.hooks import HookManager
    from mcp.types import CallToolResult, TextContent
    import drift_hook

    orig_wrap = HookManager.wrap

    async def wrap(self, ctx, call):
        try:
            return await orig_wrap(self, ctx, call)
        except drift_hook.DriftBlocked as blk:
            return CallToolResult(content=[TextContent(type="text", text=str(blk))],
                                  isError=False)

    HookManager.wrap = wrap


def _patch_agent_call_capture() -> None:
    from openai.resources.chat import completions as _cc

    _orig_create = _cc.AsyncCompletions.create

    async def create(self, *a, **kw):
        import drift_runtime as dr
        if dr.is_agent_request(kw) and dr.incomplete() is not None:
            raise dr.DriftIncomplete(
                f"DRIFT task is incomplete ({dr.incomplete()}); aborting the agent run so the "
                "task is retried, never scored"
            )
        resp = await _orig_create(self, *a, **kw)
        dr.capture_batch(kw, resp)
        return resp

    _cc.AsyncCompletions.create = create


def _patch_local_skill_tool() -> None:
    from agents.tool import FunctionTool

    original_post_init = FunctionTool.__post_init__

    def post_init(self):
        original_post_init(self)
        if getattr(self, "name", None) != "load_skill":
            return
        original_invoke = self.on_invoke_tool

        async def invoke(context, raw_arguments):
            import json

            import drift_runtime as dr

            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except Exception:
                arguments = {"_raw_arguments": raw_arguments}
            if not isinstance(arguments, dict):
                arguments = {"_raw_arguments": raw_arguments}
            verdict = await dr.audit_call("load_skill", arguments)
            if verdict is not None and verdict["disposition"] != "allow":
                return verdict["text"]
            result = await original_invoke(context, raw_arguments)
            if isinstance(result, str):
                cleaned = await dr.isolate_result("load_skill", arguments, result)
                if cleaned is not None:
                    return cleaned
            return result

        self.on_invoke_tool = invoke

    FunctionTool.__post_init__ = post_init


_patch_hook_config_path()
_patch_tool_spec_capture()
_patch_openaisdk_instructions()
_patch_agent_call_capture()
_patch_drift_block_delivery()
_patch_local_skill_tool()
