from pathlib import Path


def _patch_hook_config_path() -> None:
    import dt_arena.src.types.hooks as H

    H._HOOKS_CONFIG_PATH = Path(__file__).resolve().parent / "hooks.json"


def _patch_agent_call_capture() -> None:
    from openai.resources.chat import completions as _cc

    _orig_create = _cc.AsyncCompletions.create

    async def create(self, *a, **kw):
        resp = await _orig_create(self, *a, **kw)
        import attriguard_runtime as ar
        ar.maybe_capture(kw, resp)
        return resp

    _cc.AsyncCompletions.create = create


def _patch_attriguard_block_delivery() -> None:
    from dt_arena.src.types.hooks import HookManager
    import attriguard_hook
    attriguard_hook.install_block_delivery(HookManager)


def _patch_local_skill_tool() -> None:
    from agents.tool import FunctionTool

    original_post_init = FunctionTool.__post_init__

    def post_init(self):
        original_post_init(self)
        if getattr(self, "name", None) != "load_skill":
            return
        if getattr(self, "_attriguard_local_tool", False):
            return
        original_invoke = self.on_invoke_tool

        async def invoke(context, raw_arguments):
            import attriguard_runtime as ar

            arguments = ar.parse_tool_arguments(raw_arguments)
            verdict = await ar.audit_call("load_skill", arguments)
            if verdict is not None and not verdict.get("allow"):
                return ar.blocked_text("load_skill", arguments)
            result = await original_invoke(context, raw_arguments)
            ar.record_tool_result("load_skill", arguments, result)
            return result

        self.on_invoke_tool = invoke
        self._attriguard_local_tool = True

    FunctionTool.__post_init__ = post_init


_patch_hook_config_path()
_patch_agent_call_capture()
_patch_attriguard_block_delivery()
_patch_local_skill_tool()
