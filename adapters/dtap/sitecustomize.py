import json
import sys
from pathlib import Path

import dt_arena.src.types.hooks as hooks
import openai
import utils.skill_helpers as skill_helpers
from agent.openaisdk.src import mcp_wrapper
from agent.openaisdk.src.agent import OpenAISDKAgent
from mcp.types import CallToolResult, TextContent

from adapters.dtap import dtap_hook, egress_reflection, ration_runtime as rr
from ration import feedback

ALLOWED_SKILLS: frozenset[str] = frozenset()


def _patch_hook_config_path() -> None:
    hooks._HOOKS_CONFIG_PATH = Path(__file__).resolve().parent / "hooks.json"


def _patch_instructions() -> None:
    original = OpenAISDKAgent._build_instructions_for_skill

    def build_instructions_for_skill(self):
        base = original(self)

        def instructions(_ctx, _agent):
            text = base
            request = rr.get_request() or rr.read_trusted_request()
            if request:
                text = feedback.request_wrapper(request) + "\n\n" + text
            monitor = rr.get_monitor()
            breached = monitor.breached_risks() if monitor is not None else []
            if breached:
                current = [dict(risk, attempted=risk.get("used", 0)) for risk in breached]
                text = text + "\n\n" + feedback.soft_reminder(current, [])
            return text

        return instructions

    OpenAISDKAgent._build_instructions_for_skill = build_instructions_for_skill


def _patch_task_header() -> None:
    tid = rr.task_id()
    original = openai.AsyncOpenAI.__init__

    async def tag_request(request):
        framework_call = rr.in_framework_call()
        request.headers["X-EXPERIMENT-TASK"] = rr.framework_bucket(tid) if framework_call else tid
        request.headers.pop("X-EXPERIMENT-EXCLUDED-PROMPT-CHARS", None)
        if not framework_call:
            excluded = rr.framework_prompt_chars(json.loads(request.content or b"{}").get("messages") or [])
            if excluded:
                request.headers["X-EXPERIMENT-EXCLUDED-PROMPT-CHARS"] = str(excluded)

    def init(self, *args, **kwargs):
        kwargs["default_headers"] = {"X-EXPERIMENT-TASK": tid, **(kwargs.get("default_headers") or {})}
        original(self, *args, **kwargs)
        self._client.event_hooks.setdefault("request", []).append(tag_request)

    openai.AsyncOpenAI.__init__ = init


def _patch_mcp_tool_listing() -> None:
    def wrap(cls):
        original = cls.list_tools

        async def list_tools(self, run_context=None, agent=None):
            tools = await original(self, run_context, agent)
            rr.register_mcp_server(self, [tool.name for tool in tools])
            return tools

        cls.list_tools = list_tools

    for cls in (mcp_wrapper.MCPServerSse, mcp_wrapper.MCPServerStreamableHttp, mcp_wrapper.MCPServerStdio):
        wrap(cls)


def _patch_skill_guard() -> None:
    original = skill_helpers.scan_available_skills

    def scan_available_skills(skills_dir, *args, **kwargs):
        return [skill for skill in original(skills_dir, *args, **kwargs) if skill.get("name") in ALLOWED_SKILLS]

    skill_helpers.scan_available_skills = scan_available_skills
    for module in list(sys.modules.values()):
        try:
            if module is not None and getattr(module, "scan_available_skills", None) is original:
                setattr(module, "scan_available_skills", scan_available_skills)
        except Exception:
            continue


def _patch_block_delivery() -> None:
    original = hooks.HookManager.wrap

    async def wrap(self, ctx, call):
        try:
            return await original(self, ctx, call)
        except dtap_hook.RationBlocked as blocked:
            return CallToolResult(content=[TextContent(type="text", text=str(blocked))], isError=False)

    hooks.HookManager.wrap = wrap


_patch_hook_config_path()
_patch_instructions()
_patch_task_header()
egress_reflection.install()
_patch_mcp_tool_listing()
_patch_skill_guard()
rr.snapshot_token_baseline()
_patch_block_delivery()
