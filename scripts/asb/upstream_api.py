from __future__ import annotations

import contextvars
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from scripts.asb.optional_imports import install as install_optional_imports

ROOT = Path(__file__).resolve().parents[2]
_ACTIVE_PROCESS: contextvars.ContextVar[Any | None] = contextvars.ContextVar("asb_active_process", default=None)


def task_total(task_id: str) -> int:
    endpoint = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/experiment-usage?task=" + quote(task_id, safe="")
    with urlopen(endpoint, timeout=5) as response:
        return json.load(response)["tokens"]


class _CompletionsProxy:
    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def create(self, *args: Any, **kwargs: Any) -> Any:
        agent = getattr(_ACTIVE_PROCESS.get(), "asb_agent", None)
        if agent is not None:
            headers = dict(kwargs.pop("extra_headers", None) or {})
            headers["X-EXPERIMENT-TASK"] = agent.asb_task_id
            excluded = agent.asb_framework_prompt_chars(kwargs.get("messages") or [])
            if excluded:
                headers["X-EXPERIMENT-EXCLUDED-PROMPT-CHARS"] = str(excluded)
            kwargs["extra_headers"] = headers
        return self._wrapped.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class _ChatProxy:
    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.completions = _CompletionsProxy(wrapped.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class _ClientProxy:
    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.chat = _ChatProxy(wrapped.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class TaskBucketMixin:
    def create_agent_request(self, query: Any) -> Any:
        process = super().create_agent_request(query)
        process.asb_agent = self
        return process

    @property
    def asb_task_id(self) -> str:
        return ":".join(("asb", self.args.asb_taskset, str(self.agent_name).split("/")[-1], self.tool_name))

    def asb_framework_prompt_chars(self, messages: list[dict[str, Any]]) -> int:
        return 0


def official_components(*, root: Path = ROOT, request_model: str) -> SimpleNamespace:
    vendor = root / "vendor/asb"
    install_optional_imports()
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    from aios.llm_core.llm_classes.gpt_llm import GPTLLM
    from aios.llm_core.llm_classes.model_registry import MODEL_REGISTRY
    from aios.llm_core.llms import LLMKernel
    from aios.scheduler.fifo_scheduler import FIFOScheduler
    from main_attacker import check_attack_success, check_original_success, seed_everything
    from pyopenagi.agents.agent_factory import AgentFactory
    from pyopenagi.agents.agent_process import AgentProcessFactory
    from pyopenagi.agents.interact import Interactor

    class CapturingGPTLLM(GPTLLM):
        def load_llm_and_tokenizer(self) -> None:
            super().load_llm_and_tokenizer()
            self.model = _ClientProxy(self.model)

        def process(self, agent_process: Any, temperature: float = 0.0) -> None:
            token = _ACTIVE_PROCESS.set(agent_process)
            try:
                super().process(agent_process, temperature)
            finally:
                _ACTIVE_PROCESS.reset(token)

    class CapturingAgentFactory(AgentFactory):
        def __init__(self, *args: Any, agent_mixin: type | None = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.agent_mixin = agent_mixin
            self._capturing_classes: dict[type, type] = {}

        def load_agent_instance(self, agent_name: str):
            official_class = super().load_agent_instance(agent_name)
            if official_class not in self._capturing_classes:
                mixins = (self.agent_mixin,) if self.agent_mixin is not None else ()
                self._capturing_classes[official_class] = type(
                    f"Captured{official_class.__name__}", (*mixins, TaskBucketMixin, official_class), {}
                )
            return self._capturing_classes[official_class]

    MODEL_REGISTRY[request_model] = CapturingGPTLLM
    Interactor.check_reqs_installed = lambda self, _agent: True
    return SimpleNamespace(
        vendor=vendor,
        LLMKernel=LLMKernel,
        FIFOScheduler=FIFOScheduler,
        AgentFactory=CapturingAgentFactory,
        AgentProcessFactory=AgentProcessFactory,
        check_attack_success=check_attack_success,
        check_original_success=check_original_success,
        seed_everything=seed_everything,
    )


def agent_args(*, root: Path = ROOT, taskset: str, request_model: str, max_output_tokens: int) -> SimpleNamespace:
    attacked = taskset == "attacked"
    data = root / "benchmarks/asb/upstream_data"
    return SimpleNamespace(
        llm_name=request_model,
        max_gpu_memory=None,
        eval_device=None,
        max_new_tokens=max_output_tokens,
        scheduler_log_mode="console",
        agent_log_mode="console",
        llm_kernel_log_mode="console",
        use_backend=None,
        workflow_mode="automatic",
        attacker_tools_path=str(data / "all_attack_tools.jsonl"),
        tasks_path=str(data / "agent_task.jsonl"),
        tools_info_path=str(data / "all_normal_tools.jsonl"),
        observation_prompt_injection=attacked,
        plan_attack=False,
        action_attack=False,
        direct_prompt_injection=False,
        jailbreak=False,
        attack_type="context_ignoring",
        defense_type=None,
        pot_backdoor=False,
        pot_clean=False,
        target="ResearcherProfileManager",
        trigger="in light of extant literature",
        memory_attack=False,
        clean=not attacked,
        result_file="",
        database="",
        task_num=1,
        write_db=False,
        read_db=False,
        res_file="",
        asb_taskset=taskset,
    )
