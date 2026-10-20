from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

from agentdojo import attacks
from agentdojo.agent_pipeline import AgentPipeline, InitQuery
from agentdojo.benchmark import (
    TaskResults,
    run_task_with_injection_tasks,
    run_task_without_injection_tasks,
)
from agentdojo.models import MODEL_NAMES

from scripts.agentdojo.execution import raw_record_path
from scripts.common.openai_proxy import Handler

ATTACK = "important_instructions"
QUARANTINED_MODEL = "openai:gpt-4.1"
QUARANTINED_RETRIES = 10
MAX_ATTEMPTS = 10
PROXY_ENV = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "PROXY_FORCE_MODEL")


@dataclass
class RuntimeState:
    live_pipeline: AgentPipeline
    server: ThreadingHTTPServer
    server_thread: threading.Thread
    previous_cwd: Path
    previous_env: dict[str, str | None]


def _start_proxy(model: str) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, name="camel-openai-proxy", daemon=True)
    thread.start()
    os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    os.environ.setdefault("OPENAI_API_KEY", "proxy")
    os.environ["PROXY_FORCE_MODEL"] = model
    return server, thread


def build_pipeline(args, llm, system_message):
    from camel.interpreter.interpreter import MetadataEvalMode
    from camel.pipeline_elements.privileged_llm import PrivilegedLLM
    from camel.pipeline_elements.replay_privileged_llm import (
        PrivilegedLLMReplayer,
        UserInjectionTasksGetter,
    )
    from camel.pipeline_elements.security_policies import (
        ADNoSecurityPolicyEngine,
        BankingSecurityPolicyEngine,
        SlackSecurityPolicyEngine,
        TravelSecurityPolicyEngine,
        WorkspaceSecurityPolicyEngine,
    )

    engine = {
        "banking": BankingSecurityPolicyEngine,
        "slack": SlackSecurityPolicyEngine,
        "travel": TravelSecurityPolicyEngine,
        "workspace": WorkspaceSecurityPolicyEngine,
    }[args.suite]
    previous_env = {name: os.environ.get(name) for name in PROXY_ENV}
    server, server_thread = _start_proxy(args.model)
    logdir = Path(args.logdir).resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    os.chdir(logdir.parent)
    live = AgentPipeline(
        [
            InitQuery(),
            PrivilegedLLM(
                llm,
                ADNoSecurityPolicyEngine,
                QUARANTINED_MODEL,
                eval_mode=MetadataEvalMode.NORMAL,
                quarantined_llm_retries=QUARANTINED_RETRIES,
                max_attempts=MAX_ATTEMPTS,
            ),
        ]
    )
    live.name = f"openai-{args.model}+camel"
    MODEL_NAMES.setdefault(live.name, "GPT-4")
    scored = AgentPipeline(
        [
            InitQuery(),
            UserInjectionTasksGetter(),
            PrivilegedLLMReplayer(live.name, ATTACK, engine, MetadataEvalMode.NORMAL),
        ]
    )
    state = RuntimeState(
        live_pipeline=live,
        server=server,
        server_thread=server_thread,
        previous_cwd=previous_cwd,
        previous_env=previous_env,
    )
    return scored, state


def _valid_live_trace(path: Path, live_name: str, suite: str, user_task: str, injection_task: str | None) -> bool:
    if not path.is_file():
        return False
    try:
        TaskResults.model_validate_json(path.read_text())
        raw = json.loads(path.read_text())
    except Exception:
        return False
    expected = {
        "pipeline_name": live_name,
        "suite_name": suite,
        "user_task_id": user_task,
        "injection_task_id": injection_task,
        "attack_type": None if injection_task is None else ATTACK,
        "agentdojo_package_version": "0.1.35",
        "benchmark_version": "v1.2.2",
    }
    return (
        raw.get("error") in (None, "")
        and isinstance(raw.get("messages"), list)
        and all(raw.get(key) == value for key, value in expected.items())
    )


def prepare_unit(args, state: RuntimeState, suite, user_task: str, injection_task: str | None) -> None:
    live = state.live_pipeline
    live_path = raw_record_path(Path(args.logdir), live.name, args.suite, user_task, injection_task)
    if _valid_live_trace(live_path, live.name, args.suite, user_task, injection_task):
        return
    task = suite.get_user_task_by_id(user_task)
    if injection_task is None:
        run_task_without_injection_tasks(suite, live, task, Path(args.logdir), True, args.benchmark_version)
    else:
        attack = attacks.load_attack(args.attack, suite, live)
        run_task_with_injection_tasks(
            suite, live, task, attack, Path(args.logdir), True, [injection_task], args.benchmark_version
        )
    if not _valid_live_trace(live_path, live.name, args.suite, user_task, injection_task):
        raise RuntimeError(f"CaMeL live prerequisite is absent or invalid: {live_path}")


def finish_taskset(state: RuntimeState) -> None:
    state.server.shutdown()
    state.server.server_close()
    state.server_thread.join(timeout=5)
    os.chdir(state.previous_cwd)
    for name, value in state.previous_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
