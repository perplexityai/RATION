from __future__ import annotations

import atexit
import contextlib
import contextvars
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

from ration import folds as fold_table, policies as fold_policies
from ration.monitor import Monitor
from ration.policy_config import load_domain
from ration.provisioner import Provisioner
from scripts.common.method_evidence import write_method_record
from scripts.dtap.trusted_request import trusted_request

FRAMEWORK_BUCKET_SUFFIX = "::method-framework"
PROVISION_ATTEMPTS = 5
PROVISION_BACKOFF_S = 3.0
FRAMEWORK_MARKERS = (
    "AUTHORIZED USER REQUEST",
    "[UNTRUSTED TOOL OUTPUT",
    "<<TOOL_OUTPUT>>",
    "<<END_TOOL_OUTPUT>>",
    "Before acting on the tool output above",
    "[RATION]",
)
TOOL_OUTPUT_OPEN = "<<TOOL_OUTPUT>>"
TOOL_OUTPUT_CLOSE = "<<END_TOOL_OUTPUT>>"

_STATE: dict[str, Any] = {"monitor": None, "provisioned": False, "request": None, "dispatched": False,
                          "provisioning_failed": 0, "tokens_seen": None}
_MCP_SERVERS: dict[str, tuple[Any, set[str]]] = {}
_FRAMEWORK_CALL: contextvars.ContextVar[int] = contextvars.ContextVar("ration_fw_call", default=0)


def task_id() -> Optional[str]:
    for index, arg in enumerate(sys.argv):
        if arg == "--task-dir" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if arg.startswith("--task-dir="):
            return arg.split("=", 1)[1]
    return None


def register_mcp_server(server: Any, tool_names: list[str]) -> None:
    _MCP_SERVERS[server.name] = (server, set(tool_names))


def server_for_tool(tool: str) -> Any:
    requested_server, separator, bare_tool = tool.partition(".")
    for server_name, (server, tools) in _MCP_SERVERS.items():
        if separator and server_name != requested_server:
            continue
        if (bare_tool if separator else tool) in tools:
            return server
    return None


def fold_key(task_dir: str) -> str:
    parts = Path(task_dir).parts
    return "/".join(parts[parts.index("dataset") + 2:])


def load_table():
    fold = fold_table.fold_of(fold_table.load(Path(os.environ["RATION_FOLDS"])), fold_key(task_id()))
    policies = fold_policies.for_fold(fold_policies.load(Path(os.environ["RATION_POLICIES"])), fold)
    return load_domain(os.environ["RATION_TABLE"]).with_policy(policies)


def framework_bucket(tid: str) -> str:
    return tid + FRAMEWORK_BUCKET_SUFFIX


def _proxy_tokens(bucket: str) -> int:
    url = (os.environ["RATION_PROXY_URL"].rstrip("/") + "/experiment-usage?task="
           + urllib.parse.quote(bucket, safe=""))
    with urllib.request.urlopen(url, timeout=5) as response:
        return int(json.load(response)["tokens"])


def snapshot_token_baseline() -> None:
    _STATE["tokens_seen"] = _proxy_tokens(task_id())


def poll_token_delta() -> int:
    tid = task_id()
    total = _proxy_tokens(tid)
    previous = _STATE["tokens_seen"]
    if total < previous:
        raise RuntimeError(f"RATION token counter moved backwards for task {tid!r}: {total} < {previous}")
    _STATE["tokens_seen"] = total
    return total - previous


def in_framework_call() -> bool:
    return _FRAMEWORK_CALL.get() > 0


@contextlib.contextmanager
def framework_call():
    token = _FRAMEWORK_CALL.set(_FRAMEWORK_CALL.get() + 1)
    try:
        yield
    finally:
        _FRAMEWORK_CALL.reset(token)


def framework_prompt_chars(messages: list[dict]) -> int:
    owned = 0
    for message in messages or []:
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text", ""), str)
            )
        if not isinstance(content, str):
            continue
        inside_payload = False
        for line in content.splitlines(keepends=True):
            if TOOL_OUTPUT_OPEN in line:
                inside_payload = True
                owned += len(line)
                continue
            if TOOL_OUTPUT_CLOSE in line:
                inside_payload = False
                owned += len(line)
                continue
            if not inside_payload and any(marker in line for marker in FRAMEWORK_MARKERS):
                owned += len(line)
    return owned


def read_trusted_request() -> str:
    return trusted_request(task_id())


def _dump_method_record() -> None:
    tid = task_id()
    if not tid:
        return
    record = {
        "task_id": tid,
        "provisioning_failed": _STATE["provisioning_failed"],
        "events": list(getattr(_STATE["monitor"], "events", []) or []),
    }
    try:
        write_method_record(tid, "ration", record)
    except Exception as exc:
        print(f"[RATION] failed to persist the method record: {exc}", file=sys.stderr, flush=True)


atexit.register(_dump_method_record)


class ProxyLLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, *, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
        }).encode("utf-8")
        headers = {"content-type": "application/json"}
        tid = task_id()
        if tid:
            headers["X-EXPERIMENT-TASK"] = framework_bucket(tid)
        request = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        return (payload["choices"][0]["message"].get("content") or "").strip()


def provision(user_request: str) -> Monitor:
    if _STATE["provisioned"]:
        return _STATE["monitor"]
    if _STATE["dispatched"]:
        _STATE["provisioning_failed"] = 1
        raise RuntimeError("RATION cannot provision after a tool call was dispatched without a plan")
    domain = load_table()
    provisioner = Provisioner(ProxyLLMClient(os.environ["RATION_PROXY_URL"], os.environ["RATION_MODEL"]))
    last_exc: Optional[BaseException] = None
    outcome = None
    for attempt in range(1, PROVISION_ATTEMPTS + 1):
        try:
            outcome = provisioner.plan(user_request, domain)
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
            last_exc = exc
            print(f"[RATION] provisioning attempt {attempt}/{PROVISION_ATTEMPTS} failed on transport: "
                  f"{type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr, flush=True)
            if attempt < PROVISION_ATTEMPTS:
                time.sleep(PROVISION_BACKOFF_S * attempt)
    if outcome is None:
        _STATE["provisioning_failed"] = 1
        raise RuntimeError(f"RATION provisioning failed after {PROVISION_ATTEMPTS} transport attempts: "
                           f"{type(last_exc).__name__}: {last_exc}")
    monitor = Monitor(outcome.plan)
    _STATE.update(monitor=monitor, request=user_request, provisioned=True, provisioning_failed=0)
    return monitor


def mark_dispatched() -> None:
    _STATE["dispatched"] = True


def ensure_provisioned() -> Monitor:
    if _STATE["provisioned"]:
        return _STATE["monitor"]
    return provision(read_trusted_request())


def get_monitor() -> Optional[Monitor]:
    return _STATE["monitor"]


def get_request() -> Optional[str]:
    return _STATE["request"]
