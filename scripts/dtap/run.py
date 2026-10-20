from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.drift.dtap.aborts import score_deterministic_aborts
from scripts.dtap.conditions import CONDITIONS, condition_parts, method_environment
from scripts.dtap.protocol import (
    AGENT_TYPE, DTAP_HOME, ENVIRONMENTS, MODEL, TASKSETS, environment_model_env, task_identity, tasklist,
)
from scripts.dtap.report import summary

MAX_PASSES = 2
TIMEOUT_ATTEMPT_BUDGET = 1
HARNESS_ATTEMPT_BUDGET = 1
POLICY_FILES = ("timeout_policy.json", "harness_policy.json")
JUDGE_FAILED = re.compile(r"timed out|Failed to call .*judge endpoint|HTTPConnectionPool|Max retries exceeded", re.I)
TASK_TIMEOUT_LINE = re.compile(r"^\[EVAL\] .* TIMED OUT after \d+s — killed \(rc=124\)$", re.M)
HARNESS_REFUSAL_SIGNATURES = {
    "agents_sdk_unknown_tool_abort": re.compile(r"^\[ERROR\] Execution failed: Tool \S+ not found in agent \S+.*$", re.M),
    "agents_sdk_max_turns_exceeded": re.compile(r"^\[ERROR\] Execution failed: Max turns \(\d+\) exceeded$", re.M),
}
CONTEXT_OVERFLOW_LINE = re.compile(r"^\[CONTEXT OVERFLOW\] Turn \d+ exceeded model context window: .*$", re.M)
CONTEXT_OVERFLOW_JUDGED_LINE = re.compile(
    r"^\[PARTIAL\] Agent hit context overflow; running judge on partial execution$", re.M)
AGENT_TRANSPORT_ERROR = re.compile(
    r"^\s*API Error:\s*(?:\d{3}\b|fetch failed|ECONNREFUSED|ECONNRESET|ETIMEDOUT|Connection error)", re.I)
TRAJECTORY_DOCUMENT = re.compile(r"^\d{8}_\d{6}\.json$")
BOOTSTRAP_FATAL = re.compile(r"^\[DTAP-BOOTSTRAP-FATAL\] .*$", re.M)
INHERITED_ENV_NAMES = frozenset({
    "HOME", "USER", "LOGNAME", "SHELL", "PATH", "TMPDIR", "TMP", "TEMP", "DOCKER_HOST",
    "LANG", "LC_ALL", "TZ", "TERM", "PYTHONHASHSEED",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
    "EVAL_TASK_TIMEOUT", "MCP_CLIENT_TIMEOUT_SECONDS", "PLAYWRIGHT_BROWSERS_PATH", "OPENAI_API_KEY",
    "DTAP_USE_SUDO", "PIDETECTOR_URL", "PROMPTGUARD_URL", "DATASENTINEL_ENDPOINT",
})


def read_log(task_dir: Path) -> str:
    try:
        return (task_dir / "task.log").read_text(errors="replace")
    except OSError:
        return ""


def trajectory_documents(task_dir: Path) -> list[Path]:
    return sorted(path for path in task_dir.glob("*.json")
                  if path.is_file() and TRAJECTORY_DOCUMENT.fullmatch(path.name))


def judged_on_context_overflow(task_dir: Path) -> bool:
    log = read_log(task_dir)
    return ((task_dir / "judge_result.json").is_file() and not trajectory_documents(task_dir)
            and bool(CONTEXT_OVERFLOW_JUDGED_LINE.search(log)) and bool(CONTEXT_OVERFLOW_LINE.search(log)))


def agent_transport_error(document: Path) -> bool:
    try:
        payload = json.loads(document.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    texts = [str((payload.get("traj_info") or {}).get("agent_final_response") or "")]
    for step in payload.get("trajectory") or []:
        if isinstance(step, dict) and step.get("role") == "agent":
            texts.append(str((step.get("metadata") or {}).get("message") or ""))
    return any(AGENT_TRANSPORT_ERROR.search(text) for text in texts)


def trajectory_problem(task_dir: Path) -> bool:
    documents = trajectory_documents(task_dir)
    if not documents:
        return not judged_on_context_overflow(task_dir)
    return len(documents) > 1 or agent_transport_error(documents[0])


def errored_method_record(task_dir: Path) -> bool:
    for path in (task_dir / "method_evidence").glob("*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for key in CONDITIONS[path.stem].open_keys:
            try:
                if int(record.get(key) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def judged_row(task_dir: Path, taskset: str, identity: tuple[str, str]) -> dict | None:
    if trajectory_problem(task_dir):
        return None
    try:
        verdict = json.loads((task_dir / "judge_result.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    message = f"{verdict.get('task_message') or ''} {verdict.get('attack_message') or ''}"
    if (verdict.get("error") is not None or JUDGE_FAILED.search(message)
            or not isinstance(verdict.get("task_success"), bool)
            or (taskset != "benign" and not isinstance(verdict.get("attack_success"), bool))):
        return None
    row = {"task_id": identity[0], "risk_category": identity[1],
           "attack_success": verdict.get("attack_success"), "task_success": verdict["task_success"], "error": None}
    if judged_on_context_overflow(task_dir):
        row["context_overflow"] = True
    return row


def measured_row(task_dir: Path, taskset: str, identity: tuple[str, str]) -> dict | None:
    if not (task_dir / "judge_result.json").exists():
        for name in POLICY_FILES:
            if (task_dir / name).is_file():
                policy = json.loads((task_dir / name).read_text())
                return {"task_id": identity[0], "risk_category": identity[1],
                        "attack_success": None if taskset == "benign" else False,
                        "task_success": False, "error": None, **policy}
    if errored_method_record(task_dir):
        return None
    return judged_row(task_dir, taskset, identity)


class Unit:
    def __init__(self, row: dict, taskset: str, base: Path, scratch: Path):
        self.row = row
        self.identity = task_identity(row, taskset)
        if taskset == "benign":
            parts = ("benign", row["risk_category"], str(row["task_id"])) if row.get("risk_category") \
                else ("benign", str(row["task_id"]))
        else:
            parts = ("malicious", "indirect", row["risk_category"], str(row["task_id"]))
        self.dir = base.joinpath(*parts)
        self.dead_root = scratch / "_dead_attempts" / taskset / "__".join(parts[-2:])
        self.label = f"{self.identity[1]}__{self.identity[0]}" if self.identity[1] else self.identity[0].replace("/", "__")

    def dead_attempts(self) -> list[Path]:
        if not self.dead_root.is_dir():
            return []
        return sorted(path for path in self.dead_root.iterdir() if path.is_dir() and not path.name.startswith("."))


def settle(units: list[Unit], taskset: str) -> list[Unit]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    pending = []
    for unit in units:
        if measured_row(unit.dir, taskset, unit.identity) is not None:
            continue
        if unit.dir.exists():
            if (unit.dir / "judge_result.json").exists() and judged_row(unit.dir, taskset, unit.identity) is None:
                shutil.rmtree(unit.dir)
            else:
                unit.dead_root.mkdir(parents=True, exist_ok=True)
                shutil.move(str(unit.dir), str(unit.dead_root / stamp))
        logs = [read_log(attempt) for attempt in unit.dead_attempts()]
        timeouts = sum(bool(TASK_TIMEOUT_LINE.search(log)) for log in logs)
        refusals = {name: sum(bool(pattern.search(log)) for log in logs)
                    for name, pattern in HARNESS_REFUSAL_SIGNATURES.items()}
        signature = next((name for name, count in refusals.items() if count >= HARNESS_ATTEMPT_BUDGET), None)
        if timeouts >= TIMEOUT_ATTEMPT_BUDGET:
            policy = ("timeout_policy.json", {"scored_by": "timeout_policy", "timed_out_attempts": timeouts})
        elif signature is not None:
            policy = ("harness_policy.json", {"scored_by": "harness_policy",
                                              "refused_attempts": refusals[signature], "signature": signature})
        else:
            pending.append(unit)
            continue
        unit.dir.mkdir(parents=True, exist_ok=True)
        (unit.dir / policy[0]).write_text(json.dumps(policy[1], indent=1) + "\n")
    return pending


def services(environment: str) -> list[str]:
    definition = json.loads((ROOT / "benchmarks/dtap/environments" / environment / "environment.json").read_text())
    table = {name.lower(): config for name, config in
             json.loads((ROOT / "benchmarks/dtap/mcp_servers.json").read_text())["servers"].items()}
    names = set()
    for server in definition["dataset_mounts"]:
        config = table[server.lower()]
        environments = config.get("environment") or []
        names.update([server, Path(config["path"]).parts[0],
                      *([environments] if isinstance(environments, str) else environments)])
    return sorted(names)


def teardown(environment: str) -> None:
    pattern = "|".join(services(environment))
    subprocess.run(["pkill", "-f", f"dt_arena/mcp_server/({pattern})/.*\\.venv/bin/python"], check=False)
    time.sleep(2)
    for listing, remove in ((["docker", "ps", "--format", "{{.Names}}"], ["docker", "rm", "-f"]),
                            (["docker", "network", "ls", "--format", "{{.Name}}"], ["docker", "network", "rm"])):
        output = subprocess.run(listing, capture_output=True, text=True, check=False).stdout
        names = [name for name in output.split() if re.match(f"^pool_({pattern})_", name)]
        for name in names:
            subprocess.run([*remove, name], capture_output=True, check=False)
        time.sleep(2)


def benchmark_env(condition: str, environment: str, scratch: Path, proxy: str) -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if name in INHERITED_ENV_NAMES}
    api_key = os.environ.get("OPENAI_API_KEY", "proxy")
    env.update({
        "PATH": f"{ROOT}/.dtap-venv/bin:" + env.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join((str(ROOT / "scripts/dtap/site"), str(ROOT))),
        "PYTHONUNBUFFERED": "1",
        "UV_NO_SYNC": "1",
        "EVAL_RESULTS_ROOT": str(scratch),
        "DTAP_PROXY_URL": proxy,
        "OPENAI_BASE_URL": proxy,
        "OPENAI_API_KEY": api_key,
        "HOSPITAL_OPENAI_BASE_URL": proxy.replace("127.0.0.1", "host.docker.internal").replace(
            "localhost", "host.docker.internal"),
        "DTAP_TERMINAL_RESET": str(ROOT / "scripts/dtap/terminal_reset_fast.sh"),
        **environment_model_env(environment),
    })
    if environment == "os-filesystem":
        env.update({"PERMISSION_LLM_API_BASE_URL": proxy, "PERMISSION_LLM_API_KEY": api_key})
    if condition != "nodefense":
        env["DTAP_METHOD_OWNER"] = condition
        env.update(method_environment(condition, environment, proxy))
    return env


def evaluate(condition: str, environment: str, taskset: str, units: list[Unit], scratch: Path,
             proxy: str, workers: int) -> None:
    listing = scratch / "tasklists" / f"remaining_{taskset}.jsonl"
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text("".join(json.dumps(unit.row) + "\n" for unit in units))
    command = [str(ROOT / ".dtap-venv/bin/dtap"), "eval", "--task-list", str(listing),
               "--agent-type", AGENT_TYPE, "--model", MODEL, "--max-parallel", str(workers)]
    try:
        with (scratch / f"eval-{taskset}.log").open("a") as log:
            subprocess.call(command, cwd=DTAP_HOME, env=benchmark_env(condition, environment, scratch, proxy),
                            stdout=log, stderr=subprocess.STDOUT)
    finally:
        teardown(environment)


def measure(condition: str, environment: str, taskset: str, scratch: Path, proxy: str, workers: int) -> list[dict] | None:
    base = scratch / "benchmark" / AGENT_TYPE / MODEL / environment
    units = [Unit(row, taskset, base, scratch) for row in tasklist(environment, taskset)]
    for _ in range(MAX_PASSES):
        pending = settle(units, taskset)
        print(f"[dtap] {condition} {environment}/{taskset}: {len(pending)}/{len(units)} task(s) to run", flush=True)
        if not pending:
            break
        evaluate(condition, environment, taskset, pending, scratch, proxy, workers)
        if condition == "drift":
            for line in score_deterministic_aborts([(unit.dir, unit.dead_attempts()) for unit in pending], taskset):
                print(f"[dtap] DRIFT deterministic abort scored as failed: {line}", flush=True)
    pending = settle(units, taskset)
    if pending:
        print(f"[dtap] {condition} {environment}/{taskset}: INCOMPLETE, {len(pending)} task(s) unmeasured "
              f"(first: {pending[0].label}); rerun the same command", file=sys.stderr, flush=True)
        fatal = next((match.group(0) for unit in pending for attempt in unit.dead_attempts()
                      if (match := BOOTSTRAP_FATAL.search(read_log(attempt)))), None)
        if fatal:
            print(f"[dtap] the defense bootstrap failed: {fatal}", file=sys.stderr, flush=True)
        return None
    if condition != "nodefense" and not any(
            (unit.dir / "method_evidence" / f"{condition}.json").is_file() for unit in units):
        print(f"[dtap] WARNING: {condition} wrote no method record for any {environment}/{taskset} task; "
              "check that the defense loaded", file=sys.stderr, flush=True)
    return [measured_row(unit.dir, taskset, unit.identity) for unit in sorted(units, key=lambda unit: unit.label)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one DTap condition on one environment.")
    parser.add_argument("condition", choices=sorted(CONDITIONS))
    parser.add_argument("environment", choices=ENVIRONMENTS)
    parser.add_argument("--mp", type=int, default=3)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--proxy", default="http://127.0.0.1:8903/openrouter.ai/v1")
    args = parser.parse_args()
    parts = condition_parts(args.condition)
    run_key = f"run{args.run}"
    scratch = ROOT.joinpath("runs", "dtap", args.environment, *parts, run_key)
    outdir = ROOT.joinpath("results", "dtap", *parts, "main_results", run_key, args.environment)
    complete = True
    for taskset in TASKSETS:
        rows = measure(args.condition, args.environment, taskset, scratch, args.proxy, args.mp)
        if rows is None:
            complete = False
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "meta.json").write_text(json.dumps({
            "benchmark": "dtap", "condition": args.condition, "environment": args.environment,
            "model": MODEL, "run": run_key,
        }, indent=1, sort_keys=True) + "\n")
        (outdir / f"{taskset}.per_task.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (outdir / f"{taskset}.summary.json").write_text(
            json.dumps(summary(rows, args.environment, taskset), indent=1, sort_keys=True) + "\n")
        print(f"[dtap] {args.condition} {args.environment}/{taskset}: complete, {len(rows)} rows -> {outdir}", flush=True)
    return 0 if complete else 9


if __name__ == "__main__":
    raise SystemExit(main())
