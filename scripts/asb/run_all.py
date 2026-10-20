#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import chdir
from pathlib import Path
from typing import Any

from scripts.asb.conditions import condition_spec, registered_condition
from scripts.asb.upstream import Scenario, inventory
from scripts.asb.upstream_api import agent_args, official_components
from scripts.common.openai_proxy import LocalProxy

ROOT = Path(__file__).resolve().parents[2]
TASKSETS = ("clean", "attacked")
ERROR_MARKERS = (
    "Server connection error:",
    "OpenAI RATE LIMIT error",
    "OpenAI STATUS error",
    "OpenAI BAD REQUEST error",
    "An unexpected error occurred:",
    "upstream error:",
)


def producer_error(messages: Any) -> str | None:
    if not isinstance(messages, list) or not messages:
        return "official result has no messages"
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and any(marker in content for marker in ERROR_MARKERS):
            return content
    return None


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}")
    staged.write_text(text)
    os.replace(staged, path)


def score(official: Any, args: Any, scenario: Scenario, tool: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    messages = result["messages"]
    row = {
        "scenario": scenario.name,
        "task_index": 0,
        "attacker_tool": tool["Attacker Tool"],
        "task_success": bool(official.check_original_success(messages, args, result["normal_tools"])),
    }
    if args.observation_prompt_injection:
        row["attack_success"] = bool(official.check_attack_success(messages, args, tool["Attack goal"]))
    return row


def summary(taskset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if taskset == "clean":
        return {"n": n, "benign_utility": sum(row["task_success"] for row in rows) / n}
    return {
        "n": n,
        "utility_under_attack": sum(row["task_success"] for row in rows) / n,
        "asr": sum(row["attack_success"] for row in rows) / n,
    }


def write_results(directory: Path, *, condition: str, model: str, run_key: str, taskset: str, scenario: Scenario, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: row["attacker_tool"])
    meta = {"benchmark": "asb", "condition": condition, "environment": scenario.name, "model": model, "run": run_key}
    write_text(directory / "meta.json", json.dumps(meta, indent=1, sort_keys=True) + "\n")
    write_text(directory / f"{taskset}.per_task.jsonl", "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows))
    write_text(directory / f"{taskset}.summary.json", json.dumps(summary(taskset, rows), indent=2, sort_keys=True) + "\n")


def run_taskset(args: argparse.Namespace, settings: dict[str, Any], official: Any, taskset: str, scratch: Path) -> int:
    spec = condition_spec(args.condition)
    model = settings["model"]
    official.seed_everything(0)
    agent = agent_args(root=args.root, taskset=taskset, request_model=model["openai_request_id"], max_output_tokens=settings["max_output_tokens"])
    mixin = importlib.import_module(spec.runtime_module).agent_mixin(settings) if spec.runtime_module else None
    with chdir(scratch):
        llm = official.LLMKernel(llm_name=model["openai_request_id"], max_new_tokens=settings["max_output_tokens"], log_mode="console", use_backend=None)
    scheduler = official.FIFOScheduler(llm=llm, log_mode="console")
    factory = official.AgentFactory(
        agent_process_queue=scheduler.agent_process_queue,
        agent_process_factory=official.AgentProcessFactory(),
        agent_log_mode="console",
        agent_mixin=mixin,
    )
    results = args.root / "results/asb" / Path(*spec.result_parts) / "main_results" / args.run_key
    rows: dict[str, list[dict[str, Any]]] = {}
    futures = {}
    failures = 0
    scheduler.start()
    try:
        with chdir(official.vendor), ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            for scenario in inventory(root=args.root):
                rows[scenario.name] = []
                for tool in scenario.tools:
                    done = scratch / taskset / scenario.name / f"{tool['Attacker Tool']}.json"
                    if done.is_file():
                        rows[scenario.name].append(json.loads(done.read_text()))
                        continue
                    future = pool.submit(factory.run_agent, scenario.agent_path, scenario.task, agent, tool, None, tool["Aggressive"])
                    futures[future] = (scenario, tool, done)
            for future in as_completed(futures):
                scenario, tool, done = futures[future]
                try:
                    result = future.result()
                    error = producer_error(result["messages"])
                    row = score(official, agent, scenario, tool, result) if error is None else None
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                if error is not None:
                    failures += 1
                    print(f"UNMEASURED {taskset}/{scenario.name}/{tool['Attacker Tool']}: {error}", file=sys.stderr)
                    continue
                write_text(done, json.dumps(row, sort_keys=True) + "\n")
                rows[scenario.name].append(row)
    finally:
        scheduler.stop()
    for scenario in inventory(root=args.root):
        if len(rows[scenario.name]) == len(scenario.tools):
            write_results(results / scenario.name, condition=args.condition, model=model["id"], run_key=args.run_key, taskset=taskset, scenario=scenario, rows=rows[scenario.name])
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one ASB condition (clean and attacked tasksets, resumable).")
    parser.add_argument("--condition", required=True, type=registered_condition)
    parser.add_argument("--run-key", default="run1")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--proxy-port", type=int, default=8911)
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()
    settings = json.loads((args.root / "benchmarks/asb/protocol.json").read_text())
    scratch = args.root / "runs/asb" / args.run_key / args.condition
    scratch.mkdir(parents=True, exist_ok=True)
    official = official_components(root=args.root, request_model=settings["model"]["openai_request_id"])
    with LocalProxy(port=args.proxy_port, aliases={settings["model"]["openai_request_id"]: settings["model"]["id"]}) as proxy:
        os.environ["OPENAI_BASE_URL"] = proxy.base_url
        for taskset in TASKSETS:
            failures = run_taskset(args, settings, official, taskset, scratch)
            if failures:
                print(f"[asb] {taskset}: {failures} units unmeasured; rerun to resume", file=sys.stderr)
                return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
