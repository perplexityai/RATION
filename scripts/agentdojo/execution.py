from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
from agentdojo.logging import OutputLogger
from agentdojo.models import MODEL_NAMES
from agentdojo.task_suite.load_suites import get_suite

from scripts.agentdojo.conditions import available_conditions, condition_spec
from scripts.agentdojo.openai_llm import ChatLLM

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "benchmarks/agentdojo/protocol.json").read_text())
ATTACK = "important_instructions"


def raw_record_path(logdir: Path, pipeline: str, suite: str, user_task: str, injection_task: str | None) -> Path:
    attack = "none" if injection_task is None else ATTACK
    return logdir / pipeline / suite / user_task / attack / f"{injection_task or 'none'}.json"


def done_path(run_dir: Path, condition: str, suite: str, user_task: str, injection_task: str | None) -> Path:
    return run_dir / "done" / condition / suite / user_task / f"{injection_task or 'none'}.json"


def measured(path: Path) -> dict | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    if raw.get("error") or not isinstance(raw.get("messages"), list):
        return None
    if type(raw.get("utility")) is not bool or type(raw.get("security")) is not bool:
        return None
    return {"utility": raw["utility"], "security": raw["security"]}


def task_units(suite, taskset: str) -> list[tuple[str, str | None]]:
    if taskset == "benign":
        return [(user_task, None) for user_task in suite.user_tasks]
    return [(user_task, injection_task) for user_task in suite.user_tasks for injection_task in suite.injection_tasks]


def build(args, logdir: Path):
    os.environ["LLM_TEMPERATURE"] = str(float(CONFIG["temperature"]))
    os.environ["LLM_MAX_TOKENS"] = str(CONFIG["max_output_tokens"] or 0)
    os.environ.pop("LLM_REASONING_EFFORT", None)
    spec = condition_spec(args.condition)
    context = argparse.Namespace(
        condition=args.condition,
        suite=args.suite,
        model=CONFIG["model"],
        benchmark_version=CONFIG["benchmark_version"],
        logdir=str(logdir),
        attack=None if args.taskset == "benign" else ATTACK,
        provisioner_model=CONFIG["provisioner_model"],
    )
    MODEL_NAMES.setdefault(CONFIG["model"], "GPT-4")
    llm = ChatLLM(CONFIG["model"])
    autonomy = (ROOT / "benchmarks/agentdojo/autonomy.txt").read_text().rstrip("\n")
    system_message = SystemMessage(f"{load_system_message(None).rstrip()}\n{autonomy}")
    runtime = importlib.import_module(spec.runtime_module) if spec.runtime_module else None
    if runtime is None:
        pipeline = AgentPipeline([system_message, InitQuery(), llm, ToolsExecutionLoop([ToolsExecutor(), llm])])
        state = None
    else:
        pipeline, state = runtime.build_pipeline(context, llm, system_message)
    pipeline.name = spec.pipeline_name(CONFIG["model"])
    return pipeline, runtime, state, context


def run_units(args, suite, units: list[tuple[str, str | None]], run_dir: Path) -> None:
    logdir = run_dir / "logs"
    pipeline, runtime, state, context = build(args, logdir)
    attack = load_attack(ATTACK, suite, pipeline) if args.taskset == "indirect" else None
    with OutputLogger(str(logdir)):
        for user_task, injection_task in units:
            if hasattr(runtime, "prepare_unit"):
                runtime.prepare_unit(context, state, suite, user_task, injection_task)
            raw = raw_record_path(logdir, pipeline.name, args.suite, user_task, injection_task)
            raw.unlink(missing_ok=True)
            task = suite.get_user_task_by_id(user_task)
            if injection_task is None:
                run_task_without_injection_tasks(suite, pipeline, task, logdir, True, CONFIG["benchmark_version"])
            else:
                run_task_with_injection_tasks(
                    suite, pipeline, task, attack, logdir, True, [injection_task], CONFIG["benchmark_version"]
                )
            if hasattr(runtime, "finish_unit"):
                runtime.finish_unit(state)
            verdict = measured(raw)
            if verdict is None:
                print(f"[agentdojo] unmeasured {args.suite}/{user_task}/{injection_task or 'none'}", file=sys.stderr)
                continue
            path = done_path(run_dir, args.condition, args.suite, user_task, injection_task)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(verdict))
    if hasattr(runtime, "finish_taskset"):
        runtime.finish_taskset(state)


def write_results(args, verdicts: dict[tuple[str, str | None], dict]) -> Path:
    spec = condition_spec(args.condition)
    outdir = ROOT / "results/agentdojo" / Path(*spec.result_parts) / "main_results" / args.run / args.suite
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "benchmark": "agentdojo", "condition": args.condition, "environment": args.suite,
        "model": CONFIG["model"], "run": args.run,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    ordered = sorted(verdicts.items(), key=lambda item: (item[0][0], item[0][1] or ""))
    if args.taskset == "benign":
        rows = [{"user_task": user, "task_success": verdict["utility"]} for (user, _), verdict in ordered]
    else:
        rows = [
            {"user_task": user, "injection_task": injection, "task_success": verdict["utility"],
             "attack_success": verdict["security"]}
            for (user, injection), verdict in ordered
        ]
    (outdir / f"{args.taskset}.per_task.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (outdir / f"{args.taskset}.summary.json").write_text(json.dumps(summarize(args.taskset, rows), indent=1))
    return outdir


def summarize(taskset: str, rows: list[dict]) -> dict:
    if taskset == "benign":
        return {"taskset": "benign", "n": len(rows), "bsr": sum(row["task_success"] for row in rows) / len(rows)}
    return {
        "taskset": "indirect", "attack": ATTACK, "n": len(rows),
        "asr": sum(row["attack_success"] for row in rows) / len(rows),
        "utility_under_attack": sum(row["task_success"] for row in rows) / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one AgentDojo suite for one condition (resumable).")
    parser.add_argument("taskset", choices=("benign", "indirect"))
    parser.add_argument("condition", choices=available_conditions())
    parser.add_argument("suite", choices=CONFIG["suites"])
    parser.add_argument("run")
    args = parser.parse_args()
    suite = get_suite(CONFIG["benchmark_version"], args.suite)
    run_dir = ROOT / "runs/agentdojo" / args.run
    units = task_units(suite, args.taskset)
    pending = [unit for unit in units if not done_path(run_dir, args.condition, args.suite, *unit).is_file()]
    if pending:
        run_units(args, suite, pending, run_dir)
    verdicts = {}
    for unit in units:
        path = done_path(run_dir, args.condition, args.suite, *unit)
        if path.is_file():
            verdicts[unit] = json.loads(path.read_text())
    if len(verdicts) != len(units):
        print(f"[agentdojo] {len(units) - len(verdicts)}/{len(units)} units unmeasured; rerun to resume", file=sys.stderr)
        return 3
    print(f"[agentdojo] complete {args.condition}/{args.suite}/{args.taskset} -> {write_results(args, verdicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
