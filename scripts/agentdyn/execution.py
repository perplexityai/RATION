from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

from scripts.agentdojo.openai_llm import ChatLLM
from scripts.agentdyn.conditions import available_conditions, condition_spec

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "benchmarks/agentdyn/protocol.json").read_text())
ATTACK = "important_instructions"
BENCHMARK_VERSION = "v1.2.2"


def numeric_id(value: str) -> int:
    return int(value.rpartition("_")[2])


def raw_record_path(logdir: Path, pipeline: str, suite: str, user_task: str, injection_task: str | None) -> Path:
    attack = "none" if injection_task is None else ATTACK
    return logdir / pipeline / suite / user_task / attack / f"{injection_task or 'none'}.json"


def done_path(run_dir: Path, condition: str, suite: str, user_task: str, injection_task: str | None) -> Path:
    return run_dir / "done" / condition / suite / user_task / f"{injection_task or 'none'}.json"


def measured(path: Path) -> dict | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    if raw.get("error") or not isinstance(raw.get("messages"), list) or len(raw["messages"]) < 2:
        return None
    if type(raw.get("utility")) is not bool or type(raw.get("security")) is not bool:
        return None
    return {"utility": raw["utility"], "security": raw["security"]}


def task_units(suite, taskset: str) -> list[tuple[str, str | None]]:
    users = sorted(suite.user_tasks, key=numeric_id)
    if taskset == "benign":
        return [(user_task, None) for user_task in users]
    injections = sorted(suite.injection_tasks, key=numeric_id)
    return [(user_task, injection_task) for user_task in users for injection_task in injections]


def build(args, logdir: Path):
    os.environ["LLM_TEMPERATURE"] = str(float(CONFIG["sampling"]["temperature"]))
    os.environ["LLM_MAX_TOKENS"] = str(CONFIG["sampling"]["max_output_tokens"] or 0)
    os.environ.pop("LLM_REASONING_EFFORT", None)
    spec = condition_spec(args.condition)
    model = CONFIG["models"]["agent"]["id"]
    context = argparse.Namespace(condition=args.condition, suite=args.suite, logdir=str(logdir))
    runtime = importlib.import_module(spec.runtime_module) if spec.runtime_module else None
    if runtime is None:
        pipeline = AgentPipeline.from_config(
            PipelineConfig(
                llm=ChatLLM(model),
                model_id=None,
                defense=None,
                tool_delimiter="tool",
                system_message_name=CONFIG["runtime"]["system_message"],
                system_message=None,
                tool_output_format=CONFIG["runtime"]["tool_output_format"],
                suite_name=None,
            )
        )
        state = None
    else:
        pipeline, state = runtime.build_pipeline(context, CONFIG)
    pipeline.name = spec.pipeline_name(model, CONFIG["runtime"]["attack_model_name_key"])
    return pipeline, runtime, state


def run_units(args, suite, units: list[tuple[str, str | None]], run_dir: Path) -> None:
    logdir = run_dir / "logs"
    pipeline, runtime, state = build(args, logdir)
    attack = load_attack(ATTACK, suite, pipeline) if args.taskset == "indirect" else None
    with OutputLogger(str(logdir)):
        for user_task, injection_task in units:
            raw = raw_record_path(logdir, pipeline.name, args.suite, user_task, injection_task)
            raw.unlink(missing_ok=True)
            task = suite.get_user_task_by_id(user_task)
            if injection_task is None:
                run_task_without_injection_tasks(suite, pipeline, task, logdir, True, BENCHMARK_VERSION)
            else:
                run_task_with_injection_tasks(suite, pipeline, task, attack, logdir, True, [injection_task], BENCHMARK_VERSION)
            if hasattr(runtime, "finish_unit"):
                runtime.finish_unit(state)
            verdict = measured(raw)
            if verdict is None:
                print(f"[agentdyn] unmeasured {args.suite}/{user_task}/{injection_task or 'none'}", file=sys.stderr)
                continue
            path = done_path(run_dir, args.condition, args.suite, user_task, injection_task)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(verdict))


def summarize(taskset: str, rows: list[dict]) -> dict:
    if taskset == "benign":
        return {"taskset": "benign", "n": len(rows), "bsr": sum(row["task_success"] for row in rows) / len(rows)}
    return {
        "taskset": "indirect", "attack": ATTACK, "n": len(rows),
        "asr": sum(row["attack_success"] for row in rows) / len(rows),
        "utility_under_attack": sum(row["task_success"] for row in rows) / len(rows),
    }


def write_results(args, verdicts: dict[tuple[str, str | None], dict]) -> Path:
    spec = condition_spec(args.condition)
    outdir = ROOT / "results/agentdyn" / Path(*spec.result_parts) / "main_results" / args.run / args.suite
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "benchmark": "agentdyn", "condition": args.condition, "environment": args.suite,
        "model": CONFIG["models"]["agent"]["id"], "run": args.run,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    rows = []
    for (user_task, injection_task), verdict in verdicts.items():
        row = {"user_task": user_task, "task_success": verdict["utility"]}
        if args.taskset == "indirect":
            row.update({"injection_task": injection_task, "attack_success": verdict["security"]})
        rows.append(row)
    (outdir / f"{args.taskset}.per_task.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (outdir / f"{args.taskset}.summary.json").write_text(
        json.dumps(summarize(args.taskset, rows), indent=2, sort_keys=True) + "\n"
    )
    return outdir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one AgentDyn suite for one condition (resumable).")
    parser.add_argument("taskset", choices=("benign", "indirect"))
    parser.add_argument("condition", choices=available_conditions())
    parser.add_argument("suite", choices=CONFIG["suites"])
    parser.add_argument("run")
    args = parser.parse_args()
    suite = get_suite(BENCHMARK_VERSION, args.suite)
    run_dir = ROOT / "runs/agentdyn" / args.run
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
        print(f"[agentdyn] {len(units) - len(verdicts)}/{len(units)} units unmeasured; rerun to resume", file=sys.stderr)
        return 3
    print(f"[agentdyn] complete {args.condition}/{args.suite}/{args.taskset} -> {write_results(args, verdicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
