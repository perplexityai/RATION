from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agentdojo.conditions import condition_spec
from scripts.common.metrics import binomial, macro_average
from scripts.common.reporting import METRICS, make_report, report_main, result_directory


def read_rows(path: Path, expected: set, identity) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if sorted(map(identity, rows)) != sorted(expected):
        raise ValueError(f"{path}: rows do not match the task inventory")
    return rows


def suite_metrics(directory: Path, users: set[str], injections: set[str]) -> dict:
    benign = read_rows(directory / "benign.per_task.jsonl", users, lambda row: row["user_task"])
    attacked = read_rows(
        directory / "indirect.per_task.jsonl",
        {(user, injection) for user in users for injection in injections},
        lambda row: (row["user_task"], row["injection_task"]),
    )
    return {
        "ASR": binomial(row["attack_success"] for row in attacked),
        "UA": binomial(row["task_success"] for row in attacked),
        "BU": binomial(row["task_success"] for row in benign),
    }


def benchmark_report(benchmark: str, result_parts: tuple[str, ...], *, root: Path, source: str,
                     condition: str, run_key: str) -> dict:
    directory = result_directory(root, source, benchmark, result_parts, run_key)
    inventory = root / "benchmarks" / benchmark
    injection_tasks = json.loads((inventory / "injection_tasks.json").read_text())
    units = {
        suite: suite_metrics(
            directory / suite,
            set(json.loads((inventory / "environments" / suite / "folds.json").read_text())["folds"]),
            set(injections),
        )
        for suite, injections in injection_tasks.items()
    }
    overall = {key: macro_average(unit[key] for unit in units.values()) for key in METRICS}
    return make_report(benchmark=benchmark, condition=condition, source=source,
                       run_key=run_key, basis="equal domain means",
                       units=units, overall=overall)


def build_report(*, root: Path, source: str, condition: str, run_key: str) -> dict:
    return benchmark_report("agentdojo", condition_spec(condition).result_parts, root=root, source=source,
                            condition=condition, run_key=run_key)


if __name__ == "__main__":
    raise SystemExit(report_main("agentdojo", build_report))
