from __future__ import annotations

import json
from pathlib import Path

from scripts.asb.conditions import condition_spec
from scripts.asb.upstream import inventory
from scripts.common.metrics import binomial
from scripts.common.reporting import make_report, report_main, result_directory


def read_rows(directory: Path, taskset: str, tools: set[str], verdicts: tuple[str, ...]) -> list[dict]:
    path = directory / f"{taskset}.per_task.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    names = [row["attacker_tool"] for row in rows]
    if len(names) != len(tools) or set(names) != tools:
        raise ValueError(f"{path}: incomplete task inventory ({len(set(names) & tools)}/{len(tools)})")
    if any(type(row[field]) is not bool for row in rows for field in verdicts):
        raise ValueError(f"{path}: missing or non-boolean verdict")
    return rows


def build_report(*, root: Path, source: str, condition: str, run_key: str) -> dict:
    directory = result_directory(root, source, "asb", condition_spec(condition).result_parts, run_key)
    units = {}
    all_benign = []
    all_attacked = []
    for scenario in inventory(root=root):
        tools = {tool["Attacker Tool"] for tool in scenario.tools}
        benign = read_rows(directory / scenario.name, "clean", tools, ("task_success",))
        attacked = read_rows(directory / scenario.name, "attacked", tools, ("task_success", "attack_success"))
        all_benign.extend(benign)
        all_attacked.extend(attacked)
        units[scenario.name] = {
            "ASR": binomial(row["attack_success"] for row in attacked),
            "UA": binomial(row["task_success"] for row in attacked),
            "BU": binomial(row["task_success"] for row in benign),
        }
    overall = {
        "ASR": binomial(row["attack_success"] for row in all_attacked),
        "UA": binomial(row["task_success"] for row in all_attacked),
        "BU": binomial(row["task_success"] for row in all_benign),
    }
    return make_report(benchmark="asb", condition=condition, source=source,
                       run_key=run_key, basis="pooled over executions",
                       units=units, overall=overall)


if __name__ == "__main__":
    raise SystemExit(report_main("asb", build_report))
