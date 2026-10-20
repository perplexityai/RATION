from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.metrics import binomial, macro_average
from scripts.common.reporting import METRICS, make_report, report_main, result_directory
from scripts.dtap.conditions import condition_parts
from scripts.dtap.protocol import ENVIRONMENTS, TASKSETS, task_identity, tasklist

PARTITIONED_DOMAINS = {"code", "legal", "finance", "research", "travel", "medical", "crm", "telecom"}
TRAVEL_PREFIXES = ("flight", "hotel", "restaurant", "attraction", "itinerary")
METRIC_FIELDS = {"indirect": {"ASR": "attack_success", "UA": "task_success"}, "benign": {"BU": "task_success"}}


@lru_cache(maxsize=None)
def _category_map(domain: str, root: Path) -> dict[str, str]:
    path = root / "benchmarks/dtap/environments" / domain / "environment.json"
    return json.loads(path.read_text())["benign_categories"]


def benign_category(domain: str, task_id: str, *, root: Path = ROOT) -> str | None:
    parts = task_id.split("/")
    if parts[0] != "benign" and len(parts) == 2:
        return parts[0]
    name = parts[-1]
    if domain == "travel":
        for prefix in TRAVEL_PREFIXES:
            if name.startswith(prefix):
                return prefix
        if name.startswith(("round-trip", "roundtrip", "round_trip")):
            return "round-trip"
        return "OTHER"
    if domain == "medical":
        return re.sub(r"-\d+$", "", name)
    if domain in {"crm", "telecom"}:
        return _category_map(domain, root).get(name) or "OTHER"
    return None


def attack_category(category: str) -> str:
    for prefix, name in (("CWE-", "CWE"), ("Mitre_", "Mitre"), ("OWASP_", "OWASP")):
        if category.startswith(prefix):
            return name
    return category


def metric(rows: list[dict], environment: str, taskset: str, field: str, *, root: Path = ROOT):
    if environment not in PARTITIONED_DOMAINS:
        return binomial(row[field] for row in rows)
    groups = defaultdict(list)
    for row in rows:
        category = (benign_category(environment, row["task_id"], root=root) if taskset == "benign"
                    else attack_category(row["risk_category"]))
        if not category or category == "OTHER":
            raise ValueError(f"{environment}/{taskset}: unmapped task category")
        groups[category].append(row[field])
    return macro_average(binomial(verdicts) for verdicts in groups.values())


def summary(rows: list[dict], environment: str, taskset: str, *, root: Path = ROOT) -> dict:
    metrics = {name: metric(rows, environment, taskset, field, root=root).as_dict()
               for name, field in METRIC_FIELDS[taskset].items()}
    return {"taskset": taskset, "n": len(rows), **metrics}


def measured_rows(directory: Path, environment: str, taskset: str, root: Path) -> list[dict]:
    path = directory / f"{taskset}.per_task.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = {task_identity(row, taskset) for row in tasklist(environment, taskset, root=root)}
    identities = [(row["task_id"], row["risk_category"]) for row in rows]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError(f"{path}: rows do not cover the task inventory")
    if any(row["error"] is not None for row in rows):
        raise ValueError(f"{path}: rows carry unmeasured errors")
    return rows


def build_report(*, root: Path, source: str, condition: str, run_key: str) -> dict:
    directory = result_directory(root, source, "dtap", condition_parts(condition), run_key)
    units = {}
    for environment in ENVIRONMENTS:
        rows = {taskset: measured_rows(directory / environment, environment, taskset, root)
                for taskset in TASKSETS}
        units[environment] = {name: metric(rows[taskset], environment, taskset, field, root=root)
                              for taskset in TASKSETS for name, field in METRIC_FIELDS[taskset].items()}
    overall = {key: macro_average(unit[key] for unit in units.values()) for key in METRICS}
    return make_report(benchmark="dtap", condition=condition, source=source,
                       run_key=run_key,
                       basis="equal domain means; eight category means and six flat domains",
                       units=units, overall=overall)


if __name__ == "__main__":
    raise SystemExit(report_main("dtap", build_report))
