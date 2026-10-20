from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.reporting import METRICS, SOURCES

METHODS = (
    ("nodefense", "No defense"), ("sandwich", "Sandwich"), ("instructional", "Instructional"),
    ("spotlighting", "Spotlighting"), ("pidetector", "PI Detector"), ("promptguard", "Prompt Guard 2"),
    ("datasentinel", "DataSentinel"), ("melon", "MELON"), ("attriguard", "AttriGuard"),
    ("camel", "CaMeL"), ("twin", "Twin Agent"), ("toolfilter", "Tool Filter"),
    ("progent", "Progent"), ("drift", "DRIFT"), ("ration", "RATION"),
)
BENCHMARKS = ("dtap", "agentdyn", "agentdojo", "asb")
BENCHMARK_NAMES = {"dtap": "DTap", "agentdyn": "AgentDyn", "agentdojo": "AgentDojo", "asb": "ASB"}


def condition_names(benchmark: str) -> dict[str, str]:
    if benchmark == "dtap":
        return {method: method for method, _ in METHODS}
    registry = importlib.import_module(f"scripts.{benchmark}.conditions")
    return {registry.condition_spec(name).method: name for name in registry.available_conditions()}


def reproduce(*, source: str, run_key: str) -> dict[str, dict[str, dict[str, str]]]:
    table: dict[str, dict[str, dict[str, str]]] = {method: {} for method, _ in METHODS}
    for benchmark in BENCHMARKS:
        build_report = importlib.import_module(f"scripts.{benchmark}.report").build_report
        names = condition_names(benchmark)
        for method, _ in METHODS:
            overall = build_report(root=ROOT, source=source, condition=names[method], run_key=run_key)["overall"]
            table[method][benchmark] = {metric: overall[metric]["display"] for metric in METRICS}
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute the main comparison from compact verdicts")
    parser.add_argument("--source", choices=SOURCES, default="reproduced")
    parser.add_argument("--run-key", default="run1")
    args = parser.parse_args()
    try:
        table = reproduce(source=args.source, run_key=args.run_key)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    print(f"Source: {args.source}; run: {args.run_key}")
    print("Percentages ± 95% normal confidence intervals; decimal rounding: half up.\n")
    headers = ["Defense", *(f"{BENCHMARK_NAMES[benchmark]} {metric}" for benchmark in BENCHMARKS for metric in METRICS)]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for method, label in METHODS:
        cells = [table[method][benchmark][metric] for benchmark in BENCHMARKS for metric in METRICS]
        print("| " + " | ".join([label, *cells]) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
