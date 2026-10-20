from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Iterable

from scripts.common.metrics import Estimate

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("ASR", "UA", "BU")
SOURCES = {"reproduced": "reproduced_results", "results": "results"}


def result_directory(root: Path, source: str, benchmark: str, parts: Iterable[str], run_key: str) -> Path:
    return root / SOURCES[source] / benchmark / Path(*parts) / "main_results" / run_key


def make_report(
    *, benchmark: str, condition: str, source: str, run_key: str,
    basis: str, units: dict[str, dict[str, Estimate]], overall: dict[str, Estimate],
) -> dict:
    return {
        "benchmark": benchmark, "condition": condition, "source": source,
        "run_key": run_key, "basis": basis,
        "confidence_interval": "95% normal approximation, independent groups",
        "units": {name: {key: value.as_dict() for key, value in metrics.items()}
                  for name, metrics in units.items()},
        "overall": {key: value.as_dict() for key, value in overall.items()},
    }


def report_main(benchmark: str, build_report: Callable[..., dict]) -> int:
    parser = argparse.ArgumentParser(description=f"Recompute {benchmark} compact results")
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()
    try:
        report = build_report(root=ROOT, source=args.source, run_key=args.run_key,
                              condition=args.condition)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    print(f"{benchmark} source={args.source} run={args.run_key} condition={args.condition}")
    print(f"basis={report['basis']}; percentages ± 95% CI")
    for name, metrics in (*report["units"].items(), ("overall", report["overall"])):
        values = "  ".join(f"{key}={metrics[key]['display']}" for key in METRICS)
        print(f"  {name}: {values}")
    return 0
