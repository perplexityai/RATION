from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agentdojo.report import benchmark_report
from scripts.agentdyn.conditions import condition_spec
from scripts.common.reporting import report_main


def build_report(*, root: Path, source: str, condition: str, run_key: str) -> dict:
    return benchmark_report("agentdyn", condition_spec(condition).result_parts, root=root, source=source,
                            condition=condition, run_key=run_key)


if __name__ == "__main__":
    raise SystemExit(report_main("agentdyn", build_report))
