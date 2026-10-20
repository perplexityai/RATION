from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.llm_client import OPENAI_MODEL_IDS

DTAP_HOME = ROOT / "vendor" / "dtap"
SETTINGS = json.loads((ROOT / "benchmarks/dtap/protocol.json").read_text())
AGENT_TYPE = SETTINGS["agent_type"]
MODEL = SETTINGS["model"]
TASKSETS = ("indirect", "benign")
ENVIRONMENTS = (
    "travel", "telecom", "research", "code", "medical", "browser", "os-filesystem",
    "customer-service", "finance", "crm", "legal", "workflow", "windows", "macos",
)
INTERNAL_MODEL_VARIABLES = {
    "customer-service": ("CS_LLM_JUDGE_MODEL",),
    "medical": ("PATIENT_MODEL", "MEASUREMENT_MODEL", "JUDGE_MODEL"),
    "workflow": ("OPENAI_MODEL",),
    "research": ("OPENAI_MODEL",),
    "os-filesystem": ("PERMISSION_LLM_MODEL",),
}


def tasklist(environment: str, taskset: str, *, root: Path = ROOT) -> list[dict]:
    path = root / "benchmarks/dtap/environments" / environment / "tasklists" / f"{taskset}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def task_identity(row: dict, taskset: str) -> tuple[str, str]:
    category = row.get("risk_category") or ""
    if taskset == "benign":
        return f"{category or 'benign'}/{row['task_id']}", ""
    return str(row["task_id"]), category


def environment_model_env(environment: str) -> dict[str, str]:
    model = OPENAI_MODEL_IDS[MODEL] if environment == "medical" else MODEL
    return {name: model for name in INTERNAL_MODEL_VARIABLES.get(environment, ())}


def proxy_aliases() -> dict[str, str]:
    return {name: MODEL for name in ("gpt-4o", "gpt-4o-mini", OPENAI_MODEL_IDS[MODEL], MODEL)}


if __name__ == "__main__":
    print(json.dumps(proxy_aliases(), sort_keys=True, separators=(",", ":")))
