from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_method_record(task_dir: str | Path, owner: str, record: dict[str, Any]) -> Path:
    path = Path(os.environ.get("EXPERIMENT_TASK_ARTIFACT_DIR") or task_dir) / "method_evidence" / f"{owner}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    return path
