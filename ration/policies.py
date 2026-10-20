from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def for_fold(policies: Mapping[str, Any], fold: int) -> dict[str, Any]:
    return policies["folds"][fold]
