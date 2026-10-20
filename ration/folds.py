from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def load(path: Path) -> dict[str, int]:
    return json.loads(Path(path).read_text())["folds"]


def fold_of(folds: Mapping[str, int], key: str) -> int:
    return folds[key]
