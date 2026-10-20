from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Mapping

import yaml


def load_yaml(path: str | Path):
    return yaml.safe_load(Path(path).read_text())


def _feedback_path() -> Path:
    policy = Path(os.environ["RATION_TABLE"])
    if policy.is_dir():
        return policy / "feedback.yaml"
    return policy.parent / load_yaml(policy)["adapter_artifacts"]["feedback"]


@functools.lru_cache(maxsize=None)
def _feedback_templates(path: str) -> Mapping[str, str]:
    return load_yaml(path)["feedback"]


def feedback_templates() -> Mapping[str, str]:
    return dict(_feedback_templates(str(_feedback_path())))
