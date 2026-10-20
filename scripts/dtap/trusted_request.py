from __future__ import annotations

import sys
from pathlib import Path


def task_dir_from_argv(argv: list[str] | None = None) -> Path | None:
    args = sys.argv if argv is None else argv
    for index, arg in enumerate(args):
        if arg == "--task-dir" and index + 1 < len(args):
            return Path(args[index + 1])
        if arg.startswith("--task-dir="):
            return Path(arg.split("=", 1)[1])
    return None


def trusted_request(task_dir: str | Path) -> str:
    cfg = Path(task_dir) / "config.yaml"
    if not cfg.is_file():
        raise FileNotFoundError(f"trusted task config is missing: {cfg}")
    from dt_arena.src.types.task import TaskConfig

    request = TaskConfig.from_yaml(str(cfg)).original_instruction
    if not isinstance(request, str) or not request.strip():
        raise ValueError(f"trusted task instruction is missing or not a single string: {cfg}")
    return request
