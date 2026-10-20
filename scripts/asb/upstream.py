from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass(frozen=True)
class Scenario:
    name: str
    agent_path: str
    task: str
    tools: tuple[dict[str, Any], ...]


def inventory(*, root: Path = ROOT) -> tuple[Scenario, ...]:
    data = root / "benchmarks/asb/upstream_data"
    tools = _jsonl(data / "all_attack_tools.jsonl")
    return tuple(
        Scenario(
            name=agent["agent_name"],
            agent_path=agent["agent_path"],
            task=agent["tasks"][0],
            tools=tuple(tool for tool in tools if tool["Corresponding Agent"] == agent["agent_name"]),
        )
        for agent in _jsonl(data / "agent_task.jsonl")
    )
