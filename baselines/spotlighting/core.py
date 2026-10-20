from __future__ import annotations

import json
from pathlib import Path

_STRINGS = json.loads((Path(__file__).resolve().parent / "agentdojo_strings.json").read_text(encoding="utf-8"))
SYSTEM_SUFFIX: str = _STRINGS["system_suffix"]
WRAP_TEMPLATE: str = _STRINGS["wrap_template"]
DELIM_OPEN, DELIM_CLOSE = WRAP_TEMPLATE.split("{text}")


def spotlight(tool_output: str) -> str:
    return WRAP_TEMPLATE.format(text=tool_output)


def system_prompt_with_suffix(system_prompt: str) -> str:
    base = system_prompt or ""
    if base.endswith(SYSTEM_SUFFIX):
        return base
    return base + SYSTEM_SUFFIX


def is_spotlighted(system_prompt: str) -> bool:
    return bool(system_prompt) and system_prompt.endswith(SYSTEM_SUFFIX)
