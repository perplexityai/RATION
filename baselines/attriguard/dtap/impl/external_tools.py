from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

LOCAL_EXTERNAL_TOOLS = frozenset({"load_skill"})


@lru_cache(maxsize=None)
def classification_for_domain(domain: str) -> tuple[frozenset, frozenset]:
    tools = json.loads((Path(__file__).resolve().parent / "external_tools.json").read_text())["domains"][domain]
    return frozenset(tools), frozenset(name for name, value in tools.items() if value["external"])
