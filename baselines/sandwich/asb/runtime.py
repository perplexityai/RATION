from __future__ import annotations

import copy
from typing import Any, Mapping


class SandwichAgentMixin:
    def run(self):
        self.args = copy.copy(self.args)
        self.args.defense_type = "ob_sandwich_defense"
        return super().run()


def agent_mixin(settings: Mapping[str, Any]) -> type:
    return SandwichAgentMixin
