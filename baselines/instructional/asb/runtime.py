from __future__ import annotations

import copy
from typing import Any, Mapping


class InstructionalAgentMixin:
    def run(self):
        self.args = copy.copy(self.args)
        self.args.defense_type = "instructional_prevention"
        return super().run()


def agent_mixin(settings: Mapping[str, Any]) -> type:
    return InstructionalAgentMixin
