from __future__ import annotations

from typing import Any, Mapping

from baselines.datasentinel.core import REDACTION_TEXT, DetectorClient
from scripts.asb.observations import RedactingDetectorMixin


class DataSentinelAgentMixin(RedactingDetectorMixin):
    redaction_text = REDACTION_TEXT
    client: DetectorClient

    def is_injection(self, output: str) -> bool:
        return output != "" and self.client.detect(output, max_prompt_tokens=None).contaminated


def agent_mixin(settings: Mapping[str, Any]) -> type:
    return type("DataSentinelAgentMixin", (DataSentinelAgentMixin,), {"client": DetectorClient()})
