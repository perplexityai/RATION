from __future__ import annotations

from typing import Any, Mapping

from baselines.pidetector.core import MODEL_ID, MODEL_REVISION, REDACTION_TEXT, THRESHOLD, agentdojo_decision
from scripts.asb.classifier import ClassifierSidecar
from scripts.asb.observations import RedactingDetectorMixin


class PIDetectorAgentMixin(RedactingDetectorMixin):
    redaction_text = REDACTION_TEXT
    sidecar: ClassifierSidecar

    def is_injection(self, output: str) -> bool:
        payload = self.sidecar.classify(output)
        return bool(agentdojo_decision(payload["label"], payload["score"], THRESHOLD)[0])


def agent_mixin(settings: Mapping[str, Any]) -> type:
    return type("PIDetectorAgentMixin", (PIDetectorAgentMixin,), {"sidecar": ClassifierSidecar(MODEL_ID, MODEL_REVISION)})
