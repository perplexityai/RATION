from __future__ import annotations

from typing import Any, Mapping

from scripts.asb.classifier import ClassifierSidecar
from scripts.asb.observations import RedactingDetectorMixin

MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
MODEL_REVISION = "a8ded8e697ce7c355e395a0df51f94adb4a2fd27"
SAFE_LABEL = "LABEL_0"
THRESHOLD = 0.5
REDACTION_TEXT = "<Data omitted because a prompt injection was detected>"


class PromptGuardAgentMixin(RedactingDetectorMixin):
    redaction_text = REDACTION_TEXT
    sidecar: ClassifierSidecar

    def is_injection(self, output: str) -> bool:
        payload = self.sidecar.classify(output)
        score = float(payload["score"])
        safety_score = score if payload["label"].lower() == SAFE_LABEL.lower() else 1.0 - score
        return safety_score < THRESHOLD


def agent_mixin(settings: Mapping[str, Any]) -> type:
    return type("PromptGuardAgentMixin", (PromptGuardAgentMixin,), {"sidecar": ClassifierSidecar(MODEL_ID, MODEL_REVISION)})
