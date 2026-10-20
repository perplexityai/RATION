from __future__ import annotations

MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"
MODEL_REVISION = "89b085cd330414d3e7d9dd787870f315957e1e9f"
SAFE_LABEL = "SAFE"
THRESHOLD = 0.5
TRAINED_CONTEXT_TOKENS = 512
REDACTION_TEXT = "<Data omitted because a prompt injection was detected>"


def agentdojo_decision(label: str, score: float, threshold: float = THRESHOLD) -> tuple[bool, float]:
    safety_score = float(score) if label == SAFE_LABEL else 1.0 - float(score)
    return safety_score < float(threshold), safety_score
