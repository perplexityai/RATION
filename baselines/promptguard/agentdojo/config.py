from __future__ import annotations

MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
MODEL_REVISION = "a8ded8e697ce7c355e395a0df51f94adb4a2fd27"
SAFE_LABEL = "LABEL_0"
MALICIOUS_LABEL = "LABEL_1"
LABEL_MEANING = {SAFE_LABEL: "BENIGN", MALICIOUS_LABEL: "MALICIOUS"}
CONTEXT_WINDOW_TOKENS = 512
THRESHOLD = 0.5
REDACTION_TEXT = "<Data omitted because a prompt injection was detected>"
