from __future__ import annotations

from baselines.melon.dtap.impl.sensitive_args import REFERENCE_FOCUS

FOCUS = dict(REFERENCE_FOCUS)
MASK_TEXT = "<Data omitted because a prompt injection was detected>"
STOP_TEXT = "The tool outputs contain potential prompt injection, stopping execution."
MAX_ITERS = 15
MASKED_RUN_MAX_TOKENS = 32768
