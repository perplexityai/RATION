from __future__ import annotations

import re
from pathlib import Path

ENVS = Path(__file__).resolve().parents[2] / "vendor/dtap/dt_arena/envs"
HOSPITAL_BASE_URL = "      - OPENAI_BASE_URL=${HOSPITAL_OPENAI_BASE_URL:-http://host.docker.internal:8903/openrouter.ai/v1}\n"
TERMINAL_RESET = "      - ${DTAP_TERMINAL_RESET:?set DTAP_TERMINAL_RESET}:/scripts/reset.sh:ro\n"
TERMINAL_DATA = "      - ./data:/home/terminal-user/data\n"


def substitute(pattern: str, replacement: str, text: str) -> str:
    patched, count = re.subn(pattern, replacement, text, count=1)
    if not count:
        raise ValueError(f"compose pattern not found: {pattern!r}")
    return patched


def patch_travel(text: str) -> str:
    if "restart:" in text:
        return text
    return substitute(r"(\n(\s+)travel-api:\n)", r"\1\2  restart: unless-stopped\n", text)


def patch_hospital(text: str) -> str:
    if "OPENAI_BASE_URL" not in text:
        text = substitute(r"      - OPENAI_API_KEY=\$\{OPENAI_API_KEY:-\}\n",
                          "      - OPENAI_API_KEY=${OPENAI_API_KEY:-proxy}\n" + HOSPITAL_BASE_URL.replace("\\", "\\\\"),
                          text)
    if "host-gateway" not in text:
        text = substitute(r"(\n  hospital-server:\n)", "\\1    extra_hosts:\n      - \"host.docker.internal:host-gateway\"\n", text)
    return text


def patch_terminal(text: str) -> str:
    if TERMINAL_RESET in text:
        return text
    text = "".join(line for line in text.splitlines(keepends=True) if not line.rstrip().endswith(":/scripts/reset.sh:ro"))
    return substitute(re.escape(TERMINAL_DATA), (TERMINAL_DATA + TERMINAL_RESET).replace("\\", "\\\\"), text)


for relative, patch in (("travel/docker-compose.yml", patch_travel),
                        ("hospital/docker-compose-hub.yml", patch_hospital),
                        ("terminal/docker-compose-hub.yml", patch_terminal)):
    path = ENVS / relative
    path.write_text(patch(path.read_text()))
