#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY="${PY:-$ROOT/.venv/bin/python}"
UV_INDEX_URL=https://pypi.org/simple uv pip install --python "$PY" -q \
  -r "$ROOT/baselines/promptguard/agentdojo/requirements.txt"
PYTHONPATH="$ROOT" "$PY" -c \
  'from baselines.promptguard.agentdojo.checkpoint import resolve_snapshot; resolve_snapshot(download=True)'
