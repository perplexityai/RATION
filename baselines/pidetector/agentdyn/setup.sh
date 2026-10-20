#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY="${PY:-$ROOT/.agentdyn-venv/bin/python}"
UV_INDEX_URL=https://pypi.org/simple uv pip install --python "$PY" -q \
  -r "$ROOT/baselines/pidetector/agentdojo/requirements.txt"
PYTHONPATH="$ROOT" "$PY" -c \
  'from baselines.pidetector.agentdojo.checkpoint import resolve_snapshot; resolve_snapshot(download=True)'
