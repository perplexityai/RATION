#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY="${PY:-$ROOT/.venv/bin/python}"
UV_INDEX_URL=https://pypi.org/simple uv pip install --python "$PY" -q "json-repair==0.55.1"
