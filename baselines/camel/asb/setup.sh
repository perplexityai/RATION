#!/usr/bin/env bash

set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
VENV="$ROOT/.method-venvs/camel-asb"
[ -x "$VENV/bin/python" ] || uv venv --python 3.11 "$VENV"
UV_INDEX_URL=https://pypi.org/simple uv pip install --python "$VENV/bin/python" -q \
  --reinstall-package camel -r "$ROOT/baselines/camel/asb/requirements.txt"
