#!/usr/bin/env bash
set -euo pipefail
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
UV_INDEX_URL=https://pypi.org/simple uv pip install --python "$R/.dtap-venv/bin/python" --target "$R/.method-deps/drift" \
  -q --no-deps --reinstall-package json-repair "json-repair==0.55.1"
