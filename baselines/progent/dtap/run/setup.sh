#!/usr/bin/env bash
set -euo pipefail
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
PIN=8a8eb894b9d568a32b4e58252fea85b47f789c77
UV_INDEX_URL=https://pypi.org/simple uv pip install --python "$R/.dtap-venv/bin/python" --target "$R/.method-deps/progent" \
  -q --no-deps --reinstall-package secagent "secagent @ git+https://github.com/sunblaze-ucb/progent@$PIN" "z3-solver==4.15.4.0"
