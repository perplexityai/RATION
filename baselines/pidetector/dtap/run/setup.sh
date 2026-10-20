#!/usr/bin/env bash
set -euo pipefail
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
VENV=$R/.method-venvs/pidetector
PY=$VENV/bin/python
[ -x "$PY" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$PY" -q 'transformers==4.39.3' 'onnxruntime==1.20.1' 'sentencepiece==0.2.0' 'huggingface-hub==0.26.5'
exec "$PY" "$R/baselines/pidetector/dtap/run/serve.py" --port "${PIDETECTOR_PORT:-8908}"
