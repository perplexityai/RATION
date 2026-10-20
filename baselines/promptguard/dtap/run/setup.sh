#!/usr/bin/env bash
set -euo pipefail
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
VENV=$R/.method-venvs/promptguard
PY=$VENV/bin/python
[ -x "$PY" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$PY" -q 'torch==2.3.1' --index-url https://download.pytorch.org/whl/cpu \
  || uv pip install --python "$PY" -q 'torch==2.3.1'
uv pip install --python "$PY" -q 'transformers==4.41.2' 'tokenizers==0.19.1' 'huggingface-hub==0.36.2' \
  'safetensors==0.8.0' 'numpy==1.26.4'
SNAPSHOT=$(PYTHONPATH="$R" "$PY" -c 'from baselines.promptguard.agentdojo.checkpoint import resolve_snapshot; print(resolve_snapshot(download=True))')
exec env PYTHONPATH="$R" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" "$R/baselines/promptguard/dtap/run/serve.py" --model-dir "$SNAPSHOT" --port "${PROMPTGUARD_PORT:-8909}"
