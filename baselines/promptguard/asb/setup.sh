#!/usr/bin/env bash

set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY="$ROOT/.venv/bin/python"
UV_INDEX_URL=https://pypi.org/simple uv pip install --python "$PY" -q \
  torch==2.3.1 transformers==4.41.2 huggingface-hub==0.36.2 tokenizers==0.19.1 safetensors==0.8.0
PYTHONPATH="$ROOT" "$PY" -c 'from huggingface_hub import snapshot_download; from baselines.promptguard.asb.runtime import MODEL_ID, MODEL_REVISION; snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)'
