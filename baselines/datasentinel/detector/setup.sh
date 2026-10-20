#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
HERE="$ROOT/baselines/datasentinel/detector"
VENV="${DATASENTINEL_VENV:-$ROOT/.method-venvs/datasentinel}"
PY="$VENV/bin/python"
STORE="${DATASENTINEL_MODELS:-$ROOT/.models/datasentinel}"
BASE="$STORE/mistral-7b-v0.1"
ADAPTER="$STORE/detector_small_checkpoint-500"
PORT="${DATASENTINEL_PORT:-8904}"
BASE_ID=mistralai/Mistral-7B-v0.1
BASE_REVISION=27d67f1b5f57dc0953326b2601d68371d40ea8da
RELEASE_DRIVE_FILE_ID=1B0w5r5udH3I_aiZL0_-2a8WzBAqjuLsn
RELEASE_MEMBER=detector_small/checkpoint-500

ensure_venv() {
  command -v uv >/dev/null || { echo "uv is required" >&2; exit 3; }
  [ -x "$PY" ] || uv venv --python 3.12 "$VENV"
  uv pip install --python "$PY" -q -r "$HERE/requirements.txt"
}

fetch() {
  ensure_venv
  mkdir -p "$STORE"
  if [ ! -f "$BASE/.complete" ]; then
    "$PY" -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], max_workers=4)' \
      "$BASE_ID" "$BASE_REVISION" "$BASE"
    touch "$BASE/.complete"
  fi
  if [ ! -f "$ADAPTER/.complete" ]; then
    stage=$(mktemp -d "$STORE/.stage.XXXXXX")
    "$PY" -m gdown "$RELEASE_DRIVE_FILE_ID" -O "$stage/release.zip"
    (cd "$stage" && unzip -q release.zip -x '__MACOSX/*' '*/.DS_Store')
    rm -rf "$ADAPTER"; mv "$stage/$RELEASE_MEMBER" "$ADAPTER"; rm -rf "$stage"
    touch "$ADAPTER/.complete"
  fi
}

serve() {
  ensure_venv
  [ -f "$BASE/.complete" ] && [ -f "$ADAPTER/.complete" ] || { echo "artifacts are missing; run: $0 fetch" >&2; exit 3; }
  cd "$ROOT"
  exec env PYTHONPATH="$ROOT" "$PY" "$HERE/serve.py" \
    --base-path "$BASE" --checkpoint-path "$ADAPTER" \
    --host 127.0.0.1 --port "$PORT" --replicas "${DATASENTINEL_REPLICAS:-1}"
}

case "${1:-}" in
  fetch) fetch ;;
  serve) serve ;;
  *) echo "usage: baselines/datasentinel/detector/setup.sh fetch | serve" >&2; exit 2 ;;
esac
