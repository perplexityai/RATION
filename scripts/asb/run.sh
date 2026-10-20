#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
PY_ASB="$ROOT/.asb-venv/bin/python"
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
export PYTHONPATH="$ROOT"
cd "$ROOT"

usage(){
  cat <<'USAGE'
scripts/asb/run.sh setup                       build .asb-venv from the pinned ASB commit
scripts/asb/run.sh setup <condition>           install a baseline's extra dependencies
scripts/asb/run.sh run <condition> [runN]      clean + attacked episodes on all ten scenarios (resumable)
scripts/asb/run.sh report <condition> [runN]

Conditions: nodefense ration attriguard-official camel-nopolicy datasentinel-reimplementation
  drift-official instructional-official melon-reimplementation pidetector-reimplementation
  progent-official promptguard-reimplementation sandwich-official spotlighting-reimplementation
  toolfilter-reimplementation twin-reimplementation
USAGE
}

COMMAND="${1:-help}"
CONDITION="${2:-}"
RUN="${3:-run1}"
case "$COMMAND" in
  setup)
    if [ -z "$CONDITION" ]; then
      bash "$ROOT/scripts/vendor_asb.sh"
      [ -x "$PY_ASB" ] || uv venv --python 3.11 "$ROOT/.asb-venv"
      uv pip sync --python "$PY_ASB" "$ROOT/scripts/asb/requirements.lock"
    elif [ -f "$ROOT/baselines/${CONDITION%%-*}/asb/setup.sh" ]; then
      bash "$ROOT/baselines/${CONDITION%%-*}/asb/setup.sh"
    else
      echo "[asb] $CONDITION has no extra dependencies"
    fi
    ;;
  run)
    [ -n "${OPENAI_API_KEY:-}" ] || { echo "[asb] OPENAI_API_KEY is not set (put it in .env)" >&2; exit 3; }
    control="$ROOT/runs/asb/$RUN/${CONDITION:?condition}"
    mkdir -p "$control"
    "$PY_ASB" -m scripts.asb.run_all --condition "$CONDITION" --run-key "$RUN" \
      --proxy-port "${ASB_PROXY_PORT:-8911}" --max-workers "${ASB_AGENT_THREADS:-10}" 2>&1 | tee -a "$control/run.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[asb] run incomplete; see $control/run.log; rerun to resume" >&2; exit 3; }
    "$PY" -m scripts.asb.report --source results --run-key "$RUN" --condition "$CONDITION"
    ;;
  report) "$PY" -m scripts.asb.report --source results --run-key "$RUN" --condition "${CONDITION:?condition}" ;;
  *) usage ;;
esac
