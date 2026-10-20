#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY="$ROOT/.agentdyn-venv/bin/python"
SUITES="shopping github dailylife"
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
export PYTHONPATH="$ROOT"
cd "$ROOT"

usage(){
  cat <<'USAGE'
scripts/agentdyn/run.sh setup                      build .agentdyn-venv from the pinned AgentDyn commit
scripts/agentdyn/run.sh setup <condition>          install a baseline's extra dependencies
scripts/agentdyn/run.sh run <condition> [runN]     benign + attacked runs on all suites (resumable)
scripts/agentdyn/run.sh report <condition> [runN]

Conditions: nodefense ration attriguard-official camel-nopolicy datasentinel-reimplementation
  drift-official instructional-asb melon-reimplementation pidetector-official progent-official
  promptguard-official sandwich-official spotlighting-official toolfilter-official twin-reimplementation
USAGE
}

run_taskset(){
  local taskset="$1" condition="$2" run="$3" suite rc=0
  local control="$ROOT/runs/agentdyn/$run/control/$condition"
  local -a pids=()
  mkdir -p "$control"
  for suite in $SUITES; do
    "$PY" -m scripts.agentdyn.execution "$taskset" "$condition" "$suite" "$run" > "$control/$taskset-$suite.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  [ "$rc" -eq 0 ] || { echo "[agentdyn] $taskset incomplete; see $control; rerun to resume" >&2; exit 3; }
}

COMMAND="${1:-help}"
CONDITION="${2:-}"
RUN="${3:-run1}"
case "$COMMAND" in
  setup)
    if [ -z "$CONDITION" ]; then
      bash "$ROOT/scripts/vendor_agentdyn.sh"
      [ -x "$PY" ] || uv venv --python 3.12 "$ROOT/.agentdyn-venv"
      uv pip install --python "$PY" --reinstall "$ROOT/vendor/agentdyn"
      uv pip install --python "$PY" "openai>=1.60"
      exit 0
    fi
    setup=$("$PY" -c 'import sys; from scripts.agentdyn.conditions import condition_spec; print(condition_spec(sys.argv[1]).setup or "")' "$CONDITION")
    if [ -n "$setup" ]; then PY="$PY" bash "$ROOT/$setup"; else echo "[agentdyn] $CONDITION has no extra dependencies"; fi
    ;;
  run)
    [ -n "${OPENAI_API_KEY:-}" ] || { echo "[agentdyn] OPENAI_API_KEY is not set (put it in .env)" >&2; exit 3; }
    run_taskset benign "${CONDITION:?condition}" "$RUN"
    run_taskset indirect "$CONDITION" "$RUN"
    "$PY" -m scripts.agentdyn.report --source results --run-key "$RUN" --condition "$CONDITION"
    ;;
  report) "$PY" -m scripts.agentdyn.report --source results --run-key "$RUN" --condition "${CONDITION:?condition}" ;;
  *) usage ;;
esac
