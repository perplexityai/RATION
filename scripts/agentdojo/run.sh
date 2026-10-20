#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
SUITES="banking slack travel workspace"
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
export PYTHONPATH="$ROOT"
cd "$ROOT"

usage(){
  cat <<'USAGE'
scripts/agentdojo/run.sh setup <condition>         install a baseline's extra dependencies
scripts/agentdojo/run.sh run <condition> [runN]    benign + attacked runs on all suites (resumable)
scripts/agentdojo/run.sh report <condition> [runN]

Conditions: nodefense ration attriguard-official camel-normal datasentinel-reimplementation
  drift-official instructional-asb melon-reimplementation pidetector-official progent-official
  promptguard-official sandwich-official spotlighting-official toolfilter-official twin-reimplementation
USAGE
}

run_taskset(){
  local taskset="$1" condition="$2" run="$3" suite rc=0
  local control="$ROOT/runs/agentdojo/$run/control/$condition"
  local -a pids=()
  mkdir -p "$control"
  for suite in $SUITES; do
    "$PY" -m scripts.agentdojo.execution "$taskset" "$condition" "$suite" "$run" > "$control/$taskset-$suite.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  [ "$rc" -eq 0 ] || { echo "[agentdojo] $taskset incomplete; see $control; rerun to resume" >&2; exit 3; }
}

COMMAND="${1:-help}"
CONDITION="${2:-}"
RUN="${3:-run1}"
case "$COMMAND" in
  setup)
    setup=$("$PY" -c 'import sys; from scripts.agentdojo.conditions import condition_spec; print(condition_spec(sys.argv[1]).setup or "")' "${CONDITION:?condition}")
    if [ -n "$setup" ]; then PY="$PY" bash "$ROOT/$setup"; else echo "[agentdojo] $CONDITION has no extra dependencies"; fi
    ;;
  run)
    [ -n "${OPENAI_API_KEY:-}" ] || { echo "[agentdojo] OPENAI_API_KEY is not set (put it in .env)" >&2; exit 3; }
    run_taskset benign "${CONDITION:?condition}" "$RUN"
    run_taskset indirect "$CONDITION" "$RUN"
    "$PY" -m scripts.agentdojo.report --source results --run-key "$RUN" --condition "$CONDITION"
    ;;
  report) "$PY" -m scripts.agentdojo.report --source results --run-key "$RUN" --condition "${CONDITION:?condition}" ;;
  *) usage ;;
esac
