#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PROXY_PORT="${PROXY_PORT:-8903}"
PROXY="http://127.0.0.1:$PROXY_PORT/openrouter.ai/v1"
export EVAL_TASK_TIMEOUT="${EVAL_TASK_TIMEOUT:-1800}"
PY="$ROOT/.dtap-venv/bin/python"
DTAP="$ROOT/vendor/dtap"
export UV_EXCLUDE_NEWER=false UV_INDEX_URL=https://pypi.org/simple

setup() {
  local service dir
  bash scripts/vendor_dtap.sh
  python3 scripts/dtap/patch_compose.py
  [ -x "$PY" ] || uv venv -q --python 3.12 .dtap-venv
  uv pip sync -q --python "$PY" scripts/dtap/evaluator-requirements.lock
  uv pip install -q --python "$PY" --no-deps -e "$DTAP"
  for service in $(python3 -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1]))["services"]))' \
      "benchmarks/dtap/environments/$1/environment.json"); do
    dir="$DTAP/dt_arena/mcp_server/$service"
    [ -f "$dir/pyproject.toml" ] || continue
    if [ -f "$dir/uv.lock" ]; then
      (cd "$dir" && uv sync --frozen -q)
    else
      (cd "$dir" && uv sync -q && rm -f uv.lock)
    fi
  done
}

proxy_up() { curl -fsS "http://127.0.0.1:$PROXY_PORT/v1/models" >/dev/null 2>&1; }

start_proxy() {
  : "${OPENAI_API_KEY:?set OPENAI_API_KEY in .env}"
  proxy_up && return 0
  mkdir -p runs
  local aliases
  aliases=$("$PY" scripts/dtap/protocol.py)
  PROXY_MODEL_ALIASES_JSON="$aliases" nohup "$PY" -m scripts.common.openai_proxy --host 0.0.0.0 \
    --port "$PROXY_PORT" >> "runs/openai_proxy_$PROXY_PORT.log" 2>&1 &
  for _ in $(seq 1 50); do proxy_up && return 0; sleep 0.2; done
  echo "the proxy did not start; see runs/openai_proxy_$PROXY_PORT.log" >&2
  exit 3
}

usage() {
  cat <<'USAGE'
scripts/dtap/run.sh setup <environment>                      vendor DTap, build .dtap-venv and the environment's MCP servers
scripts/dtap/run.sh run <condition> <environment> [runN]     benign + attacked tasks of one environment (resumable)
scripts/dtap/run.sh report <condition> [runN]                all fourteen environments

Conditions: nodefense ration attriguard camel datasentinel drift instructional melon pidetector
  progent promptguard sandwich spotlighting toolfilter twin
Environments: travel telecom research code medical browser os-filesystem customer-service finance
  crm legal workflow windows macos
USAGE
}

RUN="${4:-run1}"
case "${1:-}" in
  setup) setup "${2:?environment}" ;;
  run) start_proxy; "$PY" -m scripts.dtap.run "${2:?condition}" "${3:?environment}" --mp "${DTAP_PARALLEL:-3}" \
         --run "${RUN#run}" --proxy "$PROXY" ;;
  report) python3 scripts/dtap/report.py --source results --run-key "${3:-run1}" --condition "${2:?condition}" ;;
  *) usage; exit 2 ;;
esac
