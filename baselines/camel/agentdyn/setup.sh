#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY="${PY:-$ROOT/.agentdyn-venv/bin/python}"
UV_INDEX_URL=https://pypi.org/simple uv pip install --python "$PY" -q --no-deps \
  --reinstall-package camel -r "$ROOT/baselines/camel/agentdyn/requirements.txt"
present=()
for dist in logfire opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
    opentelemetry-exporter-otlp-proto-common opentelemetry-proto opentelemetry-instrumentation \
    opentelemetry-instrumentation-httpx opentelemetry-semantic-conventions opentelemetry-util-http; do
  "$PY" -c "import importlib.metadata as m; m.version('$dist')" >/dev/null 2>&1 && present+=("$dist")
done
if [ "${#present[@]}" -gt 0 ]; then
  uv pip uninstall --python "$PY" -q "${present[@]}"
fi
