#!/usr/bin/env bash

set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY="$ROOT/.asb-venv/bin/python"
DEPS="$ROOT/.method-deps/progent-asb"
SPEC="secagent @ git+https://github.com/sunblaze-ucb/progent@8a8eb894b9d568a32b4e58252fea85b47f789c77"

mkdir -p "$DEPS"
UV_INDEX_URL=https://pypi.org/simple uv pip install --python "$PY" --target "$DEPS" -q --no-deps \
  --reinstall-package secagent "$SPEC" "z3-solver==4.15.4.0"
UV_INDEX_URL=https://pypi.org/simple uv pip install --python "$PY" --target "$DEPS" -q jsonschema
"$PY" - "$DEPS" <<'PY'
import importlib.metadata as md, pathlib, shutil, sys
deps = pathlib.Path(sys.argv[1])
have = {(d.metadata.get("Name") or "").lower().replace("_", "-") for d in md.distributions()}
for info in deps.glob("*.dist-info"):
    name = info.name.rsplit("-", 1)[0].lower().replace("_", "-")
    if name in have:
        for entry in [info, *deps.glob(name.replace("-", "_") + "*"), *deps.glob(name + "*")]:
            if entry.exists() and entry.is_relative_to(deps):
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
PY
