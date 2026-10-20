#!/usr/bin/env bash
set -uo pipefail
VU_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

vendor_upstream_fetch() {
  local name="${1:?name}" url="${2:?repo url}" pin="${3:?pinned sha}"
  local dest="$VU_REPO/vendor/$name"
  mkdir -p "$VU_REPO/vendor"
  if [ ! -d "$dest/.git" ]; then
    git clone --quiet "$url" "$dest" || return 1
  fi
  git -C "$dest" checkout --quiet "$pin" 2>/dev/null \
    || { git -C "$dest" fetch --quiet origin && git -C "$dest" checkout --quiet "$pin"; } \
    || { echo "[vendor:$name] pinned commit $pin is not reachable from $url" >&2; return 1; }
}
