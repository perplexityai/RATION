#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
. "$ROOT/scripts/vendor_upstream.sh"
. "$ROOT/benchmarks/agentdyn/upstream.conf"
vendor_upstream_fetch agentdyn "$AGENTDYN_REPOSITORY" "$PINNED_SHA"
