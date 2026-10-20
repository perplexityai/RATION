#!/usr/bin/env bash

set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
. "$ROOT/scripts/vendor_upstream.sh"
. "$ROOT/benchmarks/asb/upstream.conf"

vendor_upstream_fetch asb "$ASB_REPOSITORY" "$PINNED_SHA"
