#!/usr/bin/env bash
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/vendor_upstream.sh"
PINNED_SHA="e0323a521ba4ef88f8e14c1eccf68d0a3d19a458"
vendor_upstream_fetch dtap "https://github.com/AI-secure/DecodingTrust-Agent" "$PINNED_SHA"
