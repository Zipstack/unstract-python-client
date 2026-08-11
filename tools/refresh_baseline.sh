#!/usr/bin/env bash
# Refresh the vendored parity baseline in tests/baseline/.
#
# The compat suite compares this client against the last RELEASED one, not
# against the working tree — a baseline that moves with local edits measures
# nothing. It is vendored rather than downloaded at test time so the suite stays
# offline, and refreshing it is a deliberate act with a reviewable diff.
#
#   ./tools/refresh_baseline.sh 1.5.3
set -euo pipefail

VERSION="${1:?usage: refresh_baseline.sh <released-version>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLUG="${VERSION//./_}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

(cd "$WORK" && pip download "unstract-client==$VERSION" --no-deps -q && unzip -o -q ./*.whl -d x)

OUT="$REPO/tests/baseline/client_$SLUG.py"
{
  echo "# Vendored from the released unstract-client $VERSION wheel on PyPI. DO NOT EDIT."
  echo "# Refresh with tools/refresh_baseline.sh when the parity baseline is intentionally moved."
  cat "$WORK/x/unstract/api_deployments/client.py"
} > "$OUT"

echo "wrote $OUT"
echo "update BASELINE_VERSION in tests/test_compat.py to match"
