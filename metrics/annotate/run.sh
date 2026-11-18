#!/usr/bin/env bash
# Start the annotation tool: prepares any image missing from the cache, then serves http://127.0.0.1:8765/
#   metrics/annotate/run.sh [--port N]
# Open the printed URL in a browser (on WSL, the Windows browser reaches it as localhost). Ctrl+C stops.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$D/../run.sh" annotate/prepare.py >/dev/null
exec env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 uv run --project "$D/.." python -B "$D/server.py" "$@"
