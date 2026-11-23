#!/usr/bin/env bash
# Run a metrics/ script or pytest from any directory, in the metrics uv environment.
#   metrics/run.sh migrate_v0.py run
#   metrics/run.sh -m pytest -q
# PYTHONPATH is dropped (ROS leaks into it) and no bytecode is written (the ℓI hook imports illumination_edge/).
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -lt 1 ]; then echo "usage: $0 SCRIPT.py [args...] | -m MODULE [args...]" >&2; exit 2; fi
cd "$D"
if [ "$1" = "-m" ]; then exec env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 uv run --project "$D" python -B "$@"; fi
script="$1"; shift
case "$script" in /*) ;; *) script="$D/$script" ;; esac
exec env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 uv run --project "$D" python -B "$script" "$@"
