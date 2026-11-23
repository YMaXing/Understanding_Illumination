#!/usr/bin/env bash
# Maintainer only: build the detector maps (geometry, ISD, raw step, default-arm weights, material edges) for the 100
# images. Needs a CUDA GPU (Depth Anything 3), the geometry/dedicated and log_chromaticity environments and the DA3
# checkout. Fails early, before any work, if one is missing. Annotators never need this (docs/SETUP.md).
#   metrics/tools/build_maps.sh [OUT]          # default OUT: loss/outputs/split100
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="${1:-loss/outputs/split100}"
fail() { echo "build_maps: $1" >&2; echo "  (detector maps are a maintainer step; annotators do not need them: docs/SETUP.md)" >&2; exit 2; }
command -v nvidia-smi >/dev/null 2>&1 || fail "no NVIDIA GPU driver found (nvidia-smi missing): geom_v1_1 needs Depth Anything 3 on a CUDA GPU"
nvidia-smi -L 2>/dev/null | grep -q "GPU" || fail "nvidia-smi lists no GPU"
[ -d geometry/dedicated/.venv ] || fail "geometry/dedicated environment missing (cd geometry/dedicated && uv sync)"
[ -d DA3/Depth-Anything-3 ] || fail "DA3/Depth-Anything-3 checkout missing"
[ -f log_chromaticity/material_edges.py ] || fail "log_chromaticity/ missing (git-ignored; copy it from the project share)"
[ -f loss/unified_weight.py ] || fail "loss/unified_weight.py missing"
DATA="$(env -u PYTHONPATH uv run --project metrics python -B -c "import sys; sys.path.insert(0,'metrics'); from core import config as K; print(K.repo_path(K.load_config()['dataset']['val_set']))")"
STEMS=$(python3 -c "import json; print(' '.join(x['stem'] for x in json.load(open('metrics/splits/split100.json'))['images']))")
echo "GPU: $(nvidia-smi -L | head -1)"; echo "images: $DATA ($(echo $STEMS | wc -w) stems) -> $OUT"
loss/run.sh unified_weight.py run --images "$DATA" --stems $STEMS --out "$OUT"
illumination_edge/run.sh glossy_h35sb.py run --stems $STEMS --img-dir "$DATA" --isd-dir "$OUT/isd" \
    --raw-dir "$OUT/candidates/step_raw" --geom-dir "$OUT/geometry/stage_b/geom_v1_1" --out "$OUT/det"
echo "done. Point maps.* in metrics/config.local.yaml at $OUT (docs/SETUP.md, maintainer section), then:"
echo "  metrics/run.sh annotate/prepare.py && metrics/run.sh tools/overlays.py export --out overlays_\$(date +%F).tar.gz"
