"""Build the weight map for the illumination smoothness loss from the edge labels.

Illumination should be smooth everywhere, except where it is allowed to change:
  - shadow edges   (label 1): the amount of direct light changes
  - geometry edges (label 3): occlusions and surface orientation changes
Material edges (label 2) keep full weight: illumination should NOT jump just because the material changes.

weight = 1 where smoothness is enforced, --min-weight where illumination may change,
with a small dilation and blur so the transition is not a hard one pixel line.

Inputs:  <labels_dir>/<stem>.npy   from classify_edges.py (0 none, 1 shadow, 2 material, 3 geometry)
Outputs in <out_dir>:
  npy/<stem>.npy   float32 weight map, H x W, in [min_weight, 1]
  viz/<stem>.png   white = smoothness enforced, black = illumination may change

Usage:
  python build_smoothness_weights.py <labels_dir> <out_dir> [--min-weight 0.05] [--dilate 2] [--blur 1.0]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def build_weight(labels, min_weight, dilate, blur):
    release = ((labels == 1) | (labels == 3)).astype(np.uint8)
    if dilate > 0:
        release = cv2.dilate(release, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    w = 1.0 - (1.0 - min_weight) * release.astype(np.float32)
    if blur > 0:
        w = cv2.GaussianBlur(w, (0, 0), blur)
    return np.clip(w, min_weight, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--min-weight", type=float, default=0.05, help="weight where illumination may change")
    ap.add_argument("--dilate", type=int, default=2, help="widen released edges by this many pixels")
    ap.add_argument("--blur", type=float, default=1.0, help="soften the weight map")
    args = ap.parse_args()

    files = sorted(Path(args.labels_dir).glob("*.npy"))
    if not files:
        raise SystemExit(f"no label maps in {args.labels_dir}")
    out = Path(args.out_dir)
    (out / "npy").mkdir(parents=True, exist_ok=True)
    (out / "viz").mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(files, 1):
        w = build_weight(np.load(p), args.min_weight, args.dilate, args.blur)
        np.save(out / "npy" / f"{p.stem}.npy", w)
        cv2.imwrite(str(out / "viz" / f"{p.stem}.png"), (w * 255).astype(np.uint8))
        print(f"[{i}/{len(files)}] {p.stem}: {100 * (w < 0.5).mean():.1f}% of pixels released")
    print(f"done, results in {out}")


if __name__ == "__main__":
    main()
