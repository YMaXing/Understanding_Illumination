"""Run Depth Anything 3 (monocular) on a folder of images and save depth maps.

Linear 16 bit TIFFs are tone mapped to 8 bit input first (same display mapping
as the point annotator: 0.5/99.5 percentile stretch shared across channels,
then gamma 1/2.2). The original images are never modified.

Outputs, per image stem:
  <out>/npy/<stem>.npy          float32 relative depth, original HxW (for the pipeline)
  <out>/preview/<stem>.png      colored depth visualization (for looking at only)
  <out>/input/<stem>.png        the tone mapped 8 bit image the network saw
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch

EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def tone_map(img):
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0) ** (1 / 2.2)
    return (x * 255 + 0.5).astype(np.uint8)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="depth-anything/da3mono-large")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from depth_anything_3.api import DepthAnything3

    files = sorted(p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"no images found in {args.image_dir}")
    out = Path(args.out_dir)
    for sub in ("npy", "preview", "input"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"loading {args.model} on {device}")
    model = DepthAnything3.from_pretrained(args.model).to(device=device).eval()

    for i, path in enumerate(files, 1):
        npy_path = out / "npy" / f"{path.stem}.npy"
        if npy_path.exists() and not args.overwrite:
            print(f"[{i}/{len(files)}] {path.name}: exists, skipping")
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 3:
            print(f"[{i}/{len(files)}] {path.name}: cannot read as a color image, skipping")
            continue
        rgb = cv2.cvtColor(tone_map(raw[:, :, :3]), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        with torch.no_grad():
            pred = model.inference([rgb])
        depth = np.asarray(pred.depth[0], dtype=np.float32)
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        np.save(npy_path, depth)
        cv2.imwrite(str(out / "input" / f"{path.stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        lo, hi = np.percentile(depth, 1), np.percentile(depth, 99)
        vis = (np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(out / "preview" / f"{path.stem}.png"), cv2.applyColorMap(255 - vis, cv2.COLORMAP_INFERNO))
        print(f"[{i}/{len(files)}] {path.name}: depth {depth.min():.3f} to {depth.max():.3f}")

    print(f"done, results in {out}")


if __name__ == "__main__":
    main()
