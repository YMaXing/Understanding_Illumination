"""Second derivative of an image, its absolute value, and original / |second derivative|.

For every image in a folder:
  1. read it as float (16 bit TIFFs scaled to [0, 1], 8 bit images optionally linearized from sRGB)
  2. second derivative = Laplacian of Gaussian (d2/dx2 + d2/dy2 after Gaussian smoothing with --sigma;
     use --sigma 0 for the raw Laplacian, which is dominated by pixel noise)
  3. take the absolute value
  4. ratio = original / (|second derivative| + eps)

Outputs per image stem in <out_dir>:
  npy/<stem>_abs_d2.npy     |second derivative|, float32, same size and channels as the image
  npy/<stem>_ratio.npy      original / (|second derivative| + eps), float32
  <stem>_panel.png          original | |second derivative| | ratio (log scaled for display)

Usage:
  python second_derivative_ratio.py <image_dir> <out_dir> [--sigma 1.0] [--mode rgb|gray] [--eps 1e-4] [--linearize]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def read_float(path, linearize):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[:, :, :3]
    if img.dtype == np.uint16:
        x = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        x = img.astype(np.float32) / 255.0
        if linearize:  # undo the sRGB curve so values are proportional to light
            x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    else:
        x = img.astype(np.float32)
    return x


def second_derivative(x, sigma):
    if sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), sigma)
    return cv2.Laplacian(x, cv2.CV_32F, ksize=3)


def to_display(x, lo_pct=0.5, hi_pct=99.5, log=False):
    if log:
        x = np.log10(np.maximum(x, 1e-12))
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    y = np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
    return y


def to_bgr8(y, gamma=True):
    if gamma:
        y = y ** (1 / 2.2)
    y = (y * 255 + 0.5).astype(np.uint8)
    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR) if y.ndim == 2 else y


def label(img, text):
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian smoothing before the Laplacian; 0 for none")
    ap.add_argument("--mode", choices=["rgb", "gray"], default="rgb",
                    help="rgb: second derivative per channel; gray: of the luminance, shared by all channels")
    ap.add_argument("--eps", type=float, default=1e-4, help="added to |second derivative| to avoid dividing by zero")
    ap.add_argument("--linearize", action="store_true", help="convert 8 bit sRGB images (phone JPEGs) to linear first")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "npy").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"no images in {args.image_dir}")

    for i, path in enumerate(files, 1):
        x = read_float(path, args.linearize)
        if x is None:
            print(f"[{i}/{len(files)}] {path.name}: cannot read, skipping")
            continue
        if args.mode == "gray" and x.ndim == 3:
            lum = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            d2 = np.abs(second_derivative(lum, args.sigma))[:, :, None].repeat(3, axis=2)
        else:
            d2 = np.abs(second_derivative(x, args.sigma))
        ratio = x / (d2 + args.eps)

        np.save(out / "npy" / f"{path.stem}_abs_d2.npy", d2.astype(np.float32))
        np.save(out / "npy" / f"{path.stem}_ratio.npy", ratio.astype(np.float32))

        panels = [
            label(to_bgr8(to_display(x)), "original"),
            label(to_bgr8(to_display(d2.mean(axis=2) if d2.ndim == 3 else d2), gamma=False), "|second derivative|"),
            label(to_bgr8(to_display(ratio.mean(axis=2) if ratio.ndim == 3 else ratio, log=True), gamma=False),
                  "original / |second derivative| (log)"),
        ]
        cv2.imwrite(str(out / f"{path.stem}_panel.png"), np.hstack(panels))
        print(f"[{i}/{len(files)}] {path.name}: |d2| median {np.median(d2):.2e}, ratio median {np.median(ratio):.2e}")

    print(f"done, results in {out}")


if __name__ == "__main__":
    main()
