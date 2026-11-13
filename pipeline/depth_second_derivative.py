"""Second derivative of the depth image: its absolute value, and the ratio depth / |second derivative|.

The same two steps as for the RGB image (second_derivative_ratio.py), applied to the depth map:
  1. D = depth from run_depth_da3.py (<depth_dir>/<stem>.npy); --quantity inverse uses 1/D instead
  2. bilateral filter on the depth (removes the small steps the depth network leaves on surfaces
     receding from the camera, while keeping real depth jumps), then light Gaussian smoothing (--sigma)
     and the Laplacian
  3. absolute value |lap D|
  4. ratio = D / (|lap D| + eps), with eps = --eps-frac times the 99th percentile of |lap D|
The ratio is unchanged if depth is multiplied by a constant, so the unknown depth scale cancels.

Outputs in <out_dir>:
  npy/<stem>_abs_d2.npy   |second derivative of depth|, H x W
  npy/<stem>_ratio.npy    depth / |second derivative of depth|, H x W
  <stem>_panel.png        image | depth | |second derivative| (log) | ratio (log)

Usage:
  python depth_second_derivative.py <depth_dir> <image_dir> <out_dir> [--quantity depth|inverse] [--sigma 1.0] [--bilateral-iters 2] [--eps-frac 0.01]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def load_depth(path):
    d = np.load(path).astype(np.float32)
    bad = ~np.isfinite(d) | (d <= 0)
    if bad.any():
        d[bad] = np.median(d[~bad])
    return d


def bilateral(d, diameter, sigma_color, sigma_space, iters):
    """Edge preserving smoothing of depth (removes small steps, keeps real depth jumps).

    Filters log depth, so sigma_color is a relative depth difference: 0.03 means steps smaller
    than about 3% of the depth are smoothed away, while larger jumps (occlusions) are preserved.
    """
    if iters <= 0:
        return d
    x = np.log(d).astype(np.float32)
    for _ in range(iters):
        x = cv2.bilateralFilter(x, diameter, sigma_color, sigma_space)
    return np.exp(x).astype(np.float32)


def find_image(image_dir, stem):
    for ext in EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def show(x, log=False, gamma=False, color=False, decades=2.0):
    if log:  # only `decades` below the bright end, so weak undulation on flat areas stays dark
        x = np.log10(np.maximum(x, 1e-12))
        hi = np.percentile(x, 99.5)
        lo = max(np.percentile(x, 0.5), hi - decades)
    else:
        lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    y = np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
    if gamma:
        y = y ** (1 / 2.2)
    y = (y * 255 + 0.5).astype(np.uint8)
    if color:
        return cv2.applyColorMap(y, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR) if y.ndim == 2 else y


def label(img, text):
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("depth_dir")
    ap.add_argument("image_dir", help="only used to show the photo in the panel")
    ap.add_argument("out_dir")
    ap.add_argument("--quantity", choices=["depth", "inverse"], default="depth")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--bilateral-iters", type=int, default=2, help="bilateral passes on depth before derivatives; 0 to disable")
    ap.add_argument("--bilateral-d", type=int, default=9, help="bilateral filter diameter in pixels")
    ap.add_argument("--bilateral-sigma-color", type=float, default=0.03, help="relative depth difference to smooth across")
    ap.add_argument("--bilateral-sigma-space", type=float, default=5.0, help="spatial sigma in pixels")
    ap.add_argument("--decades", type=float, default=2.0, help="log display range below the strongest values")
    ap.add_argument("--eps-frac", type=float, default=0.01)
    args = ap.parse_args()

    depth_dir, image_dir, out = Path(args.depth_dir), Path(args.image_dir), Path(args.out_dir)
    (out / "npy").mkdir(parents=True, exist_ok=True)
    files = sorted(depth_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"no .npy depth maps in {depth_dir}")

    for i, path in enumerate(files, 1):
        stem = path.stem
        d = load_depth(path)
        d = bilateral(d, args.bilateral_d, args.bilateral_sigma_color, args.bilateral_sigma_space, args.bilateral_iters)
        q = 1.0 / d if args.quantity == "inverse" else d
        qs = cv2.GaussianBlur(q, (0, 0), args.sigma) if args.sigma > 0 else q
        d2 = np.abs(cv2.Laplacian(qs, cv2.CV_32F, ksize=3))
        b = 3  # one sided derivatives at the border are unreliable
        d2[:b], d2[-b:], d2[:, :b], d2[:, -b:] = 0, 0, 0, 0
        eps = args.eps_frac * float(np.percentile(d2, 99)) + 1e-12
        ratio = q / (d2 + eps)

        np.save(out / "npy" / f"{stem}_abs_d2.npy", d2.astype(np.float32))
        np.save(out / "npy" / f"{stem}_ratio.npy", ratio.astype(np.float32))

        name = "depth" if args.quantity == "depth" else "inverse depth"
        panels = [
            label(show(1.0 / d, color=True), name),
            label(show(d2, log=True, decades=args.decades), f"|second derivative of {name}|"),
            label(show(ratio, log=True, decades=args.decades), f"{name} / |second derivative|"),
        ]
        img_path = find_image(image_dir, stem)
        if img_path is not None:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            img = img[:, :, :3].astype(np.float32)
            if img.shape[:2] != d.shape:
                img = cv2.resize(img, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
            panels.insert(0, label(show(img, gamma=True), "image"))
        cv2.imwrite(str(out / f"{stem}_panel.png"), np.hstack(panels))
        print(f"[{i}/{len(files)}] {stem}: 99th pct |d2| {np.percentile(d2, 99):.3e}")

    print(f"done, results in {out}")


if __name__ == "__main__":
    main()
