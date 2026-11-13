"""Depth edges, depth boundaries, surface orientation and orientation changes.

Reads the Depth Anything 3 outputs (<depth_dir>/<stem>.npy, float relative depth at
the original image size) and, for every image, writes:

  mag/<stem>.npy           relative depth edge magnitude |grad d| / d (scale invariant)
  mag/<stem>.png           preview of the magnitude
  normals/<stem>.npy       unit surface normals, H x W x 3, camera frame (x right, y down, z forward)
  normals/<stem>.png       normals as RGB (red = facing right, green = facing up, blue = facing camera)
  orient/<stem>.npy        orientation change in degrees between normals `step` pixels apart
  orient/<stem>.png        preview of the orientation change
  sweep/<stem>.png         binary maps for every threshold, to compare and pick one
  overlay/<stem>.png       image with depth boundaries (red) and orientation changes (cyan)

Depth boundaries = pixels whose relative edge magnitude exceeds a threshold (occlusions, e.g.
a post in front of a wall). Orientation changes = pixels where the surface normal turns by more
than an angle threshold while depth stays continuous (creases, e.g. where two walls meet).

Usage:
  python depth_geometry.py <depth_npy_dir> <image_dir> <out_dir>
      [--depth-thresh 0.02 0.05 0.1] [--angle-thresh 10 20 30] [--fov 60] [--sigma 1.0] [--step 3]
"""
import argparse
import csv
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


def smooth(d, sigma):
    return cv2.GaussianBlur(d, (0, 0), sigma) if sigma > 0 else d


def edge_magnitude(d, sigma):
    """Relative gradient magnitude |grad d| / d, per pixel. Unchanged if depth is scaled."""
    ds = smooth(d, sigma)
    gx = cv2.Sobel(ds, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(ds, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(gx, gy) / np.maximum(ds, 1e-6)


def surface_normals(d, fov_deg, sigma):
    """Back project with a pinhole camera of the given horizontal field of view, then cross the tangents."""
    h, w = d.shape
    f = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = smooth(d, sigma)
    pts = np.dstack([(u - (w - 1) / 2.0) / f * z, (v - (h - 1) / 2.0) / f * z, z])
    du = np.gradient(pts, axis=1)
    dv = np.gradient(pts, axis=0)
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-12)
    flip = np.sum(n * pts, axis=2) > 0  # make every normal face the camera
    n[flip] *= -1
    return n.astype(np.float32)


def orientation_change(n, step):
    """Largest angle (degrees) between a pixel's normal and the normals `step` pixels right or below."""
    h, w, _ = n.shape
    out = np.zeros((h, w), np.float32)
    ax = np.degrees(np.arccos(np.clip(np.sum(n[:, step:] * n[:, :-step], axis=2), -1, 1)))
    ay = np.degrees(np.arccos(np.clip(np.sum(n[step:] * n[:-step], axis=2), -1, 1)))
    out[:, :-step] = np.maximum(out[:, :-step], ax)
    out[:, step:] = np.maximum(out[:, step:], ax)
    out[:-step] = np.maximum(out[:-step], ay)
    out[step:] = np.maximum(out[step:], ay)
    b = step + 2  # one sided derivatives at the image border are unreliable
    out[:b], out[-b:], out[:, :b], out[:, -b:] = 0, 0, 0, 0
    return out


def tone_map(img):
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    return (np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def heat(x, hi):
    v = (np.clip(x / max(hi, 1e-6), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(v, cv2.COLORMAP_INFERNO)


def label(img, text):
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def find_image(image_dir, stem):
    for ext in EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("depth_dir")
    ap.add_argument("image_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--depth-thresh", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    ap.add_argument("--angle-thresh", type=float, nargs="+", default=[10, 20, 30])
    ap.add_argument("--fov", type=float, default=60.0, help="assumed horizontal field of view in degrees")
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian smoothing of depth before derivatives")
    ap.add_argument("--bilateral-iters", type=int, default=2, help="bilateral passes on depth before derivatives; 0 to disable")
    ap.add_argument("--bilateral-d", type=int, default=9, help="bilateral filter diameter in pixels")
    ap.add_argument("--bilateral-sigma-color", type=float, default=0.03, help="relative depth difference to smooth across")
    ap.add_argument("--bilateral-sigma-space", type=float, default=5.0, help="spatial sigma in pixels")
    ap.add_argument("--step", type=int, default=3, help="pixel spacing for orientation change")
    args = ap.parse_args()

    depth_dir, image_dir, out = Path(args.depth_dir), Path(args.image_dir), Path(args.out_dir)
    for sub in ("mag", "normals", "orient", "sweep", "overlay"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    files = sorted(depth_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"no .npy depth maps in {depth_dir}")
    dt_mid = sorted(args.depth_thresh)[len(args.depth_thresh) // 2]
    at_mid = sorted(args.angle_thresh)[len(args.angle_thresh) // 2]

    stats = []
    for i, path in enumerate(files, 1):
        stem = path.stem
        d = load_depth(path)
        d = bilateral(d, args.bilateral_d, args.bilateral_sigma_color, args.bilateral_sigma_space, args.bilateral_iters)
        mag = edge_magnitude(d, args.sigma)
        n = surface_normals(d, args.fov, args.sigma)
        oc = orientation_change(n, args.step)

        np.save(out / "mag" / f"{stem}.npy", mag)
        np.save(out / "normals" / f"{stem}.npy", n.astype(np.float16))
        np.save(out / "orient" / f"{stem}.npy", oc)
        cv2.imwrite(str(out / "mag" / f"{stem}.png"), heat(mag, max(args.depth_thresh) * 2))
        rgb = ((n * np.array([1, -1, -1], np.float32) + 1) / 2 * 255).astype(np.uint8)
        cv2.imwrite(str(out / "normals" / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "orient" / f"{stem}.png"), heat(oc, max(args.angle_thresh) * 1.5))

        # depth boundaries for each threshold, and orientation changes away from those boundaries
        row = {"image": stem}
        tiles = []
        r = args.step + int(np.ceil(3 * args.sigma)) + 1  # normals are unreliable this close to an occlusion
        near_boundary = cv2.dilate((mag > min(args.depth_thresh)).astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0
        for t in args.depth_thresh:
            b = mag > t
            row[f"depth_edge_frac@{t}"] = round(float(b.mean()), 5)
            tiles.append(label(cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), f"depth > {t}"))
        for t in args.angle_thresh:
            c = (oc > t) & ~near_boundary
            row[f"orient_change_frac@{t}"] = round(float(c.mean()), 5)
            tiles.append(label(cv2.cvtColor((c * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), f"angle > {t} deg"))
        row["mag_p99"] = round(float(np.percentile(mag, 99)), 5)
        stats.append(row)
        scale = 300.0 / d.shape[0]
        tiles = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in tiles]
        cv2.imwrite(str(out / "sweep" / f"{stem}.png"), np.hstack(tiles))

        img_path = find_image(image_dir, stem)
        if img_path is not None:
            raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            base = tone_map(raw[:, :, :3])
            if base.shape[:2] != d.shape:
                base = cv2.resize(base, (d.shape[1], d.shape[0]))
            ov = (base * 0.55).astype(np.uint8)
            ov[(oc > at_mid) & ~near_boundary] = (255, 255, 0)
            ov[mag > dt_mid] = (0, 0, 255)
            cv2.imwrite(str(out / "overlay" / f"{stem}.png"), label(ov, f"red: depth > {dt_mid}   cyan: angle > {at_mid} deg"))
        print(f"[{i}/{len(files)}] {stem}: 99th pct edge magnitude {row['mag_p99']}")

    with open(out / "stats.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(stats[0].keys()))
        wr.writeheader()
        wr.writerows(stats)
    print(f"done, results in {out} (summary in stats.csv)")


if __name__ == "__main__":
    main()
