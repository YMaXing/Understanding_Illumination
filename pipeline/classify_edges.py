"""Label every image edge as a shadow edge, a material edge, or a geometry edge.

Inputs per image stem:
  <image_dir>/<stem>.tif                       16 bit linear image
  <isd_dir>/logchrom_npy/<stem>.npy            log chromaticity from run_isd_vit.py
  <geometry_dir>/mag/<stem>.npy                relative depth edge magnitude from depth_geometry.py
  <geometry_dir>/orient/<stem>.npy             orientation change (degrees) from depth_geometry.py

Rule, per pixel:
  1. image edge: scale normalized gradient of log RGB above --edge-thresh; fine scale first,
     coarser scales only add edges away from finer ones (catches soft shadow edges)
  2. geometry edge: an image edge on a depth boundary or an orientation change
  3. otherwise compare the edge in log chromaticity with the edge in the image:
       ratio = |grad logchrom| / |grad log RGB|
     log chromaticity removes the change along the ISD, so for a shadow the ratio is small.
       ratio < --ratio-thresh  -> shadow edge
       ratio >= --ratio-thresh -> material edge
     (ratio is the sine of the angle between the color change and the ISD; 0.10 is about 6 degrees)

Outputs in <out_dir>:
  labels/<stem>.npy      uint8: 0 no edge, 1 shadow, 2 material, 3 geometry
  ratio/<stem>.npy       the ratio, float32 (0 where there is no edge)
  overlay/<stem>.png     image with shadow edges yellow, material edges magenta, geometry edges green
  stats.csv              fraction of pixels with each label

Usage:
  python classify_edges.py <image_dir> <isd_dir> <geometry_dir> <out_dir>
      [--edge-thresh 0.25] [--ratio-thresh 0.10] [--depth-thresh 0.05] [--angle-thresh 30] [--scales 1 4 12]
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

COLORS = {1: (0, 255, 255), 2: (255, 0, 255), 3: (0, 255, 0)}  # BGR: yellow, magenta, green


def grad_mag(x, sigma):
    """Gradient magnitude of a multichannel image, combined over channels."""
    if sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), sigma)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.sqrt((gx ** 2 + gy ** 2).sum(axis=2))


def tone_map(img16):
    x = img16.astype(np.float32)
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    return (np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("isd_dir")
    ap.add_argument("geometry_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--edge-thresh", type=float, default=0.25, help="log RGB gradient that counts as an edge")
    ap.add_argument("--ratio-thresh", type=float, default=0.10, help="below this an edge is a shadow edge")
    ap.add_argument("--depth-thresh", type=float, default=0.05)
    ap.add_argument("--angle-thresh", type=float, default=30.0)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 4.0, 12.0], help="smoothing scales in pixels; larger finds soft edges")
    args = ap.parse_args()

    image_dir, isd_dir, geo_dir, out = Path(args.image_dir), Path(args.isd_dir), Path(args.geometry_dir), Path(args.out_dir)
    for sub in ("labels", "ratio", "overlay"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in (isd_dir / "logchrom_npy").glob("*.npy"))
    if not stems:
        raise SystemExit(f"no log chromaticity maps in {isd_dir / 'logchrom_npy'}")

    rows = []
    for i, stem in enumerate(stems, 1):
        img_path = next((image_dir / f"{stem}{e}" for e in (".tif", ".tiff") if (image_dir / f"{stem}{e}").exists()), None)
        if img_path is None:
            print(f"[{i}/{len(stems)}] {stem}: image not found, skipping")
            continue
        raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        rgb16 = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
        log_rgb = np.log(np.maximum(rgb16.astype(np.float32), 1.0))  # natural log, as in log_chroma_utils
        logchrom = np.load(isd_dir / "logchrom_npy" / f"{stem}.npy").astype(np.float32)

        # scale normalized gradients (s * |grad|): a step of height h scores about h / 2.5 at any scale.
        # The finest scale has priority; a coarser scale only adds edges away from finer ones, so soft
        # shadow edges (wide penumbra) are found without blurring and relabeling the sharp edges.
        scales = sorted(args.scales)
        g_img = np.zeros(log_rgb.shape[:2], np.float32)
        g_chr = np.zeros_like(g_img)
        found = np.zeros(g_img.shape, bool)
        for s in scales:
            gi = s * grad_mag(log_rgb, s)
            gc = s * grad_mag(logchrom, s)
            r = int(round(2 * s))  # a blur of scale s spreads an edge about 2 to 3 s pixels
            near = cv2.dilate(found.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
            new = (gi > args.edge_thresh) & ~near
            g_img[new], g_chr[new] = gi[new], gc[new]
            found |= new
        edge = g_img > args.edge_thresh
        ratio = np.where(edge, g_chr / np.maximum(g_img, 1e-6), 0).astype(np.float32)

        geom = np.zeros(edge.shape, bool)
        mag_p, ori_p = geo_dir / "mag" / f"{stem}.npy", geo_dir / "orient" / f"{stem}.npy"
        if mag_p.exists() and ori_p.exists():
            g = (np.load(mag_p) > args.depth_thresh) | (np.load(ori_p) > args.angle_thresh)
            geom = cv2.dilate(g.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # allow small misalignment
        else:
            print(f"    {stem}: no depth geometry found, geometry edges skipped")

        labels = np.zeros(edge.shape, np.uint8)
        labels[edge & (ratio < args.ratio_thresh)] = 1
        labels[edge & (ratio >= args.ratio_thresh)] = 2
        labels[edge & geom] = 3

        np.save(out / "labels" / f"{stem}.npy", labels)
        np.save(out / "ratio" / f"{stem}.npy", ratio)
        ov = (cv2.cvtColor(tone_map(rgb16), cv2.COLOR_RGB2BGR) * 0.5).astype(np.uint8)
        for k, c in COLORS.items():
            ov[labels == k] = c
        cv2.imwrite(str(out / "overlay" / f"{stem}.png"), ov)

        n = labels.size
        row = {"image": stem, "shadow": round((labels == 1).sum() / n, 5),
               "material": round((labels == 2).sum() / n, 5), "geometry": round((labels == 3).sum() / n, 5)}
        rows.append(row)
        print(f"[{i}/{len(stems)}] {stem}: shadow {row['shadow']:.4f}  material {row['material']:.4f}  geometry {row['geometry']:.4f}")

    with open(out / "stats.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "shadow", "material", "geometry"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"done, results in {out}")


if __name__ == "__main__":
    main()
