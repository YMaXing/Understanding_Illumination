"""Baseline material map: cluster pixels by log chromaticity color.

In log chromaticity the change caused by shadows is removed, so one material has roughly one color
in sun and in shade. Clustering those colors gives a material map: the log chromaticity baseline.

Steps per image:
  1. features = log chromaticity (projected log RGB from run_isd_vit.py), lightly smoothed
  2. pixels too dark to have a reliable color are marked invalid (label 0)
  3. k means on the valid pixels (cv2.kmeans, on a random sample, then every pixel to the nearest center)
  4. clusters  = material groups (the same material in two places shares a cluster)
     regions   = connected pieces of each cluster; pieces smaller than --min-region are merged
                 into the most common neighboring region

Outputs in <out_dir>:
  clusters/<stem>.npy   int32, 0 invalid, 1..k material group
  regions/<stem>.npy    int32, 0 invalid, 1..n connected material region (used by the losses)
  viz/<stem>.png        image | cluster colors | region boundaries
  scores.csv            if --annotations is given: pairwise scores against the point annotations

Scoring (sparse point annotations, image,x,y,region_id,label):
  same_recall     fraction of same region point pairs that land in the same cluster  (higher is better)
  diff_separation fraction of different region point pairs that land in different clusters (higher is better)
  cross_shadow    same_recall restricted to pairs whose image brightness differs by more than 2x,
                  i.e. pairs that cross a shadow (the hard case)

Usage:
  python cluster_materials.py <image_dir> <isd_dir> <out_dir> [--k 6] [--annotations annotations_dir]
"""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


def tone_map(img16):
    x = img16.astype(np.float32)
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    return (np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def kmeans_labels(feat, valid, k, sample=60000, seed=0):
    rng = np.random.default_rng(seed)
    pts = feat[valid]
    take = pts if len(pts) <= sample else pts[rng.choice(len(pts), sample, replace=False)]
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _, _, centers = cv2.kmeans(take.astype(np.float32), k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    d = ((feat[:, :, None, :] - centers[None, None]) ** 2).sum(-1)
    lab = d.argmin(-1).astype(np.int32) + 1
    lab[~valid] = 0
    return lab


def connected_regions(clusters, min_region):
    """Split clusters into connected regions, then merge small regions into a neighbor."""
    regions = np.zeros(clusters.shape, np.int32)
    nxt = 1
    for c in np.unique(clusters):
        if c == 0:
            continue
        n, cc = cv2.connectedComponents((clusters == c).astype(np.uint8), connectivity=4)
        m = cc > 0
        regions[m] = cc[m] + nxt - 1
        nxt += n - 1
    ids, counts = np.unique(regions, return_counts=True)
    kernel = np.ones((3, 3), np.uint8)
    for rid, cnt in sorted(zip(ids, counts), key=lambda t: t[1]):
        if rid == 0 or cnt >= min_region:
            continue
        m = (regions == rid).astype(np.uint8)
        ring = (cv2.dilate(m, kernel) > 0) & (m == 0)
        nb = regions[ring]
        nb = nb[(nb != 0) & (nb != rid)]
        if len(nb):
            regions[m > 0] = np.bincount(nb).argmax()
    # renumber 1..n
    uniq = np.unique(regions)
    remap = np.zeros(uniq.max() + 1, np.int32)
    remap[uniq[uniq > 0]] = np.arange(1, (uniq > 0).sum() + 1)
    return remap[regions]


def score(clusters, lum, csv_path):
    pts = []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            x, y = int(row["x"]), int(row["y"])
            if 0 <= y < clusters.shape[0] and 0 <= x < clusters.shape[1] and row["label"] != "specular":
                pts.append((int(row["region_id"]), clusters[y, x], lum[y, x]))
    if len(pts) < 2:
        return None
    same = diff = same_ok = diff_ok = cross = cross_ok = 0
    for (ra, ca, la), (rb, cb, lb) in combinations(pts, 2):
        if ra == rb:
            same += 1
            same_ok += ca == cb and ca != 0
            if max(la, lb) > 2 * max(min(la, lb), 1e-6):
                cross += 1
                cross_ok += ca == cb and ca != 0
        else:
            diff += 1
            diff_ok += ca != cb
    f = lambda a, b: round(a / b, 4) if b else ""
    return {"points": len(pts), "same_recall": f(same_ok, same), "diff_separation": f(diff_ok, diff),
            "cross_shadow": f(cross_ok, cross), "cross_pairs": cross}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("isd_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--k", type=int, default=6, help="number of material groups")
    ap.add_argument("--smooth", type=float, default=2.0, help="Gaussian smoothing of the chromaticity features")
    ap.add_argument("--dark", type=float, default=200.0, help="pixels whose mean 16 bit value is below this are invalid")
    ap.add_argument("--min-region", type=int, default=400, help="merge connected regions smaller than this")
    ap.add_argument("--annotations", default=None, help="folder of point annotation CSVs to score against")
    args = ap.parse_args()

    image_dir, isd_dir, out = Path(args.image_dir), Path(args.isd_dir), Path(args.out_dir)
    for sub in ("clusters", "regions", "viz"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in (isd_dir / "logchrom_npy").glob("*.npy"))
    palette = (np.random.default_rng(1).integers(40, 255, (4096, 3))).astype(np.uint8)
    palette[0] = 0
    rows = []

    for i, stem in enumerate(stems, 1):
        img_path = next((image_dir / f"{stem}{e}" for e in (".tif", ".tiff") if (image_dir / f"{stem}{e}").exists()), None)
        if img_path is None:
            continue
        raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        rgb16 = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
        lum = rgb16.astype(np.float32).mean(axis=2)

        feat = np.load(isd_dir / "logchrom_npy" / f"{stem}.npy").astype(np.float32)
        feat = feat - feat.reshape(-1, 3).mean(axis=0)  # the projected values only differ within the plane
        if args.smooth > 0:
            feat = cv2.GaussianBlur(feat, (0, 0), args.smooth)
        valid = (lum > args.dark) & (rgb16.max(axis=2) < 65000)

        clusters = kmeans_labels(feat, valid, args.k)
        regions = connected_regions(clusters, args.min_region)
        np.save(out / "clusters" / f"{stem}.npy", clusters)
        np.save(out / "regions" / f"{stem}.npy", regions)

        base = cv2.cvtColor(tone_map(rgb16), cv2.COLOR_RGB2BGR)
        col = palette[clusters]
        edges = cv2.morphologyEx(regions.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3))) > 0
        bnd = base.copy()
        bnd[edges] = (0, 0, 255)
        cv2.imwrite(str(out / "viz" / f"{stem}.png"), np.hstack([base, col, bnd]))

        row = {"image": stem, "regions": int(regions.max())}
        if args.annotations:
            p = Path(args.annotations) / f"{stem}.csv"
            if p.exists():
                s = score(clusters, lum, p)
                if s:
                    row.update(s)
        rows.append(row)
        extra = f"  same {row.get('same_recall')}  diff {row.get('diff_separation')}  cross shadow {row.get('cross_shadow')}" if "same_recall" in row else ""
        print(f"[{i}/{len(stems)}] {stem}: {row['regions']} regions{extra}")

    keys = ["image", "regions", "points", "same_recall", "diff_separation", "cross_shadow", "cross_pairs"]
    with open(out / "scores.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})
    print(f"done, results in {out}")


if __name__ == "__main__":
    main()
