"""TASK_GEOM_INTEGRATION G1: per-class evaluation of any list of Stage B runs on batch1's pinned ROIs.

ROIs (upright network px): scripts/depth_rois.py POLYLINES (with RELABELLED), BOXES (the trimmed 083 bollard interior;
the legacy box reported separately) and the 070 sky band (tri_veto4 sky mask); scripts/fold_null_rois.py TARGETS
(narrow-face edges -> class fold, crease targets -> class crease; RELABELLED names skipped) and the texture boxes
(rois(): 070 floor, 079 grass, 079 grass sunlit).
Classes: step, limb, fold, crease, texture.
Metrics on the run's combined w_net (network resolution, upright); for geometry runs also on each channel's w:
  share of w < 0.5, median w, and for polylines the coverage = the fraction of polyline samples (every 0.5 px) whose
  3x3 neighbourhood (within 1 px of the nearest pixel) holds a pixel with w < 0.5.
Per class: pixel/sample-weighted share, the median over all samples, the sample-weighted coverage of the polylines.

Run from geometry/dedicated:
  env -u PYTHONPATH uv run python scripts/geom_eval.py [--runs tri_veto4 tri_veto4_fold_v2 geom_v0] [--out DIR]
Writes <out>/eval.json (default baseline_output/diag/geom/g1) and prints the per-class table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from depth_rois import BOXES, POLYLINES, RELABELLED, box_mask, sky_band_070  # noqa: E402
from fold_null_rois import TARGETS, _poly_samples, rois  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.stage_b import load_upright  # noqa: E402

SB = ROOT / "baseline_output" / "stage_b"
CLASSES = ("step", "limb", "fold", "crease", "texture")
DEFAULT_RUNS = ("tri_veto4", "tri_veto4_fold_v2", "geom_v0")


def roi_list() -> list[dict]:
    """[{name, stem, class, kind: polyline|area, poly|box|mask_fn}] for batch1."""
    out = []
    for name, (s, cls, poly) in POLYLINES.items():
        out.append({"name": name, "stem": s, "class": cls, "kind": "polyline", "poly": poly})
    for name, (s, kind, poly) in TARGETS.items():
        if name in RELABELLED:
            continue
        out.append({"name": name, "stem": s, "class": "crease" if kind == "crease" else "fold", "kind": "polyline",
                    "poly": poly})
    for s in ("070", "079"):
        for name in rois(f"taiwei_cui_{s}", (1, 1)):
            out.append({"name": name, "stem": s, "class": "texture", "kind": "area", "fold_null_roi": name})
    for name, (s, cls, box) in BOXES.items():
        out.append({"name": name, "stem": s, "class": cls, "kind": "area", "box": box,
                    "legacy": "legacy" in name})
    out.append({"name": "070 sky boundary", "stem": "070", "class": "step", "kind": "area", "sky_band": True})
    return out


def area_mask(roi: dict, shape) -> np.ndarray:
    if "fold_null_roi" in roi:
        return rois(f"taiwei_cui_{roi['stem']}", shape)[roi["fold_null_roi"]]
    if "box" in roi:
        return box_mask(shape, roi["box"])
    return sky_band_070()


def roi_metrics(w: np.ndarray, roi: dict) -> dict:
    if roi["kind"] == "polyline":
        S = _poly_samples(roi["poly"])
        H, W = w.shape
        iu = np.clip(np.rint(S[:, 0]).astype(int), 0, W - 1)
        iv = np.clip(np.rint(S[:, 1]).astype(int), 0, H - 1)
        vals = w[iv, iu]
        near = ndimage.minimum_filter(w, size=3, mode="nearest")[iv, iu] < 0.5
        return {"n": int(len(S)), "share_w<0.5": float((vals < 0.5).mean()), "median_w": float(np.median(vals)),
                "coverage": float(near.mean()), "_vals": vals, "_cov": near}
    m = area_mask(roi, w.shape)
    vals = w[m]
    return {"n": int(vals.size), "share_w<0.5": float((vals < 0.5).mean()), "median_w": float(np.median(vals)),
            "coverage": None, "_vals": vals, "_cov": None}


def evaluate(runs: list[str]) -> dict:
    res = {"rois": [], "classes": []}
    cache: dict[tuple[str, str], tuple[dict, dict]] = {}
    for run in runs:
        cfg = load_config(SB / run / "config_resolved.yaml")
        pooled: dict[tuple[str, str], dict] = {}
        for roi in roi_list():
            key = (run, roi["stem"])
            if key not in cache:
                cache[key] = load_upright(cfg, SB / run, f"taiwei_cui_{roi['stem']}")
            a, _ = cache[key]
            maps = {"combined": a["w_net"].astype(np.float64)}
            for ch in ("depth", "fold", "laplacian"):
                if f"{ch}_w_net" in a:
                    maps[ch] = a[f"{ch}_w_net"].astype(np.float64)
            for which, w in maps.items():
                m = roi_metrics(w, roi)
                res["rois"].append({"run": run, "w": which, "roi": roi["name"], "stem": roi["stem"],
                                    "class": roi["class"], "legacy": bool(roi.get("legacy", False)),
                                    **{k: v for k, v in m.items() if not k.startswith("_")}})
                if roi.get("legacy"):
                    continue
                p = pooled.setdefault((which, roi["class"]), {"vals": [], "cov": []})
                p["vals"].append(m["_vals"])
                if m["_cov"] is not None:
                    p["cov"].append(m["_cov"])
        for (which, cls), p in pooled.items():
            v = np.concatenate(p["vals"])
            c = np.concatenate(p["cov"]) if p["cov"] else None
            res["classes"].append({"run": run, "w": which, "class": cls, "n": int(v.size),
                                   "share_w<0.5": float((v < 0.5).mean()), "median_w": float(np.median(v)),
                                   "coverage": float(c.mean()) if c is not None else None})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--out", default=str(ROOT / "baseline_output" / "diag" / "geom" / "g1"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = evaluate(args.runs)
    res["runs"] = args.runs
    (out / "eval.json").write_text(json.dumps(res, indent=1))
    cls = pd.DataFrame(res["classes"])
    comb = cls[cls["w"] == "combined"].copy()
    comb["class"] = pd.Categorical(comb["class"], CLASSES, ordered=True)
    tab = comb.pivot_table(index="class", columns="run", values=["share_w<0.5", "median_w", "coverage"],
                           observed=False)[[(m, r) for m in ("share_w<0.5", "median_w", "coverage") for r in args.runs]]
    pd.set_option("display.width", 250)
    print("combined w, per class (share w<0.5 | median w | polyline coverage):\n" + tab.round(3).to_string())
    rois_df = pd.DataFrame(res["rois"])
    r = rois_df[rois_df["w"] == "combined"].pivot_table(index=["class", "roi"], columns="run",
                                                         values=["share_w<0.5", "coverage"])
    print("\nper ROI (combined w):\n" + r.round(3).to_string())
    print(f"\n-> {out / 'eval.json'}")


if __name__ == "__main__":
    main()
