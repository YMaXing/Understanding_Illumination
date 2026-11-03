"""TASK_GEOM_INTEGRATION G5: fold-placement diagnostic (script only; reads RGB; no channel change).

Two sites per accepted fold candidate of geom_v1, both along the candidate's e1 (the fit direction):
  NMS site   x0 (the rotation-rate ridge the fold channel places the fold at), offset 0 along e1;
  plane site x0 + s_gap e1, where the two fitted side planes intersect (rho_+ = rho_- along e1: the k1 fit's gap root,
             recorded as s_gap when planes_cross; rho is affine on planes, so this is the image of the 3-D crease
             line). Candidates whose planes do not cross within [-k1, k1] have no plane site (counted).
RGB edge: s_rgb = the maximum |grad Y . e1| within +-8 px of x0 (scripts/rgb_edges.py). Offsets of the RGB edge from
each site: NMS s_rgb, plane s_rgb - s_gap; signed with the G4 conventions (vertical corners: + toward the more frontal
face, |n_z| larger; contact folds and nosings: + image up) and, for the hypothesis, also signed toward the more frontal
face for every candidate. Hypothesis (convex folds): the NMS peak shifts toward the more camera-facing face (where n_hat
is most sensitive to grad rho), so the NMS offset is negative in the frontal convention and the plane offset ~ 0.
Sets: the 15 geom_offset units (audit_split/, accepted candidates whose own band reaches the unit; sub-type from the
tags, 173 by orientation) and the batch1 fold targets (scripts/fold_null_rois.py TARGETS + the depth_rois fold
polylines; accepted candidates within 3 px; sub-type by orientation and convexity: near-vertical -> vertical-corner,
near-horizontal convex -> nosing, concave -> contact).
Writes baseline_output/diag/geom/g5/fold_site.json and fold_site_samples.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from depth_rois import POLYLINES  # noqa: E402
from fold_null_rois import TARGET_TOL, TARGETS, _poly_samples  # noqa: E402
from geom_g4 import AUD  # noqa: E402
from rgb_edges import luminance, rgb_offset  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.stage_b import load_upright  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
RUN = "geom_v1"
OUT = ROOT / "baseline_output" / "diag" / "geom" / "g5"
SUBTYPES = ("vertical-corner", "contact", "nosing")


def offsets(t: pd.DataFrame, Y: np.ndarray, sub: list[str]) -> pd.DataFrame:
    x0 = t[["u0", "v0"]].to_numpy()
    e1 = t[["e1_u", "e1_v"]].to_numpy()
    s_rgb = rgb_offset(Y, x0, e1)
    s_gap = np.where(t["planes_cross"].to_numpy(), t["s_gap"].to_numpy(), np.nan)
    frontal = np.sign(np.abs(t["n_p_z"].to_numpy()) - np.abs(t["n_m_z"].to_numpy()))   # + side more frontal: +1
    up = -np.sign(e1[:, 1])
    sgn = np.where(np.isin(sub, ["contact", "nosing"]), up, frontal)
    return pd.DataFrame({"subtype": sub, "convexity": t["convexity"].to_numpy(), "s_rgb_raw": s_rgb, "s_gap_raw": s_gap,
                         "nms": s_rgb * sgn, "plane": (s_rgb - s_gap) * sgn,
                         "nms_frontal": s_rgb * frontal, "plane_frontal": (s_rgb - s_gap) * frontal,
                         "site_shift_frontal": s_gap * frontal,   # plane site relative to the NMS site
                         "rescued": t["rescued"].to_numpy() if "rescued" in t else False})


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for sub, g in df.groupby("subtype"):
        ok = g["plane"].notna()
        row = {"n_cand": int(len(g)), "n_units_or_targets": int(g["set_id"].nunique()),
               "share_planes_cross": float(ok.mean()), "share_rgb_at_limit": float((g["s_rgb_raw"].abs() >= 7.99).mean())}
        for col in ("nms", "plane", "nms_frontal", "plane_frontal", "site_shift_frontal"):
            v = g.loc[ok, col].dropna()  # the same candidates for both sites
            row[col] = {"median": float(v.median()), "P25": float(v.quantile(0.25)), "P75": float(v.quantile(0.75)),
                        "share_abs_le_1": float((v.abs() <= 1).mean())} if len(v) else None
        row["per_set_median"] = {str(k): {"nms": round(float(x["nms"].median()), 2),
                                          "plane": round(float(x["plane"].median()), 2) if x["plane"].notna().any()
                                          else None, "n": int(len(x))}
                                 for k, x in g.groupby("set_id")}
        out[sub] = row
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(AUD / "audit_labels.csv", dtype={"tags": str}).fillna({"tags": ""})
    strata = pd.read_csv(AUD / "audit_strata.csv")
    batch_of = dict(zip(strata["stem"], strata["batch"]))
    masks = np.load(AUD / "audit_masks.npz")
    rows = []
    cache = {}

    def load(batch, stem):
        if stem not in cache:
            rd = BATCHES[batch] / "stage_b" / RUN
            cfg = load_config(rd / "config_resolved.yaml")
            a, _ = load_upright(cfg, rd, stem)
            fc = pd.read_parquet(rd / f"{stem}_fold_candidates.parquet")
            cache[stem] = (a["w_net"].shape, fc[fc["accepted"]].reset_index(drop=True),
                           luminance(BATCHES[batch] / "stage_a", stem))
        return cache[stem]

    # geom_offset units
    for r in lab[lab["bucket"] == "geom_offset"].itertuples():
        shape, fc, Y = load(batch_of[r.stem], r.stem)
        m = np.zeros(shape[0] * shape[1], bool)
        m[masks[f"id{r.id:03d}"]] = True
        dist = ndimage.distance_transform_edt(~m.reshape(shape))
        dd = dist[np.clip(fc["v"], 0, shape[0] - 1), np.clip(fc["u"], 0, shape[1] - 1)]
        t = fc[dd <= fc["band"].to_numpy()]
        if not len(t):
            continue
        tags = set(r.tags.split("|"))
        tag_sub = next((x for x in SUBTYPES if x in tags), None)
        horiz = np.abs(t["e1_v"].to_numpy()) > np.abs(t["e1_u"].to_numpy())
        sub = [tag_sub or ("contact" if h else "vertical-corner") for h in horiz]
        o = offsets(t, Y, sub)
        o.insert(0, "set", "geom_offset")
        o.insert(1, "set_id", r.id)
        o.insert(2, "stem", r.stem)
        rows.append(o)
    # batch1 fold targets
    targets = {k: (s, kind, poly) for k, (s, kind, poly) in TARGETS.items()}
    targets |= {k: (s, "fold", poly) for k, (s, cls, poly) in POLYLINES.items() if cls == "fold"}
    for name, (s, kind, poly) in targets.items():
        stem = f"taiwei_cui_{s}"
        shape, fc, Y = load("batch1", stem)
        S = _poly_samples(poly)
        x = fc[["u0", "v0"]].to_numpy()
        if not len(x):
            continue
        d = np.sqrt(((x[:, None, :] - S[None]) ** 2).sum(-1)).min(1)
        t = fc[d <= TARGET_TOL]
        if not len(t):
            continue
        horiz = np.abs(t["e1_v"].to_numpy()) > np.abs(t["e1_u"].to_numpy())
        conv = t["convexity"].to_numpy()
        sub = ["vertical-corner" if not h else ("nosing" if c > 0 else "contact") for h, c in zip(horiz, conv)]
        o = offsets(t, Y, sub)
        o.insert(0, "set", "fold_target")
        o.insert(1, "set_id", name)
        o.insert(2, "stem", stem)
        rows.append(o)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT / "fold_site_samples.csv", index=False)
    rep = {st: summarize(df[df["set"] == st]) for st in ("geom_offset", "fold_target")}
    conv = df[df["convexity"] > 0]
    rep["convex_all"] = {c: {"median": float(conv.loc[conv["plane"].notna(), c].median()), "n":
                             int(conv["plane"].notna().sum())} for c in ("nms_frontal", "plane_frontal",
                                                                          "site_shift_frontal")}
    (OUT / "fold_site.json").write_text(json.dumps(rep, indent=1))
    for st in ("geom_offset", "fold_target"):
        print(f"\n== {st}")
        for sub, v in rep[st].items():
            f = lambda d: "-" if d is None else f"{d['median']:+.2f} [{d['P25']:+.2f}, {d['P75']:+.2f}]"  # noqa: E731
            print(f"{sub:16s} n {v['n_cand']:4d} sets {v['n_units_or_targets']:2d} cross {v['share_planes_cross']:.2f} "
                  f"lim {v['share_rgb_at_limit']:.2f} | NMS {f(v['nms'])} | plane {f(v['plane'])} | "
                  f"frontal: NMS {f(v['nms_frontal'])} plane {f(v['plane_frontal'])} site shift "
                  f"{f(v['site_shift_frontal'])}")
            print("   per set:", v["per_set_median"])
    print("\nconvex, all sets (frontal convention):", rep["convex_all"])


if __name__ == "__main__":
    main()
