"""TASK_GEOM_V2 V2 / V2b: fold re-siting report, nms (geom_v1) vs plane_intersection (geom_v1_site) vs hessian_peak
(geom_v1_hessian), side by side. Diagnostic; reads RGB.

1. Signed RGB-edge offsets along e1, NMS site (x0) vs re-sited (x0 + t e1, t = site_t_applied of each mode's run),
   conventions of scripts/geom_g5_fold_site.py (vertical corners + toward the more frontal face; contact folds and
   nosings + image up; and the frontal-face convention for every candidate), per sub-type, for:
   the 15 geom_offset units (sub-type from the tags), the batch1 fold targets (sub-type by orientation and convexity),
   and all accepted convex fold candidates of the 20 images (frontal convention). Also |offset| medians and the share
   at the +-8 px search limit.
2. Validity fraction (accepted candidates with a valid site), per batch, per set and per sub-type.
3. Fold-target RGB-edge coverage (combined and core w), split into targets detected by the fold channel (accepted
   fold candidates within 3 px on >= 50 % of the polyline, fold_null_rois.target_coverage) and not detected.
4. Closure of geom_gain / geom_offset units and texture ROIs (scripts/geom_score.py), combined and core.
Rule (propose, don't switch), per mode: propose it if the convex-candidate median |offset| drops by >= 0.5 px, fold-target
coverage drops by <= 0.01, and no geom_gain unit loses closure.
Writes baseline_output/diag/geom/v2/site.json.
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

from fold_null_rois import TARGET_TOL, _poly_samples, target_coverage  # noqa: E402
from geom_g2_coverage import covered  # noqa: E402
from geom_g4 import AUD  # noqa: E402
from geom_score import OUT, Run, fold_target_points, fold_targets, score  # noqa: E402
from rgb_edges import luminance, rgb_offset  # noqa: E402
from illum_baseline.stage_b import load_upright  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
V1 = "geom_v1"
MODES = {"plane_intersection": "geom_v1_site", "hessian_peak": "geom_v1_hessian"}
SUBTYPES = ("vertical-corner", "contact", "nosing")


def offsets(t: pd.DataFrame, Y: np.ndarray, sub) -> pd.DataFrame:
    x0 = t[["u0", "v0"]].to_numpy()
    e1 = t[["e1_u", "e1_v"]].to_numpy()
    s_rgb = rgb_offset(Y, x0, e1)
    ta = np.nan_to_num(t["site_t_applied"].to_numpy())
    frontal = np.sign(np.abs(t["n_p_z"].to_numpy()) - np.abs(t["n_m_z"].to_numpy()))
    up = -np.sign(e1[:, 1])
    sub = np.asarray(sub)
    sgn = np.where(np.isin(sub, ["contact", "nosing"]), up, frontal)
    return pd.DataFrame({"subtype": sub, "convexity": t["convexity"].to_numpy(), "valid": t["site_valid"].to_numpy(),
                         "s_rgb_raw": s_rgb, "t": ta, "nms": s_rgb * sgn, "resited": (s_rgb - ta) * sgn,
                         "nms_frontal": s_rgb * frontal, "resited_frontal": (s_rgb - ta) * frontal})


def summarize(df: pd.DataFrame, cols=("nms", "resited")) -> dict:
    out = {"n": int(len(df)), "valid_share": float(df["valid"].mean()) if len(df) else float("nan"),
           "share_at_limit": float((df["s_rgb_raw"].abs() >= 7.99).mean()) if len(df) else float("nan")}
    for c in cols:
        v = df[c].dropna()
        out[c] = {"median": float(v.median()), "P25": float(v.quantile(0.25)), "P75": float(v.quantile(0.75)),
                  "median_abs": float(v.abs().median())} if len(v) else None
    return out


def collect(run_name: str, lab: pd.DataFrame, masks, Ys: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    run = Run(run_name)
    tables = {}

    def load(stem):
        if stem not in tables:
            b = run.batch_of(stem)
            fc = pd.read_parquet(BATCHES[b] / "stage_b" / run_name / f"{stem}_fold_candidates.parquet")
            tables[stem] = fc[fc["accepted"]].reset_index(drop=True)
            if stem not in Ys:
                Ys[stem] = luminance(BATCHES[b] / "stage_a", stem)
        return tables[stem], Ys[stem]

    rows = []
    for r in lab[lab["bucket"] == "geom_offset"].itertuples():
        fc, Y = load(r.stem)
        H, W = run.maps(r.stem)["combined_net"].shape
        m = np.zeros(H * W, bool)
        m[masks[f"id{r.id:03d}"]] = True
        dist = ndimage.distance_transform_edt(~m.reshape(H, W))
        t = fc[dist[np.clip(fc["v"], 0, H - 1), np.clip(fc["u"], 0, W - 1)] <= fc["band"].to_numpy()]
        if not len(t):
            continue
        tags = set(r.tags.split("|"))
        tag_sub = next((x for x in SUBTYPES if x in tags), None)
        horiz = np.abs(t["e1_v"].to_numpy()) > np.abs(t["e1_u"].to_numpy())
        o = offsets(t, Y, [tag_sub or ("contact" if h else "vertical-corner") for h in horiz])
        rows.append(o.assign(set="geom_offset", set_id=str(r.id)))
    det = {}
    for tg in fold_targets():
        fc, Y = load(tg["stem"])
        S = _poly_samples(tg["poly"])
        x = fc[["u0", "v0"]].to_numpy()
        det[tg["name"]] = bool(target_coverage(fc, tg["poly"]) >= 0.5) if len(fc) else False
        if not len(x):
            continue
        d = np.sqrt(((x[:, None, :] - S[None]) ** 2).sum(-1)).min(1)
        t = fc[d <= TARGET_TOL]
        if not len(t):
            continue
        horiz = np.abs(t["e1_v"].to_numpy()) > np.abs(t["e1_u"].to_numpy())
        sub = ["vertical-corner" if not h else ("nosing" if c > 0 else "contact")
               for h, c in zip(horiz, t["convexity"].to_numpy())]
        rows.append(offsets(t, Y, sub).assign(set="fold_target", set_id=tg["name"]))
    allc = []
    for batch, stems in run.stems.items():
        for stem in stems:
            fc, Y = load(stem)
            t = fc[fc["convexity"] > 0]
            allc.append(offsets(t, Y, ["all"] * len(t)).assign(set="all_convex", set_id=stem, batch=batch))
    validity = {b: {"accepted": int(sum(len(tables[s]) for s in st)),
                    "valid_share": float(pd.concat([tables[s] for s in st])["site_valid"].mean())}
                for b, st in run.stems.items()}
    return pd.concat(rows, ignore_index=True), pd.concat(allc, ignore_index=True), validity, det


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(AUD / "audit_labels.csv", dtype={"tags": str}).fillna({"tags": ""})
    masks = np.load(AUD / "audit_masks.npz")
    Ys: dict = {}
    rep = {"modes": {}}
    samples, det = [], None
    for mode, run_name in MODES.items():
        O, C, validity, det = collect(run_name, lab, masks, Ys)
        samples.append(O.assign(mode=mode))
        m = {"validity": validity, "offsets": {}}
        for st in ("geom_offset", "fold_target"):
            m["offsets"][st] = {sub: summarize(g) | {"per_set": {k: {"nms": round(float(x["nms"].median()), 2),
                                                                  "resited": round(float(x["resited"].median()), 2),
                                                                  "valid": round(float(x["valid"].mean()), 2)}
                                                              for k, x in g.groupby("set_id")}}
                                for sub, g in O[O["set"] == st].groupby("subtype")}
        m["offsets"]["all_convex"] = summarize(C, ("nms_frontal", "resited_frontal"))
        m["offsets"]["all_convex_valid_only"] = summarize(C[C["valid"]], ("nms_frontal", "resited_frontal"))
        m["offsets"]["all_convex"]["valid_share_per_batch"] = C.groupby("batch")["valid"].mean().to_dict()
        rep["modes"][mode] = m
    pd.concat(samples, ignore_index=True).to_csv(OUT / "site_samples.csv", index=False)
    rep["targets_detected"] = det

    runs = [V1] + list(MODES.values())
    split = {}
    for name in runs:
        run = Run(name)
        res = {w: {"detected": [], "not_detected": []} for w in ("combined", "core")}
        for tg in fold_targets():
            mp = run.maps(tg["stem"])
            a, _ = load_upright(run.cfg["batch1"], BATCHES["batch1"] / "stage_b" / name, tg["stem"])
            net_wh = (a["w_net"].shape[1], a["w_net"].shape[0])
            Y = Ys.get(tg["stem"])
            if Y is None:
                Y = Ys[tg["stem"]] = luminance(BATCHES["batch1"] / "stage_a", tg["stem"])
            pts = fold_target_points(tg, Y, net_wh, mp["up_wh"])
            for w in ("combined", "core"):
                res[w]["detected" if det[tg["name"]] else "not_detected"].append(covered(mp[f"{w}_full"], pts))
        split[name] = {w: {k: {"coverage": float(np.concatenate(v).mean()) if v else None, "n_targets": len(v)}
                           for k, v in d.items()} for w, d in res.items()}
    rep["fold_target_coverage_split"] = split

    sc = score(runs)
    rep["score"] = {name: {"g1": {w: {"classes": r["g1"][w]["classes"], "texture_rois": r["g1"][w]["texture_rois"]}
                                  for w in r["g1"]}, "rgb": r["rgb"],
                           "audit_geom_only": {w: r["audit"][w]["geom-only"] for w in r["audit"]},
                           "wfrac": r["wfrac"]} for name, r in sc.items()}
    U = {name: pd.DataFrame(r["units"]).set_index("id") for name, r in sc.items()}
    for mode, run_name in MODES.items():
        lost = {}
        for w in ("combined", "core"):
            a, b, gid = U[V1][f"frac_{w}"] >= 0.5, U[run_name][f"frac_{w}"] >= 0.5, U[V1]["bucket"]
            lost[w] = {bk: {"v1_closed": int(a[gid == bk].sum()), "closed": int(b[gid == bk].sum()),
                            "lost": sorted(int(i) for i in a[(gid == bk) & a & ~b].index),
                            "gained": sorted(int(i) for i in a[(gid == bk) & ~a & b].index)}
                       for bk in ("geom_gain", "geom_offset")}
        m = rep["modes"][mode]
        m["closures"] = lost
        ac = m["offsets"]["all_convex"]
        drop = ac["nms_frontal"]["median_abs"] - ac["resited_frontal"]["median_abs"]
        cov = {w: sc[run_name]["rgb"][w]["fold_targets"] - sc[V1]["rgb"][w]["fold_targets"] for w in ("combined", "core")}
        no_loss = all(not lost[w]["geom_gain"]["lost"] for w in ("combined", "core"))
        m["rule"] = {"convex_median_abs_drop_px": drop, "fold_target_cov_change": cov, "no_geom_gain_lost": no_loss,
                     "pass_combined": bool(drop >= 0.5 and cov["combined"] >= -0.01 and no_loss),
                     "pass_core": bool(drop >= 0.5 and cov["core"] >= -0.01 and no_loss)}
    (OUT / "site.json").write_text(json.dumps(rep, indent=1, default=float))

    # side-by-side tables
    print("offsets: median (median |.|), nms -> per mode; valid share per mode")
    for st in ("geom_offset", "fold_target"):
        subs = sorted(set().union(*[rep["modes"][m]["offsets"][st] for m in MODES]))
        for sub in subs:
            line = f"{st:11s} {sub:16s}"
            first = rep["modes"]["plane_intersection"]["offsets"][st][sub]
            line += f" n {first['n']:4d} | nms {first['nms']['median']:+.2f} ({first['nms']['median_abs']:.2f})"
            for mode in MODES:
                v = rep["modes"][mode]["offsets"][st].get(sub)
                line += f" | {mode[:5]} {v['resited']['median']:+.2f} ({v['resited']['median_abs']:.2f}) valid {v['valid_share']:.2f}"
            print(line)
    for key in ("all_convex", "all_convex_valid_only"):
        f0 = rep["modes"]["plane_intersection"]["offsets"][key]
        line = f"{key:28s} nms {f0['nms_frontal']['median']:+.2f} ({f0['nms_frontal']['median_abs']:.2f})"
        for mode in MODES:
            v = rep["modes"][mode]["offsets"][key]
            line += f" | {mode[:5]} n {v['n']} {v['resited_frontal']['median']:+.2f} ({v['resited_frontal']['median_abs']:.2f}) valid {v['valid_share']:.2f}"
        print(line)
    for mode in MODES:
        m = rep["modes"][mode]
        print(f"\n{mode}: validity {m['validity']}; rule {m['rule']}; closures {m['closures']}")
    print("\ncoverage split:", json.dumps(split, indent=0, default=float))
    for name in runs:
        r = rep["score"][name]
        print(name, "rgb (depth edges, fold targets, fold candidates)",
              {w: (round(v["depth_edges"], 4), round(v["fold_targets"], 4), round(v["fold_candidates"], 4),
                   v["n_fold_candidates"]) for w, v in r["rgb"].items()},
              "| cov", {w: {c: round(v["coverage"], 4) for c, v in r["g1"][w]["classes"].items() if v["coverage"]}
                        for w in r["g1"]},
              "| texture", {w: {k: round(v, 4) for k, v in r["g1"][w]["texture_rois"].items()} for w in r["g1"]},
              "| geom_gain px", {w: round(r["audit_geom_only"][w]["geom_gain | relaxed per pixel"]["share"], 4)
                                 for w in r["audit_geom_only"]})


if __name__ == "__main__":
    main()
