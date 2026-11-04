"""TASK_GEOM_INTEGRATION G5: geom_v1 results (diagnostic only).

1. G1 evaluation table (scripts/geom_eval.py) for tri_veto4, geom_v0, geom_v1 (batch1 pinned ROIs).
2. Audit re-scored with the G4 labels (audit_split/): a unit is closed if geom_v1's w < 0.5 on >= 50 % of it; shares
   per bucket with the stratified pps estimator (scripts/geom_g4.py Estimator). The named units 15, 37 (narrow faces),
   97, 124, 125 (textured-surface folds): w under geom_v1 and per channel.
3. Marginal contribution per channel and option, per batch: the geom_v1 w < 0.5 area that exists only because of it,
   i.e. where w_v1 < 0.5 but the min over everything else is >= 0.5. Channels: depth, fold, laplacian. Options:
   sky_band (depth w re-rasterized from the accepted depth candidates without the sky ring) and crease_rescue (fold w
   re-rasterized without the rescued candidates); the re-rasterization with everything reproduces the saved channel w
   (checked).
4. Report-only sweep of the Laplacian pooling radius r in {1, 2, 3} inside geom_v1 (w = min(depth, fold, lap_r); the
   Laplacian channel rerun from the Stage A cache with geometry.laplacian.pool_radius = r; the run's own radius
   reproduces the saved laplacian_w_net, checked): closure of the labelled gap units, tri_band_excess (and tri_fp) area still relaxed, the
   texture ROI shares (batch1). geom_v1 uses r = 2 (scripts/geom_g5_radius.py rule); geom_v1_r3 keeps r = 3.
Writes baseline_output/diag/geom/g5/g5.json and eval.json.
Run from geometry/dedicated:  env -u PYTHONPATH uv run python scripts/geom_g5.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geom_eval import CLASSES, area_mask, evaluate, roi_list  # noqa: E402
from geom_g2_coverage import raster  # noqa: E402
from geom_g4 import AUD, BUCKETS, GAPS, Estimator  # noqa: E402
from illum_baseline.config import dense_fit_params, detector_params, load_config  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.depth import sky_boundary_ring  # noqa: E402
from illum_baseline.geom.laplacian import LaplacianChannel  # noqa: E402
from illum_baseline.stage_b import load_upright  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
RUN = "geom_v1"
OUT = ROOT / "baseline_output" / "diag" / "geom" / "g5"
RADII = (1, 2, 3)
NAMED = (15, 37, 97, 124, 125)
SEED = 0


def depth_w(dc: pd.DataFrame, shape, S: np.ndarray, ring: bool, prm: dict) -> np.ndarray:
    d = dc[dc["accepted"]]
    w = raster(shape, d["u"].to_numpy(), d["v"].to_numpy(), d["w_c"].to_numpy(), d["band"].to_numpy().astype(int))
    if ring:
        sk = d[d["sky_edge"]]
        if len(sk):
            seed = np.ones(shape, dtype=bool)
            seed[np.clip(sk["v"], 0, shape[0] - 1), np.clip(sk["u"], 0, shape[1] - 1)] = False
            m = sky_boundary_ring(S, prm["sky_ring_px"]) & (ndimage.distance_transform_edt(seed) <= prm["sky_ring_near_px"])
            w[m] = np.minimum(w[m], prm["w_min"])
    return w


def fold_w(fc: pd.DataFrame, shape, rescued: bool) -> np.ndarray:
    f = fc[fc["accepted"] & (True if rescued else ~fc["rescued"])]
    return raster(shape, f["u"].to_numpy(), f["v"].to_numpy(), f["w_c"].to_numpy(), f["band"].to_numpy().astype(int))


def same16(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a.astype(np.float16), b.astype(np.float16)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- 1. G1 table
    ev = evaluate(["tri_veto4", "geom_v0", RUN])
    (OUT / "eval.json").write_text(json.dumps(ev, indent=1))
    cls = pd.DataFrame(ev["classes"])
    comb = cls[cls["w"] == "combined"].copy()
    comb["class"] = pd.Categorical(comb["class"], CLASSES, ordered=True)
    g1 = comb.pivot_table(index="class", columns="run", values=["share_w<0.5", "median_w", "coverage"], observed=False)
    per_ch = cls[cls["run"] == RUN].pivot_table(index="class", columns="w", values="share_w<0.5", observed=False)

    # ---- labels
    lab = pd.read_csv(AUD / "audit_labels.csv", dtype={"tags": str, "note": str}).fillna({"tags": "", "note": ""})
    draws = pd.read_csv(AUD / "audit_draws.csv")
    strata = pd.read_csv(AUD / "audit_strata.csv")
    totals = pd.DataFrame(json.loads((AUD / "audit_totals.json").read_text())["per_image"])
    masks = np.load(AUD / "audit_masks.npz")
    batch_of = dict(zip(strata["stem"], strata["batch"]))
    tex_rois = [r for r in roi_list() if r["class"] == "texture" and not r.get("legacy")]

    units, marg, checks, tex = [], [], [], {r_: {} for r_ in RADII}
    for batch, root in BATCHES.items():
        rd = root / "stage_b" / RUN
        cfg = load_config(rd / "config_resolved.yaml")
        prm_d = dataclasses.asdict(cfg.geometry.depth)
        for stem in sorted(s for s, b in batch_of.items() if b == batch):
            a, meta = load_upright(cfg, rd, stem)
            w = a["w_net"].astype(np.float64)
            wd, wf, wl = (a[k].astype(np.float64) for k in ("depth_w_net", "fold_w_net", "laplacian_w_net"))
            shape, S = w.shape, a["sky_mask"].astype(bool)
            dc = pd.read_parquet(rd / f"{stem}_depth_candidates.parquet")
            fc = pd.read_parquet(rd / f"{stem}_fold_candidates.parquet")
            wd_all, wd_noring = depth_w(dc, shape, S, True, prm_d), depth_w(dc, shape, S, False, prm_d)
            wf_all, wf_norescue = fold_w(fc, shape, True), fold_w(fc, shape, False)
            # Laplacian sweep (rerun from the Stage A cache)
            sa = root / "stage_a"
            meta_a = json.loads((sa / f"{stem}.json").read_text())
            z = np.load(sa / f"{stem}.npz")
            ctx = GeometryContext(z["D"], z["sky_prob"].astype(np.float64), Intrinsics.from_stage_a(meta_a),
                                  detector_params(cfg), dense_fit_params(cfg))
            base = dataclasses.asdict(detector_params(cfg))
            wl_r = {r_: LaplacianChannel().run(ctx, base | {"pool_radius": r_}).w_net for r_ in RADII}
            r_run = cfg.geometry.laplacian.pool_radius or 3   # null = the detector's r = 3
            checks.append({"stem": stem, "depth_raster": same16(wd_all, wd), "fold_raster": same16(wf_all, wf),
                           "lap_run_radius": same16(wl_r[r_run], wl), "combined": same16(np.minimum.reduce([wd, wf, wl]), w)})
            # ---- 3. marginal contributions
            lo = w < 0.5
            marg.append({"batch": batch, "stem": stem, "image_px": int(w.size), "w_lt_0.5": int(lo.sum()),
                         "depth": int((lo & (np.minimum(wf, wl) >= 0.5)).sum()),
                         "fold": int((lo & (np.minimum(wd, wl) >= 0.5)).sum()),
                         "laplacian": int((lo & (np.minimum(wd, wf) >= 0.5)).sum()),
                         "sky_band": int((lo & (np.minimum.reduce([wd_noring, wf, wl]) >= 0.5)).sum()),
                         "crease_rescue": int((lo & (np.minimum.reduce([wd, wf_norescue, wl]) >= 0.5)).sum())})
            # ---- texture ROIs (batch1) per r
            if batch == "batch1":
                for roi in tex_rois:
                    if stem.endswith(roi["stem"]):
                        m = area_mask(roi, shape)
                        for r_ in RADII:
                            tex[r_][roi["name"]] = float((np.minimum.reduce([wd, wf, wl_r[r_]])[m] < 0.5).mean())
            # ---- 2./4. units
            for r in lab[lab["stem"] == stem].itertuples():
                m = np.zeros(w.size, bool)
                m[masks[f"id{r.id:03d}"]] = True
                m = m.reshape(shape)
                row = {"id": r.id, "batch": batch, "stem": stem, "direction": r.direction, "label": r.label,
                       "bucket": r.bucket, "tags": r.tags, "area_px": r.area_px,
                       "frac_v1": float((w[m] < 0.5).mean()), "median_w_v1": float(np.median(w[m])),
                       "frac_depth": float((wd[m] < 0.5).mean()), "frac_fold": float((wf[m] < 0.5).mean()),
                       "frac_lap": float((wl[m] < 0.5).mean())}
                for r_ in RADII:
                    row[f"frac_r{r_}"] = float((np.minimum.reduce([wd, wf, wl_r[r_]])[m] < 0.5).mean())
                units.append(row)
    U = pd.DataFrame(units).sort_values("id").reset_index(drop=True)
    U["closed_v1"] = U["frac_v1"] >= 0.5
    for r_ in RADII:
        U[f"closed_r{r_}"] = U[f"frac_r{r_}"] >= 0.5
    U.to_csv(OUT / "units_v1.csv", index=False)
    M = pd.DataFrame(marg)

    # ---- estimates (tri-only and geom-only) of the closed / relaxed area per bucket, per setting
    est = []
    D = draws.merge(U, on=["id", "stem", "direction"], how="left")
    for direction in ("tri-only", "geom-only"):
        d = D[D["direction"] == direction].reset_index(drop=True)
        st = strata[strata["direction"] == direction]
        t = totals[totals["direction"] == direction]
        E = Estimator(d, st, float(t["area_px"].sum()), float(t["image_px"].sum()), rng)
        ind = {}
        for b in BUCKETS[direction]:
            ind[f"{b}"] = d["bucket"] == b
            ind[f"{b}|closed_v1"] = (d["bucket"] == b) & d["closed_v1"]
            for r_ in RADII:
                ind[f"{b}|closed_r{r_}"] = (d["bucket"] == b) & d[f"closed_r{r_}"]
        if direction == "tri-only":
            for r_ in RADII:
                ind[f"missed_true_geometry|r{r_}"] = d["bucket"].isin(GAPS) & ~d[f"closed_r{r_}"]
        df, _ = E.run(ind)
        df.insert(0, "direction", direction)
        est.append(df)
    EST = pd.concat(est, ignore_index=True)
    EST.to_csv(OUT / "estimates_v1.csv", index=False)

    # ---- sweep table
    tri = U[U["direction"] == "tri-only"]
    sweep = []
    for r_ in RADII:
        row = {"r": r_}
        for b in ("gap_fixable", "gap_unresolved", "gap_subscale", "tri_band_excess", "tri_fp"):
            g = tri[tri["bucket"] == b]
            row[f"{b} closed units"] = f"{int(g[f'closed_r{r_}'].sum())}/{len(g)}"
            e = EST[(EST["direction"] == "tri-only") & (EST["quantity"] == f"{b}|closed_r{r_}")].iloc[0]
            row[f"{b} area share"] = round(float(e["share"]), 4)
        bx = tri[tri["bucket"] == "tri_band_excess"]
        row["tri_band_excess px relaxed"] = float((bx[f"frac_r{r_}"] * bx["area_px"]).sum() / bx["area_px"].sum())
        e = EST[(EST["direction"] == "tri-only") & (EST["quantity"] == f"missed_true_geometry|r{r_}")].iloc[0]
        row["missed true geometry"] = round(float(e["share"]), 4)
        row.update({f"tex {k}": round(v, 4) for k, v in tex[r_].items()})
        sweep.append(row)
    SW = pd.DataFrame(sweep)

    # ---- marginals per batch
    mb = M.groupby("batch")[["image_px", "w_lt_0.5", "depth", "fold", "laplacian", "sky_band", "crease_rescue"]].sum()
    mrep = {}
    for b, r in mb.iterrows():
        mrep[b] = {"image_px": int(r["image_px"]), "w_lt_0.5_px": int(r["w_lt_0.5"]),
                   **{k: {"px": int(r[k]), "share_of_image": float(r[k] / r["image_px"]),
                          "share_of_w_lt_0.5": float(r[k] / r["w_lt_0.5"])}
                      for k in ("depth", "fold", "laplacian", "sky_band", "crease_rescue")}}

    named = U[U["id"].isin(NAMED)][["id", "stem", "label", "bucket", "tags", "frac_v1", "median_w_v1", "frac_depth",
                                    "frac_fold", "frac_lap"]]
    closures = U.groupby(["direction", "bucket", "label"]).agg(n=("id", "size"), closed_v1=("closed_v1", "sum"),
                                                               area_px=("area_px", "sum")).reset_index()
    rep = {"g1_classes": comb.to_dict(orient="records"), "g1_per_channel_v1": per_ch.reset_index().to_dict(orient="records"),
           "audit_estimates": EST.to_dict(orient="records"), "closures_v1": closures.to_dict(orient="records"),
           "named_units": named.to_dict(orient="records"), "marginal": mrep, "marginal_per_image": marg,
           "sweep": sweep, "checks": checks, "all_checks_pass": bool(all(all(v for k, v in c.items() if k != "stem")
                                                                       for c in checks))}
    (OUT / "g5.json").write_text(json.dumps(rep, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))

    pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 200)
    print("G1 (combined w): \n" + g1.round(3).to_string())
    print("\ngeom_v1 per channel, share w<0.5:\n" + per_ch.round(3).to_string())
    print("\nchecks pass:", rep["all_checks_pass"])
    if not rep["all_checks_pass"]:
        print(pd.DataFrame(checks).to_string(index=False))
    print("\nclosures under geom_v1:\n" + closures.to_string(index=False))
    print("\nestimates:\n" + EST[EST["quantity"].str.contains("closed_v1|missed")].round(4).to_string(index=False))
    print("\nnamed units:\n" + named.round(3).to_string(index=False))
    print("\nmarginal (px, share of image, share of w<0.5):")
    for b, v in mrep.items():
        print(b, v["image_px"], v["w_lt_0.5_px"], {k: (v[k]["px"], round(v[k]["share_of_image"], 4),
                                                         round(v[k]["share_of_w_lt_0.5"], 4))
                                                     for k in ("depth", "fold", "laplacian", "sky_band", "crease_rescue")})
    print("\nsweep:\n" + SW.T.to_string())


if __name__ == "__main__":
    main()
