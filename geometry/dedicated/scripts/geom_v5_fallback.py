"""TASK_GEOM_V2 V5: the fallback Laplacian, sweep and selection (report only; RGB for the edge coverage).

Settings: d_excl in {2, 3, 4, 6} x pool_radius in {2, 3} x band_min in {2, 3}, with the V2-V4 options as settled
(fold.site hessian_peak, depth.short_thin off, fold.rescue_undecidable reject): the tables of geom_v1_hessian. Per image:
  depth w, fold w   re-rasterized from the accepted candidates at band_min (depth band max(band_min, ceil(sigma_wc) + 1)
                    + the depth_and_mask sky ring; fold band max(band_min, ceil(width / 2)) at the V2b site); band_min 2
                    reproduces the saved arrays (checked);
  Laplacian w       the saved laplacian_w_net of geom_v1_hessian (pool_radius 2) or geom_v1_r3 (3: tri_veto4's w);
  exclusion zone    illum_baseline.geom.fallback.exclusion_zone (fold accepted at the site, depth accepted except thin
                    doublets, sky edges kept), d_excl;
  combined w        min(depth, fold, fallback(Laplacian, zone)); core w = min(depth, fold).
Every setting is scored with scripts/geom_score.py's functions (combined per-pixel missed true geometry, band excess and
tri_fp still relaxed, texture ROIs, ROI coverage, RGB-edge coverage of depth edges / fold targets / fold candidates,
geom-only buckets, w < 0.5 per batch).
Selection rule (propose, don't switch): among the settings with missed true geometry (per pixel) <= geom_v1 + 0.02,
step / limb / fold ROI coverage each >= geom_v1 - 0.02, RGB-edge coverage of depth edges and fold targets each >=
geom_v1 - 0.02, and RGB-edge coverage of all accepted fold candidates >= geom_v1 - 0.02: the lowest band-excess share,
ties by the lower texture class share (all combined w).
Held-out: the same selection with the audit restricted to batch1 units, and the chosen setting's audit numbers on the
batch2 units. Retirement metric: core per-pixel missed true geometry (gap_unresolved apart). Check: the chosen setting
with all depth edges in the exclusion (thin doublets included).
Writes baseline_output/diag/geom/v2/fallback.json and fallback_settings.csv.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geom_g2_coverage import raster  # noqa: E402
from geom_score import (OUT, W_MIN, Run, audit, g1_table, load_labels, rgb_coverage, rgb_edge_set,  # noqa: E402
                        unit_fractions, wfrac)
from illum_baseline.geom.depth import sky_boundary_ring  # noqa: E402
from illum_baseline.geom.fallback import apply_fallback, exclusion_zone  # noqa: E402
from illum_baseline.outputs import upsample_weights  # noqa: E402
from illum_baseline.stage_b import load_upright  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
V1, BASE, R3 = "geom_v1", "geom_v1_hessian", "geom_v1_r3"
D_EXCL, RADII, BAND_MINS = (2, 3, 4, 6), (2, 3), (2, 3)
TOL = 0.02


def depth_w(dc, shape, S, band_min, prm):
    d = dc[dc["accepted"]]
    band = np.maximum(band_min, np.ceil(d["sigma_wc"].to_numpy()).astype(int) + 1)
    w = raster(shape, d["u"].to_numpy(), d["v"].to_numpy(), d["w_c"].to_numpy(), band)
    sk = d[d["sky_edge"]]
    if prm["sky_band"] == "depth_and_mask" and len(sk):
        seed = np.ones(shape, dtype=bool)
        seed[np.clip(sk["v"], 0, shape[0] - 1), np.clip(sk["u"], 0, shape[1] - 1)] = False
        m = sky_boundary_ring(S, prm["sky_ring_px"]) & (ndimage.distance_transform_edt(seed) <= prm["sky_ring_near_px"])
        w[m] = np.minimum(w[m], prm["w_min"])
    return w


def fold_w(fc, shape, band_min):
    f = fc[fc["accepted"]]
    band = np.maximum(band_min, np.ceil(f["width"].to_numpy() / 2).astype(int))
    return raster(shape, f["u_site"].to_numpy(), f["v_site"].to_numpy(), f["w_c"].to_numpy(), band)


class SettingRun(Run):
    """In-memory maps of one setting (network res precomputed; full res on demand, one image cached)."""

    def __init__(self, name, base: Run, net: dict, up: dict):
        self.name, self.cfg, self.stems = name, base.cfg, base.stems
        self.base, self.net, self.up = base, net, up
        self._cache = {}

    def maps(self, stem):
        if stem not in self._cache:
            comb, core = self.net[stem]
            up = self.up[stem]
            self._cache = {stem: {"combined_net": comb, "core_net": core, "up_wh": up,
                                  "combined_full": upsample_weights(comb, up, W_MIN),
                                  "core_full": upsample_weights(core, up, W_MIN)}}
        return self._cache[stem]


def batch_subset(draws, strata, totals, stems):
    return (draws[draws["stem"].isin(stems)], strata[strata["stem"].isin(stems)], totals[totals["stem"].isin(stems)])


def metrics(run, lab, draws, strata, totals, masks, edges, batch1, batch2) -> dict:
    U = unit_fractions(run, lab, masks)
    A = audit(run, U, draws, strata, totals)
    A1 = audit(run, U, *batch_subset(draws, strata, totals, batch1))
    A2 = audit(run, U, *batch_subset(draws, strata, totals, batch2))
    G = g1_table(run)
    R = rgb_coverage(run, edges)
    F = wfrac(run)

    def aud(a, w):
        t, g = a[w]["tri-only"], a[w]["geom-only"]
        return {"missed_px": t["missed true geometry | per pixel"], "missed_unit": t["missed true geometry | per unit"],
                "unresolved_px": t["gap_unresolved missed | per pixel"],
                "band_excess": t["tri_band_excess | relaxed per pixel"], "tri_fp": t["tri_fp | relaxed per pixel"],
                "texture_label": t["texture label | relaxed per pixel"],
                **{f"geom_only {b}": g[f"{b} | relaxed per pixel"] for b in ("geom_gain", "geom_offset",
                                                                              "geom_band_excess", "geom_fp")}}
    out = {}
    for w in ("combined", "core"):
        out[w] = {"audit": aud(A, w), "audit_batch1": aud(A1, w), "audit_batch2": aud(A2, w),
                  "coverage": {c: G[w]["classes"][c]["coverage"] for c in ("step", "limb", "fold", "crease")},
                  "texture_class": G[w]["classes"]["texture"]["share_w<0.5"], "texture_rois": G[w]["texture_rois"],
                  "rgb": R[w], "wfrac": {b: v["deliverable"] for b, v in F[w].items()}}
    return out


def flat(name, m, w="combined", aud="audit"):
    a = m[w][aud]
    return {"setting": name, "missed/px": a["missed_px"]["share"], "band excess": a["band_excess"]["share"],
            "band excess lo": a["band_excess"]["ci_lo"], "band excess hi": a["band_excess"]["ci_hi"],
            "tri_fp": a["tri_fp"]["share"], "texture label": a["texture_label"]["share"],
            **{f"cov {c}": v for c, v in m[w]["coverage"].items()}, "texture class": m[w]["texture_class"],
            **{f"tex {k}": v for k, v in m[w]["texture_rois"].items()},
            "rgb depth": m[w]["rgb"]["depth_edges"], "rgb fold targets": m[w]["rgb"]["fold_targets"],
            "rgb fold cand": m[w]["rgb"]["fold_candidates"],
            **{k.replace("geom_only ", "go "): a[k]["share"] for k in a if k.startswith("geom_only")},
            **{f"w<.5 {b}": v for b, v in m[w]["wfrac"].items()}}


def select(rows: pd.DataFrame, ref: dict) -> tuple[pd.DataFrame, str | None, dict]:
    ok = pd.DataFrame({
        "missed": rows["missed/px"] <= ref["missed/px"] + TOL,
        "step": rows["cov step"] >= ref["cov step"] - TOL, "limb": rows["cov limb"] >= ref["cov limb"] - TOL,
        "fold": rows["cov fold"] >= ref["cov fold"] - TOL,
        "rgb depth": rows["rgb depth"] >= ref["rgb depth"] - TOL,
        "rgb fold targets": rows["rgb fold targets"] >= ref["rgb fold targets"] - TOL,
        "rgb fold cand": rows["rgb fold cand"] >= ref["rgb fold cand"] - TOL}, index=rows.index)
    passing = rows[ok.all(axis=1)]
    if not len(passing):
        return ok, None, {c: int((~ok[c]).sum()) for c in ok}
    best = passing.sort_values(["band excess", "texture class"]).iloc[0]["setting"]
    return ok, best, {c: int((~ok[c]).sum()) for c in ok}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab, draws, strata, totals, masks = load_labels()
    edges = rgb_edge_set()
    base, r3 = Run(BASE), Run(R3)
    batch1, batch2 = base.stems["batch1"], base.stems["batch2"]
    pre, up, checks = {}, {}, []
    for batch, stems in base.stems.items():
        cfg = base.cfg[batch]
        prm = dataclasses.asdict(cfg.geometry.depth)
        for stem in stems:
            a, meta = load_upright(cfg, BATCHES[batch] / "stage_b" / BASE, stem)
            a3, _ = load_upright(r3.cfg[batch], BATCHES[batch] / "stage_b" / R3, stem)
            dc = pd.read_parquet(BATCHES[batch] / "stage_b" / BASE / f"{stem}_depth_candidates.parquet")
            fc = pd.read_parquet(BATCHES[batch] / "stage_b" / BASE / f"{stem}_fold_candidates.parquet")
            shape, S = a["w_net"].shape, a["sky_mask"].astype(bool)
            dw = {bm: depth_w(dc, shape, S, bm, prm) for bm in BAND_MINS}
            fw = {bm: fold_w(fc, shape, bm) for bm in BAND_MINS}
            checks.append(bool(np.array_equal(dw[2].astype(np.float16), a["depth_w_net"]) and
                               np.array_equal(fw[2].astype(np.float16), a["fold_w_net"])))
            lap = {2: a["laplacian_w_net"].astype(np.float64), 3: a3["laplacian_w_net"].astype(np.float64)}
            zones = {d: exclusion_zone(dc, fc, shape, d) for d in D_EXCL}
            pre[stem] = {"dw": dw, "fw": fw, "lap": lap, "zones": zones, "dc": dc, "fc": fc, "shape": shape}
            up[stem] = tuple(meta["geometry"]["upright_WxH"])
    print("band_min 2 re-rasterization reproduces the saved depth / fold w:", all(checks), f"({sum(checks)}/{len(checks)})")

    v1m = metrics(Run(V1), lab, draws, strata, totals, masks, edges, batch1, batch2)
    ref, ref1 = flat("geom_v1", v1m), flat("geom_v1", v1m, aud="audit_batch1")
    results, rows, rows1, rows2, core_rows = {}, [], [], [], []
    for d, r, bm in itertools.product(D_EXCL, RADII, BAND_MINS):
        name = f"d{d}_r{r}_b{bm}"
        net = {}
        for stem, p in pre.items():
            core = np.minimum(p["dw"][bm], p["fw"][bm])
            net[stem] = (np.minimum(core, apply_fallback(p["lap"][r], p["zones"][d])), core)
        m = metrics(SettingRun(name, base, net, up), lab, draws, strata, totals, masks, edges, batch1, batch2)
        excl = float(np.mean([pre[s]["zones"][d].mean() for s in pre]))
        results[name] = m | {"exclusion_share_mean": excl}
        rows.append(flat(name, m) | {"d_excl": d, "pool_radius": r, "band_min": bm, "exclusion share": excl})
        rows1.append(flat(name, m, aud="audit_batch1"))
        rows2.append(flat(name, m, aud="audit_batch2"))
        print(name, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "fallback_settings.csv", index=False)
    ok, best, binding = select(R, ref)
    ok1, best1, binding1 = select(pd.DataFrame(rows1).assign(**{c: R[c] for c in R if c.startswith(("cov ", "rgb "))}),
                                  ref1)
    rep = {"rebuild_bit_identical": all(checks), "geom_v1": v1m, "settings": results, "selection": {
        "full": {"chosen": best, "fails_per_constraint": binding, "pass": ok.all(axis=1).tolist()},
        "batch1_audit": {"chosen": best1, "fails_per_constraint": binding1},
        "same_choice": best == best1}}
    if best1 is not None:
        rep["selection"]["batch1_audit"]["chosen_on_batch2_units"] = flat(best1, results[best1], aud="audit_batch2")
        rep["selection"]["batch1_audit"]["geom_v1_on_batch2_units"] = flat("geom_v1", v1m, aud="audit_batch2")
    # retirement metric: core w (V2-V4 options on) at each band_min
    for bm in BAND_MINS:
        m = results[f"d3_r2_b{bm}"]["core"]["audit"]
        rep.setdefault("retirement_core", {})[f"band_min {bm}"] = {"missed_px": m["missed_px"],
                                                                   "gap_unresolved_px": m["unresolved_px"]}
    # all-edges check at the chosen setting
    if best is not None:
        d, r, bm = (int(x[1:]) for x in best.split("_"))
        net = {}
        for stem, p in pre.items():
            core = np.minimum(p["dw"][bm], p["fw"][bm])
            z = exclusion_zone(p["dc"], p["fc"], p["shape"], d, include_thin=True)
            net[stem] = (np.minimum(core, apply_fallback(p["lap"][r], z)), core)
        m = metrics(SettingRun(best + "_all_edges", base, net, up), lab, draws, strata, totals, masks, edges, batch1,
                    batch2)
        rep["all_edges_check"] = flat(best + " all edges", m)
    (OUT / "fallback.json").write_text(json.dumps(rep, indent=1, default=float))

    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 80)
    show = ["setting", "missed/px", "band excess", "tri_fp", "texture label", "cov step", "cov limb", "cov fold",
            "cov crease", "texture class", "tex 079 grass", "tex 079 grass sunlit", "rgb depth", "rgb fold targets",
            "rgb fold cand", "go geom_gain", "go geom_fp", "w<.5 batch1", "w<.5 batch2", "exclusion share"]
    print("geom_v1:", {k: round(v, 4) for k, v in ref.items() if isinstance(v, float)})
    print(R[show].round(4).to_string(index=False))
    print("pass:", dict(zip(R["setting"], ok.all(axis=1))))
    print("selection:", json.dumps(rep["selection"], indent=1, default=float))
    print("retirement (core):", json.dumps(rep["retirement_core"], indent=1, default=float))
    if "all_edges_check" in rep:
        print("all-edges check:", {k: round(v, 4) for k, v in rep["all_edges_check"].items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
