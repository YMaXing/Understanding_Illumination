"""TASK_GEOM_V2 W2 (report and selection, propose only): geometry.laplacian.hysteresis ridge_bridge, kappa 0.6 / 0.8
(runs geom_v1_1_hys06 / geom_v1_1_hys08) against geom_v1_1, both batches; scored with scripts/geom_score.py's functions.

Constraints, each on all units and on the batch1 units (same verdict required):
  combined w: texture ROIs each <= +0.01; 083 interior unchanged (|diff| < 0.0005); texture-labelled and tri_fp
    relaxed <= +0.02; band excess <= +0.01 (paired bootstrap CI reported); geom_fp not increased; step / limb / fold
    ROI coverage >= -0.01; crease coverage >= -0.02; 088 rock ridge coverage >= 0.70; geom_gain >= -0.01; missed true
    geometry per pixel <= +0.002; RGB-edge coverage of depth edges, fold targets, fold candidates each >= -0.005;
  core w: every one of those numbers unchanged to 4 decimals (the change is confined to the Laplacian).
Gain target: along the 068 right-wall and the 069 lap lines, the share of columns with combined w < 0.5 rises by
  >= 0.10 on 068, on 069 and inside the 069 shadow.
Report: per kappa the hysteresis counts; the gap-length distribution of continuation runs on the lap lines (068;
  069 inside / outside the shadow) and of the sub-threshold stretches along the lines; the single-ended extension
  variant (report only: recomputed from the saved q, combined with geom_v1_1's depth and fold weights); q' / tau along
  the 070 middle riser's two folds (P10 / P50 / P90, share in (kappa tau, tau]).
Writes baseline_output/diag/geom/w2/hysteresis.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geom_score import Run, g1_table, load_labels, rgb_edge_set  # noqa: E402
from geom_v1_1_laps import R068, R069, box, edges, shadow_069, v_floor_069  # noqa: E402
from geom_v5_fallback import BATCHES, SettingRun, metrics, paired  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.detector import disk  # noqa: E402
from illum_baseline.geom.hysteresis import ridge_hysteresis  # noqa: E402
from illum_baseline.geom.sampling import bilinear  # noqa: E402
from illum_baseline.stage_b import load_upright  # noqa: E402

REF = "geom_v1_1"
RUNS = {0.6: "geom_v1_1_hys06", 0.8: "geom_v1_1_hys08"}
OUT = ROOT / "baseline_output" / "diag" / "geom" / "w2"
B2 = BATCHES["batch2"]
RISERS = {"middle": (386, 414, 400.5)}


def dist(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "P10": float(np.percentile(x, 10)), "P50": float(np.percentile(x, 50)),
            "P90": float(np.percentile(x, 90)), "max": float(x.max())}


# ---------------------------------------------------------------- lap lines
def lap_lines():
    """Fitted lap lines (from geom_v1_1's fold candidates, fixed for every run): {stem: [(b1, b0, u_lo, u_hi)]}."""
    out = {}
    for s, b, extra in (("068", R068, None), ("069", R069, v_floor_069)):
        stem = f"acharya_mrudula_{s}"
        fc = pd.read_parquet(B2 / "stage_b" / REF / f"{stem}_fold_candidates.parquet")
        d = box(fc, b).copy()
        if extra is not None:
            d = d[d.v0 <= extra(d.u0)]
        d["edge"] = edges(d, (504, 378))
        lines = []
        for _, g in d[d.edge >= 0].groupby("edge"):
            b1, b0 = np.polyfit(g.u0, g.v0, 1)
            lines.append((b1, b0, math.ceil(g.u0.min()), math.floor(g.u0.max())))
        out[stem] = lines
    return out


def columns(lines, shape):
    """Per line, every integer column on it: (line id, u, v)."""
    rows = []
    for i, (b1, b0, lo, hi) in enumerate(lines):
        u = np.arange(lo, hi + 1)
        rows.append(pd.DataFrame({"line": i, "u": u, "v": b0 + b1 * u}))
    c = pd.concat(rows, ignore_index=True)
    c["ui"], c["vi"] = np.clip(np.rint(c.u).astype(int), 0, shape[1] - 1), np.clip(np.rint(c.v).astype(int), 0, shape[0] - 1)
    return c


def line_max(arr, c, lines, half=1.5, reducer=np.nanmax):
    """Per column, the max (or min) of arr within |s| <= half across the line (bilinear)."""
    b1 = np.array([lines[i][0] for i in c.line])
    nrm = np.stack([-b1, np.ones_like(b1)], -1) / np.hypot(b1, 1.0)[:, None]
    s = np.arange(-half, half + 0.25, 0.5)
    x0 = c[["u", "v"]].to_numpy(dtype=float)
    pts = x0[:, None, :] + s[None, :, None] * nrm[:, None, :]
    v, ok = bilinear(arr, np.ones(arr.shape, bool), pts[..., 0], pts[..., 1])
    return reducer(np.where(ok, v, np.nan), axis=1)


def stretches(mask_cols: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Runs of consecutive columns with c[flag] True per line: (line, length, u0, u1, cols index)."""
    rows = []
    for ln, g in mask_cols.groupby("line"):
        f = g[flag].to_numpy()
        idx = g.index.to_numpy()
        lab, n = ndimage.label(f)
        for k in range(1, n + 1):
            sel = idx[lab == k]
            rows.append({"line": ln, "length": len(sel), "first": int(sel[0]), "last": int(sel[-1])})
    return pd.DataFrame(rows, columns=["line", "length", "first", "last"])


# ---------------------------------------------------------------- recompute from the saved q (report only)
def lap_from_q(meta, q, kappa, single_ended):
    det = meta["detector"]
    tau, eps, r_pool = det["tau"], det["eps"], det.get("pool_radius", det["r"])
    h = ridge_hysteresis(q, tau, kappa, 16, single_ended=single_ended)
    q_eff = np.maximum(np.maximum(q - tau, 0.0), h["q_cont"])
    q_pool = ndimage.grey_dilation(q_eff, footprint=disk(r_pool), mode="constant", cval=0.0)
    w = np.maximum(0.05, eps / (q_pool + eps))
    w[q_pool == 0] = 1.0
    return w, h


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab, draws, strata, totals, masks = load_labels()
    edges_rgb = rgb_edge_set()
    ref = Run(REF)
    b1, b2 = ref.stems["batch1"], ref.stems["batch2"]
    cfg_ref = {b: load_config(r / "stage_b" / REF / "config_resolved.yaml") for b, r in BATCHES.items()}

    # single-ended variant (report only) and the bridge recompute check, from the saved q
    variants, recompute_check, per_image_variant = {}, {}, {}
    for kappa in RUNS:
        for single in (False, True):
            net, up = {}, {}
            for batch, stems in ref.stems.items():
                for stem in stems:
                    a, meta = load_upright(cfg_ref[batch], BATCHES[batch] / "stage_b" / REF, stem)
                    wl, h = lap_from_q(meta, a["laplacian_q"].astype(np.float64), kappa, single)
                    core = np.minimum(a["depth_w_net"].astype(np.float64), a["fold_w_net"].astype(np.float64))
                    net[stem] = (np.minimum(core, wl), core)
                    up[stem] = tuple(meta["geometry"]["upright_WxH"])
                    r = h["runs"]
                    per_image_variant.setdefault(f"k{kappa}_{'single' if single else 'bridge'}", {})[stem] = {
                        "runs_kept": int(r.bridged.sum()), "px_kept": int(h["bridged"].sum())}
                    if not single:
                        ra, _ = load_upright(load_config(BATCHES[batch] / "stage_b" / RUNS[kappa] / "config_resolved.yaml"),
                                             BATCHES[batch] / "stage_b" / RUNS[kappa], stem)
                        dw = np.abs(ra["laplacian_w_net"].astype(np.float64) - wl)
                        recompute_check.setdefault(kappa, []).append(float((dw > 1e-3).mean()))
            if single:
                variants[kappa] = SettingRun(f"single_k{kappa}", ref, net, up)

    runs = {REF: ref} | {RUNS[k]: Run(RUNS[k]) for k in RUNS} | {f"single_k{k}": v for k, v in variants.items()}
    M = {n: metrics(r, lab, draws, strata, totals, masks, edges_rgb, b1, b2) for n, r in runs.items()}
    G = {n: g1_table(r) for n, r in runs.items()}

    def num(n, w, aud="audit"):
        m, a = M[n][w], M[n][w][aud]
        rock = next(x for x in G[n][w]["rois"] if x["roi"] == "088 rock ridge")["coverage"]
        return {"texture_rois": m["texture_rois"], "texture label": a["texture_label"]["share"],
                "tri_fp": a["tri_fp"]["share"], "band excess": a["band_excess"]["share"],
                "geom_fp": a["geom_only geom_fp"]["share"], "geom_gain": a["geom_only geom_gain"]["share"],
                **{f"cov {c}": v for c, v in m["coverage"].items()}, "088 rock ridge": rock,
                "missed/px": a["missed_px"]["share"],
                **{f"rgb {k}": m["rgb"][k] for k in ("depth_edges", "fold_targets", "fold_candidates")},
                "texture class": m["texture_class"], **{f"w<0.5 {b}": v for b, v in m["wfrac"].items()}}

    def constraints(n, aud):
        r, s = num(REF, "combined", aud), num(n, "combined", aud)
        out = {}
        for roi, v in s["texture_rois"].items():
            d = v - r["texture_rois"][roi]
            if "083" in roi and "interior" in roi:
                out[f"083 interior unchanged"] = (abs(d) < 5e-4, d)
            out[f"texture ROI {roi} <= +0.01"] = (d <= 0.01, d)
        lim = {"texture label": ("<=", 0.02), "tri_fp": ("<=", 0.02), "band excess": ("<=", 0.01),
               "geom_fp": ("<=", 0.0), "cov step": (">=", -0.01), "cov limb": (">=", -0.01),
               "cov fold": (">=", -0.01), "cov crease": (">=", -0.02), "geom_gain": (">=", -0.01),
               "missed/px": ("<=", 0.002), "rgb depth_edges": (">=", -0.005), "rgb fold_targets": (">=", -0.005),
               "rgb fold_candidates": (">=", -0.005)}
        for k, (op, t) in lim.items():
            d = s[k] - r[k]
            out[f"combined {k} {op} {t:+g}"] = ((d <= t) if op == "<=" else (d >= t), d)
        out["combined 088 rock ridge >= 0.70"] = (s["088 rock ridge"] >= 0.70, s["088 rock ridge"])
        rc, sc = num(REF, "core", aud), num(n, "core", aud)
        diffs = {k: (sc[k] - rc[k]) for k in rc if k != "texture_rois"}
        diffs |= {f"tex {k}": sc["texture_rois"][k] - v for k, v in rc["texture_rois"].items()}
        worst = max(diffs.items(), key=lambda kv: abs(kv[1]))
        out["core unchanged (4 decimals)"] = (abs(worst[1]) < 5e-5, worst[1])
        return {k: {"ok": bool(ok), "diff": float(d)} for k, (ok, d) in out.items()}

    rep = {"constraints": {}, "constraints_batch1_audit": {}, "paired": {}, "paired_batch1": {}, "numbers": {},
           "recompute_check_share_px_differing": {k: float(np.mean(v)) for k, v in recompute_check.items()}}
    from geom_v5_fallback import batch_subset
    for n in runs:
        rep["numbers"][n] = {w: num(n, w) for w in ("combined", "core")}
        if n == REF:
            continue
        rep["constraints"][n] = constraints(n, "audit")
        rep["constraints_batch1_audit"][n] = constraints(n, "audit_batch1")
        rep["paired"][n] = paired(M[n]["_U"], M[REF]["_U"], draws, strata, totals)
        rep["paired_batch1"][n] = paired(M[n]["_U"], M[REF]["_U"], *batch_subset(draws, strata, totals, b1))

    # gain target and gap lengths on the lap lines
    lines = lap_lines()
    _, Ys, wall, sh, band, thr = shadow_069(None)
    gain, gaps = {}, {}
    for stem, ls in lines.items():
        cfg = load_config(B2 / "stage_b" / REF / "config_resolved.yaml")
        a, meta = load_upright(cfg, B2 / "stage_b" / REF, stem)
        tau = meta["detector"]["tau"]
        c = columns(ls, a["w_net"].shape)
        c["shadow"] = "all"
        if stem.endswith("069"):
            c["shadow"] = np.where(band[c.vi, c.ui], "boundary", np.where(sh[c.vi, c.ui], "inside", "outside"))
        c["q_tau"] = line_max(a["laplacian_q"].astype(np.float64), c, ls) / tau
        groups = {"all": c.index} | ({"inside": c.index[c.shadow == "inside"], "outside": c.index[c.shadow == "outside"]}
                                     if stem.endswith("069") else {})
        g_rec = {}
        for n in [REF, *RUNS.values(), *(f"single_k{k}" for k in RUNS)]:
            if n.startswith("single"):
                comb = runs[n].maps(stem)["combined_net"]
            else:
                comb = load_upright(load_config(B2 / "stage_b" / n / "config_resolved.yaml"), B2 / "stage_b" / n, stem)[0][
                    "w_net"].astype(np.float64)
            wmin = line_max(comb, c, ls, reducer=np.nanmin)
            g_rec[n] = {k: float((wmin[ix] < 0.5).mean()) for k, ix in groups.items()}
        gain[stem] = g_rec | {"columns": {k: int(len(ix)) for k, ix in groups.items()}}
        # gap lengths: sub-threshold stretches along the lines, and continuation runs on the lines
        c["sub"] = c.q_tau <= 1.0
        st = stretches(c, "sub")
        rec = {}
        for kappa in RUNS:
            st[f"all_above_k{kappa}"] = [bool((c.q_tau.loc[f:l] > kappa).all()) for f, l in zip(st["first"], st["last"])]
            h = lap_from_q(meta, a["laplacian_q"].astype(np.float64), kappa, False)[1]
            hs = lap_from_q(meta, a["laplacian_q"].astype(np.float64), kappa, True)[1]
            runmap, r = h["run_map"], h["runs"].set_index("run")
            on = np.zeros(a["w_net"].shape, bool)
            vv, uu = np.mgrid[0:on.shape[0], 0:on.shape[1]]
            for b1_, b0_, lo, hi in ls:
                on |= (np.abs(vv - (b0_ + b1_ * uu)) <= 1.5) & (uu >= lo) & (uu <= hi)
            ids = np.unique(runmap[on & (runmap >= 0) & h["cont"]])
            rr = r.loc[ids].copy()
            single_kept = hs["runs"].set_index("run").loc[ids, "bridged"] if len(ids) else pd.Series(dtype=bool)
            rr["single_kept"] = single_kept.to_numpy() if len(ids) else []
            if stem.endswith("069"):
                where = []
                for i in ids:
                    pv, pu = np.nonzero(runmap == i)
                    where.append("inside" if sh[pv, pu].mean() > 0.5 else "outside")
                rr["shadow"] = where
            else:
                rr["shadow"] = "all"
            rec[f"k{kappa}"] = {wh: {"continuation_runs": dist(g.n_px), "n_runs": len(g),
                                     "n_bridged": int(g.bridged.sum()), "bridged_px": int(g.n_px[g.bridged].sum()),
                                     "n_single_kept": int(g.single_kept.sum()),
                                     "runs_touching_2_seed_runs": int((g.n_seed_runs >= 2).sum()),
                                     "runs_longer_than_16": int((g.n_px > 16).sum())}
                                for wh, g in rr.groupby("shadow")}
        if stem.endswith("069"):
            st["shadow"] = [c.shadow.loc[f:l].mode().iloc[0] for f, l in zip(st["first"], st["last"])]
        else:
            st["shadow"] = "all"
        rec["sub_threshold_stretches"] = {
            wh: {"length_px": dist(g.length), "n": len(g), "longer_than_16": int((g.length > 16).sum()),
                 **{f"all_above_k{k}": int(g[f"all_above_k{k}"].sum()) for k in RUNS},
                 **{f"all_above_k{k}_and_<=16": int((g[f"all_above_k{k}"] & (g.length <= 16)).sum()) for k in RUNS}}
            for wh, g in st.groupby("shadow")}
        rec["line_q_over_tau"] = {wh: dist(c.q_tau[ix]) for wh, ix in groups.items()}
        gaps[stem] = rec
    rep["gain"], rep["gaps"] = gain, gaps

    # 070 middle riser q' / tau along its two folds (geom_v1_1)
    stem = "acharya_mrudula_070"
    cfg = load_config(B2 / "stage_b" / REF / "config_resolved.yaml")
    a, meta = load_upright(cfg, B2 / "stage_b" / REF, stem)
    tau = meta["detector"]["tau"]
    fc = pd.read_parquet(B2 / "stage_b" / REF / f"{stem}_fold_candidates.parquet")
    riser = {}
    v0, v1, sp = RISERS["middle"]
    d = fc[(fc.u0 >= 90) & (fc.u0 <= 290) & (fc.v0 >= v0) & (fc.v0 <= v1) & (fc.e1_v.abs() >= 0.85)]
    for f, g in d.groupby(np.where(d.v0 < sp, "upper", "lower")):
        x0, e1 = g[["u0", "v0"]].to_numpy(), g[["e1_u", "e1_v"]].to_numpy()
        s = np.arange(-3, 3.25, 0.5)
        pts = x0[:, None, :] + s[None, :, None] * e1[:, None, :]
        v, _ = bilinear(a["laplacian_q"].astype(np.float64), np.ones(a["w_net"].shape, bool), pts[..., 0], pts[..., 1])
        r = np.nanmax(v, 1) / tau
        riser[f] = {"q_over_tau": dist(r), "share_gt_tau": float((r > 1).mean()),
                    **{f"share_in_({k}tau,tau]": float(((r > k) & (r <= 1)).mean()) for k in RUNS}}
    rep["riser_070_middle"] = riser

    rep["hysteresis_scalars"] = {}
    for kappa, n in RUNS.items():
        rows = {}
        for b, root in BATCHES.items():
            for p in sorted((root / "stage_b" / n).glob("*.json")):
                j = json.loads(p.read_text())
                if "channels" in j:
                    lp = j["channels"]["laplacian"]
                    rows[p.stem] = {k: lp[k] for k in ("n_ridge_px", "n_seed_px", "n_continuation_px",
                                                       "n_continuation_runs", "n_bridged_runs", "n_bridged_px",
                                                       "frac_w_lt_0.5_before_hysteresis", "frac_w_lt_0.5")}
        rep["hysteresis_scalars"][n] = rows
    rep["single_ended_per_image"] = per_image_variant

    def target(n):
        g68, g69 = gain["acharya_mrudula_068"], gain["acharya_mrudula_069"]
        return {"068": g68[n]["all"] - g68[REF]["all"], "069": g69[n]["all"] - g69[REF]["all"],
                "069 inside": g69[n]["inside"] - g69[REF]["inside"]}
    rep["verdict"] = {}
    for kappa, n in RUNS.items():
        t = target(n)
        ok_c = all(v["ok"] for v in rep["constraints"][n].values())
        ok_1 = all(v["ok"] for v in rep["constraints_batch1_audit"][n].values())
        rep["verdict"][n] = {"constraints": ok_c, "constraints_batch1_audit": ok_1, "gain": t,
                             "gain_met": all(v >= 0.10 for v in t.values()),
                             "propose": bool(ok_c and ok_1 and all(v >= 0.10 for v in t.values()))}
    for kappa in RUNS:
        rep["verdict"][f"single_k{kappa} (report only)"] = {"gain": target(f"single_k{kappa}"),
                                                           "constraints_failed": [k for k, v in rep["constraints"][f"single_k{kappa}"].items() if not v["ok"]]}
    (OUT / "hysteresis.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("verdict", "riser_070_middle", "recompute_check_share_px_differing")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
