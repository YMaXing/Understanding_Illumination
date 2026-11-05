"""Thin-structure fragmentation diagnostic (report only, 504, geom_v1_1's depth parameters): where do the contours of the
25 component-stage step-gap units break, and does the same happen on foliage and the tri_fp units?

Per image: the depth channel's own steps (illum_baseline.geom.candidates.depth_candidates, depth_null, depth.jump with
the channel's obstacles), but jump() is run for every candidate, not only those that pass D-3, so every candidate has
r and its D-4 / D-5 quantities.
Pre-gate pieces: candidates with r > tau_high inside the unit (dilated 2 px), before any test, linked within the link
radius (3 px) as the channel links components; each piece ordered into chains (geom.depth.chains, radius 3 px).
Tests, every one evaluated on every candidate, in pipeline order:
  polarity (D-3), height (two-sided lobe height > k_h s_q), sharpness (s_lo <= s_sep <= s_hi), centring,
  plane fit (both sides fitted with planes, i.e. not a doublet fallback: recorded, the channel does not reject on it),
  sign (d_rho > 0), significance (additive null > z), tau_low (r > tau_low), sky_near;
  then the component stage (distinct px >= 5 and component max r > tau_high) on the survivors.
Per test: the share of the pieces' candidates it rejects first; the breaks it alone would cause along the chains (runs of
its failures with passing candidates on both sides); the pieces under 5 px it alone would create (sub-pieces of
passing candidates, re-linked within 3 px, under 5 distinct px, from pieces that are >= 5 px); alternation vs
clustering: its failure runs against the expected number for the same count placed at random (ratio ~1: alternating,
<< 1: clustered). Then the pieces the full chain of tests leaves (accepted pieces).
Contour-level check (report only): per piece, the median of each test's statistic and whether that median passes.
Controls: foliage boxes on 080, 075, 082, 067 and the 11 tri_fp units.
Writes baseline_output/diag/geom/thin_frag/frag.json and pieces.csv.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geom_score import Run, audit, load_labels, unit_fractions  # noqa: E402
from illum_baseline.config import dense_fit_params, detector_params, load_config  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.candidates import depth_candidates, depth_field  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.depth import DepthParams, chains, depth_null, jump  # noqa: E402
from illum_baseline.geom.sampling import bilinear  # noqa: E402

OUT = ROOT / "baseline_output" / "diag" / "geom" / "thin_frag"
BATCH_ROOT = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
TESTS = ("polarity", "height", "sharpness", "centring", "plane_fit", "sign", "significance", "tau_low", "sky_near")
REJECTING = tuple(t for t in TESTS if t != "plane_fit")
FOLIAGE = {"taiwei_cui_080": [(0, 170, 0, 90)], "taiwei_cui_075": [(0, 130, 0, 140)],
           "taiwei_cui_082": [(270, 360, 265, 345)], "acharya_mrudula_067": [(300, 377, 30, 95), (225, 315, 345, 385)]}


def image_tests(stem, batch):
    root = BATCH_ROOT[batch]
    meta = json.loads((root / "stage_a" / f"{stem}.json").read_text())
    z = np.load(root / "stage_a" / f"{stem}.npz")
    cfg = load_config(root / "stage_b" / "geom_v1_1" / "config_resolved.yaml")
    sky = z["sky_prob"].astype(np.float64) if "sky_prob" in z else None
    ctx = GeometryContext(z["D"], sky, Intrinsics.from_stage_a(meta), detector_params(cfg), dense_fit_params(cfg))
    p = DepthParams.from_dict(dataclasses.asdict(cfg.geometry.depth))
    f = depth_field(ctx, p)
    c = depth_candidates(ctx, p, field=f)
    nul = depth_null(ctx, p, f)
    P = c.reason == ""
    with np.errstate(invalid="ignore", divide="ignore"):
        r_doublet = p.doublet_coef * 0.5 * (c.h_p + np.abs(c.h_m)) * c.s_sep**2 / c.rho_s0
    obst = c.x0[P][r_doublet[P] > p.tau_low]
    J = jump(ctx, p, c.x0, c.n, c.k_c, c.q_p, c.q_m, c.h_p, c.h_m, np.nan_to_num(c.s_sep), c.rho_s0, f.s_q, nul, obst)
    allpx = np.ones(f.q.shape, bool)
    qa, oka = bilinear(f.q, allpx, c.x0[:, 0] - c.n[:, 0] / 2, c.x0[:, 1] - c.n[:, 1] / 2)
    qb, okb = bilinear(f.q, allpx, c.x0[:, 0] + c.n[:, 0] / 2, c.x0[:, 1] + c.n[:, 1] / 2)
    signif = J["signif_add"] if p.null_model == "additive" else J["signif"]
    sky_edge = J["sky_edge"]
    with np.errstate(invalid="ignore"):
        T = pd.DataFrame({
            "polarity": oka & okb & (qa > 0) & (qb < 0),
            "height": np.minimum(c.q_p, -c.q_m) > p.k_h * f.s_q,
            "sharpness": (c.s_sep >= c.s_lo) & (c.s_sep <= c.s_hi),
            "centring": np.abs(c.s_p + c.s_m) <= p.centre_tol * c.s_sep,
            "plane_fit": ~J["thin"],
            "sign": sky_edge | (J["d_rho"] > 0),
            "significance": sky_edge | (signif > p.z),
            "tau_low": sky_edge | (J["r"] > p.tau_low),
            "sky_near": ~J["sky_near"]})
    stats = pd.DataFrame({"height": np.minimum(c.q_p, -c.q_m) / (p.k_h * f.s_q), "s_sep": c.s_sep,
                          "centring": np.abs(c.s_p + c.s_m) / c.s_sep, "signif": signif / p.z, "r": J["r"],
                          "thin": J["thin"]})
    return {"x0": c.x0, "n": c.n, "r": J["r"], "T": T, "stats": stats, "p": p, "s_lo": c.s_lo, "s_hi": c.s_hi,
            "shape": f.q.shape}


def components(x, radius):
    if not len(x):
        return np.zeros(0, int)
    pairs = cKDTree(x).query_pairs(radius, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
    return connected_components(g, directed=False)[1]


def distinct_px(x, W):
    return len(np.unique(np.rint(x[:, 1]).astype(int) * W + np.rint(x[:, 0]).astype(int)))


def expected_runs(n, k):
    return k * (n - k + 1) / n if n else 0.0


def analyse(region_mask, im, label, rid):
    """Pieces in a region and the per-test bookkeeping."""
    x0, n, r, T, st, p = im["x0"], im["n"], im["r"], im["T"], im["stats"], im["p"]
    H, W = im["shape"]
    m = ndimage.binary_dilation(region_mask, iterations=2)
    ui, vi = np.clip(np.rint(x0[:, 0]).astype(int), 0, W - 1), np.clip(np.rint(x0[:, 1]).astype(int), 0, H - 1)
    sel = np.nonzero(m[vi, ui] & (r > p.tau_high))[0]
    rows, per_test = [], {t: {"first": 0, "fails": 0, "breaks": 0, "short_pieces": 0, "runs": 0, "exp_runs": 0.0}
                          for t in TESTS}
    n_cand = 0
    if not len(sel):
        return rows, per_test, 0
    lab = components(x0[sel], p.link_radius)
    for k in np.unique(lab):
        idx = sel[lab == k]
        size = distinct_px(x0[idx], W)
        Tk = T.iloc[idx]
        n_cand += len(idx)
        pass_all = Tk[list(REJECTING)].all(axis=1).to_numpy()
        first = np.full(len(idx), "", dtype=object)
        for t in REJECTING[::-1]:
            first[~Tk[t].to_numpy()] = t
        for t in REJECTING:
            per_test[t]["first"] += int((first == t).sum())
        # chains for ordering
        chs = chains(x0[idx], n[idx], radius=p.link_radius)
        for t in TESTS:
            ok = Tk[t].to_numpy()
            per_test[t]["fails"] += int((~ok).sum())
            for ch in chs:
                seq = ok[ch]
                if len(seq) < 3:
                    continue
                lab1, nr = ndimage.label(~seq)
                per_test[t]["runs"] += nr
                per_test[t]["exp_runs"] += expected_runs(len(seq), int((~seq).sum()))
                for j in range(1, nr + 1):
                    w = np.nonzero(lab1 == j)[0]
                    if w[0] > 0 and w[-1] < len(seq) - 1:
                        per_test[t]["breaks"] += 1
            if size >= p.min_component_px:
                keep = idx[ok]
                if len(keep):
                    sub = components(x0[keep], p.link_radius)
                    per_test[t]["short_pieces"] += int(sum(distinct_px(x0[keep][sub == s], W) < p.min_component_px
                                                           for s in np.unique(sub)))
        # pieces the full chain of tests leaves (component stage on the survivors)
        surv = idx[pass_all]
        acc_px, n_acc = 0, 0
        if len(surv):
            sub = components(x0[surv], p.link_radius)
            for s in np.unique(sub):
                ii = surv[sub == s]
                if distinct_px(x0[ii], W) >= p.min_component_px and np.nanmax(r[ii]) > p.tau_high:
                    acc_px += distinct_px(x0[ii], W)
                    n_acc += 1
        stk = st.iloc[idx]
        med = {"height": float(np.nanmedian(stk.height)), "s_sep": float(np.nanmedian(stk.s_sep)),
               "centring": float(np.nanmedian(stk.centring)), "signif": float(np.nanmedian(stk.signif)),
               "r": float(np.nanmedian(stk.r)), "thin_share": float(stk.thin.mean())}
        med_pass = {"height": med["height"] > 1, "sharpness": im["s_lo"] <= med["s_sep"] <= im["s_hi"],
                    "centring": med["centring"] <= p.centre_tol, "significance": med["signif"] > 1,
                    "tau_low": med["r"] > p.tau_low}
        rows.append({"group": label, "region": rid, "piece": int(k), "n_cand": len(idx), "px": size,
                     "pass_all_share": float(pass_all.mean()), "accepted_pieces": n_acc, "accepted_px": acc_px,
                     **{f"first_{t}": int((first == t).sum()) for t in REJECTING},
                     **{f"median_{k_}": v for k_, v in med.items()}, **{f"median_pass_{k_}": v for k_, v in med_pass.items()}})
    return rows, per_test, n_cand


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab, draws, strata, totals, masks = load_labels()
    u = pd.read_csv(ROOT / "baseline_output" / "diag" / "geom" / "g4" / "units.csv")
    t = u[u.direction == "tri-only"]
    groups = {"component_step": sorted(t.id[t.stage == "component"]), "tri_fp": sorted(t.id[t.bucket == "tri_fp"])}
    batch_of = dict(zip(u.stem, u.batch))
    info = u.set_index("id")
    stems = sorted({info.loc[i, "stem"] for g in groups.values() for i in g} | set(FOLIAGE))
    all_rows, agg = [], {}
    for stem in stems:
        im = image_tests(stem, batch_of[stem])
        H, W = im["shape"]
        regions = []
        for g, ids in groups.items():
            for i in ids:
                if info.loc[i, "stem"] == stem:
                    mk = np.zeros(H * W, bool)
                    mk[masks[f"id{i:03d}"]] = True
                    regions.append((g, f"unit {i}", mk.reshape(H, W)))
        for j, (u0, u1, v0, v1) in enumerate(FOLIAGE.get(stem, [])):
            mk = np.zeros((H, W), bool)
            mk[v0:v1 + 1, u0:u1 + 1] = True
            regions.append(("foliage", f"{stem[-3:]} box {j}", mk))
        for g, rid, mk in regions:
            rows, pt, nc = analyse(mk, im, g, rid)
            for r_ in rows:
                r_["stem"] = stem
            all_rows += rows
            a = agg.setdefault(g, {"n_cand": 0, "tests": {t_: {k: 0.0 for k in ("first", "fails", "breaks", "short_pieces",
                                                                               "runs", "exp_runs")} for t_ in TESTS}})
            a["n_cand"] += nc
            for t_, d in pt.items():
                for k, v in d.items():
                    a["tests"][t_][k] += v
        print(stem, "done", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "pieces.csv", index=False)
    rep = {}
    for g, a in agg.items():
        n = max(a["n_cand"], 1)
        gp = df[df.group == g]
        rep[g] = {"n_pieces": len(gp), "n_cand": a["n_cand"], "pieces_ge_5px": int((gp.px >= 5).sum()),
                  "piece_px": {"P50": float(gp.px.median()) if len(gp) else None, "P90": float(gp.px.quantile(0.9)) if len(gp) else None},
                  "accepted_pieces": int(gp.accepted_pieces.sum()), "accepted_px": int(gp.accepted_px.sum()),
                  "pre_gate_px": int(gp.px.sum()), "pass_all_share": float((gp.pass_all_share * gp.n_cand).sum() / n),
                  "tests": {t_: {"first_reject_share": d["first"] / n, "fail_share": d["fails"] / n,
                                 "breaks": int(d["breaks"]), "pieces_under_5px": int(d["short_pieces"]),
                                 "runs_over_random": d["runs"] / d["exp_runs"] if d["exp_runs"] else None}
                            for t_, d in a["tests"].items()},
                  "median_pass_share": {k: float(gp[f"median_pass_{k}"].mean()) for k in
                                        ("height", "sharpness", "centring", "significance", "tau_low")} if len(gp) else {},
                  "median_pass_share_pieces_ge_5px": {k: float(gp[gp.px >= 5][f"median_pass_{k}"].mean()) for k in
                                                      ("height", "sharpness", "centring", "significance", "tau_low")}
                  if (gp.px >= 5).any() else {}}
    (OUT / "frag.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
