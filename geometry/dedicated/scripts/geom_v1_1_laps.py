"""Report only (no channel change): geom_v1_1 on the batch2 lap-siding and step images 068, 069, 070.

1. 068 right-wall laps: per lap edge, the fold candidates' dtheta, the chi^2 statistic T / lambda_hat against
   T_crit = 2 ln(1/p_fa) (the multiplicative gate is T > lambda_hat T_crit), and w_c; the depth channel's r at the same
   edges against tau_low: (a) the nearest depth candidate (its stage, and r where D-4 ran), (b) r measured at each fold
   candidate by the depth channel's own measure_jump (near side by rho_f, skip k_min, m = 8, contamination check).
2. 070 middle riser: per candidate on its two folds, the stage it reaches (window contamination -> gate -> capped
   veto -> component), with the top and bottom risers for comparison; each riser's face width in network px, from
   the fold lines (per column, lower fold v minus upper fold v) and from the dense normals (per column, pixels whose
   normal is closer to the riser's than to the tread's).
3. 069 laps: per lap edge, the candidates inside vs outside the pole's shadow (smoothed-luminance mask; RGB used for
   this evaluation only): DA3 ramp amplitude (peak-to-peak rotation of the dense normal along e1, |s| <= 6 px; and
   r_n), Laplacian q / tau (max of the vetoed q within |s| <= 3 px), fold acceptance rate (accepted or rescued).

Regions are hand-set boxes in the upright network frame (drawn in the check figure). Lap / riser edges are the
8-connected components of the candidates' rounded pixels (|e1_v| >= 0.85, dilated 1 x 7 px along the line).
Writes baseline_output/diag/geom/v1_1/laps.json and laps_regions.png.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rgb_edges import luminance  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.geom.sampling import bilinear  # noqa: E402
from illum_baseline.stage_b import load_upright  # noqa: E402

B2 = ROOT / "baseline_output_batch2"
RUN = B2 / "stage_b" / "geom_v1_1"
OUT = ROOT / "baseline_output" / "diag" / "geom" / "v1_1"
STEM = "acharya_mrudula_{}"
E1V_MIN = 0.85
ACC = ("accepted", "rescued")

# regions (u0, u1, v0, v1), upright network px
R068 = (284, 372, 32, 316)                        # right wall, right of the corner trim
R070 = {"top": (90, 290, 344, 382), "middle": (90, 290, 386, 414), "bottom": (90, 290, 443, 478)}
SPLIT070 = {"top": 362.5, "middle": 400.5, "bottom": 460.5}   # upper / lower fold of each riser
R069 = (18, 226, 44, 330)                          # left wall; below it the floor line v_floor(u)


def v_floor_069(u):
    return 338.0 - 0.277 * (np.asarray(u, dtype=np.float64) - 15.0) - 8.0


def box(df, b):
    return df[(df.u0 >= b[0]) & (df.u0 <= b[1]) & (df.v0 >= b[2]) & (df.v0 <= b[3])]


def edges(df, shape, min_n=12):
    """Label of each candidate's lap / riser edge (-1: none): components of the rounded pixels, dilated along u."""
    H, W = shape
    d = df[df.e1_v.abs() >= E1V_MIN]
    u = np.clip(np.rint(d.u0).astype(int), 0, W - 1)
    v = np.clip(np.rint(d.v0).astype(int), 0, H - 1)
    m = np.zeros(shape, bool)
    m[v, u] = True
    m = ndimage.binary_dilation(m, np.ones((1, 7), bool))
    lab, _ = ndimage.label(m, np.ones((3, 3), bool))
    L = pd.Series(lab[v, u], index=d.index)
    keep = L.value_counts()
    keep = keep[keep >= min_n].index
    out = pd.Series(-1, index=df.index)
    out.loc[L.index] = np.where(L.isin(keep), L, -1)
    # renumber top to bottom
    out = merge_lines(df, out)
    order = df[out >= 0].groupby(out[out >= 0]).v0.median().sort_values()
    ren = {old: i for i, old in enumerate(order.index)}
    return out.map(lambda x: ren.get(x, -1))


def merge_lines(df, lab, tol=1.5, dslope=0.04, max_overlap=10.0):
    """Join collinear pieces of one lap line (gaps where no candidate fired): same slope within dslope, the median
    distance of one piece to the other's fitted line below tol px, and side by side (u overlap < max_overlap px)."""
    labs = sorted(set(lab) - {-1})
    g = {l: df[lab == l] for l in labs}
    fit = {l: np.polyfit(g[l].u0, g[l].v0, 1) for l in labs}
    parent = {l: l for l in labs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            if abs(fit[a][0] - fit[b][0]) > dslope:
                continue
            ov = min(g[a].u0.max(), g[b].u0.max()) - max(g[a].u0.min(), g[b].u0.min())
            if ov >= max_overlap:
                continue
            ra = np.median(np.abs(np.polyval(fit[a], g[b].u0) - g[b].v0))
            rb = np.median(np.abs(np.polyval(fit[b], g[a].u0) - g[a].v0))
            if min(ra, rb) < tol:
                parent[find(b)] = find(a)
    return lab.map(lambda x: find(x) if x != -1 else -1)


def line_samples(df_all, d, a, tau, extra=None):
    """Per merged edge, every integer column u on its fitted line v(u) between its end candidates: the DA3 ramp
    amplitude, q / tau, and whether a candidate (any stage; accepted or rescued) lies within 1.5 px."""
    nhat, nval = a["n_hat"].astype(np.float64), a["normal_valid"].astype(bool)
    qmap = a["laplacian_q"].astype(np.float64)
    cand = df_all[df_all.e1_v.abs() >= E1V_MIN]
    tree_all = cKDTree(cand[["u0", "v0"]].to_numpy())
    acc = cand[cand.reason.isin(ACC)]
    tree_acc = cKDTree(acc[["u0", "v0"]].to_numpy())
    rows = []
    for e, g in d.groupby("edge"):
        b1, b0 = np.polyfit(g.u0, g.v0, 1)
        u = np.arange(math.ceil(g.u0.min()), math.floor(g.u0.max()) + 1, dtype=np.float64)
        v = b0 + b1 * u
        x0 = np.stack([u, v], -1)
        nrm = np.array([-b1, 1.0]) / math.hypot(b1, 1.0)
        e1 = np.broadcast_to(nrm, x0.shape)
        rows.append(pd.DataFrame({
            "edge": e, "u": u, "v": v, "amp_deg": normal_profile_amp(nhat, nval, x0, e1),
            "q_over_tau": q_over_tau(qmap, tau, x0, e1),
            "has_cand": np.isfinite(tree_all.query(x0, distance_upper_bound=1.5)[0]),
            "has_acc": np.isfinite(tree_acc.query(x0, distance_upper_bound=1.5)[0]) if len(acc) else False}))
    return pd.concat(rows, ignore_index=True)


def q(x, p):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float("nan")


def dist(x, scale=1.0):
    x = np.asarray(x, dtype=np.float64) * scale
    x = x[np.isfinite(x)]
    return {"n": int(x.size), "P10": q(x, 10), "P50": q(x, 50), "P90": q(x, 90)} if x.size else {"n": 0}


def load(s):
    stem = STEM.format(s)
    cfg = load_config(RUN / "config_resolved.yaml")
    a, meta = load_upright(cfg, RUN, stem)
    fc = pd.read_parquet(RUN / f"{stem}_fold_candidates.parquet")
    dc = pd.read_parquet(RUN / f"{stem}_depth_candidates.parquet")
    return cfg, a, meta, fc, dc


def context(cfg, s):
    from illum_baseline.config import dense_fit_params, detector_params
    from illum_baseline.geom.camera import Intrinsics
    from illum_baseline.geom.context import GeometryContext
    from illum_baseline.stage_a import stage_a_dir
    sa = stage_a_dir(cfg) if (stage_a_dir(cfg) / f"{STEM.format(s)}.npz").exists() else B2 / "stage_a"
    meta_a = json.loads((sa / f"{STEM.format(s)}.json").read_text())
    z = np.load(sa / f"{STEM.format(s)}.npz")
    sky = z["sky_prob"].astype(np.float64) if "sky_prob" in z else None
    return GeometryContext(z["D"], sky, Intrinsics.from_stage_a(meta_a), detector_params(cfg), dense_fit_params(cfg))


def depth_r_at(cfg, ctx, x0, e1):
    """The depth channel's r at points x0 along +-e1 (oriented to the near side by rho_f), skip k_min, m = 8."""
    import dataclasses
    from illum_baseline.geom.candidates import depth_field
    from illum_baseline.geom.depth import DepthParams, depth_null, measure_jump, relative_jump
    p = DepthParams.from_dict(dataclasses.asdict(cfg.geometry.depth))
    f = depth_field(ctx, p)
    nul = depth_null(ctx, p, f)
    ok = np.ones(ctx.shape, bool)
    a, _ = bilinear(ctx.rho_f, ok, (x0 + 4 * e1)[:, 0], (x0 + 4 * e1)[:, 1])
    b, _ = bilinear(ctx.rho_f, ok, (x0 - 4 * e1)[:, 0], (x0 - 4 * e1)[:, 1])
    n = np.where((a >= b)[:, None], e1, -e1)
    res = measure_jump(ctx, p, x0, n, p.k_min, nul)
    return relative_jump(res), res.valid, p


def normal_profile_amp(nhat, valid, x0, e1, half=6):
    """Peak-to-peak rotation (deg) of the dense normal along x0 + s e1, |s| <= half (bilinear, renormalized)."""
    s = np.arange(-half, half + 1, dtype=np.float64)
    pts = x0[:, None, :] + s[None, :, None] * e1[:, None, :]
    comp = []
    for c in range(3):
        val, ok = bilinear(nhat[..., c], valid, pts[..., 0], pts[..., 1])
        comp.append(np.where(ok, val, np.nan))
    n = np.stack(comp, -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    cos = np.einsum("nik,njk->nij", n, n)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return np.nanmax(ang.reshape(len(x0), -1), axis=1)


def q_over_tau(qmap, tau, x0, e1, half=3):
    s = np.arange(-half, half + 0.5, 0.5)
    pts = x0[:, None, :] + s[None, :, None] * e1[:, None, :]
    val, ok = bilinear(qmap, np.ones(qmap.shape, bool), pts[..., 0], pts[..., 1])
    return np.nanmax(np.where(ok, val, np.nan), axis=1) / tau


def counts(sr):
    return {k: int(v) for k, v in sr.value_counts().items()}


# ---------------------------------------------------------------- 1. 068
def part1():
    cfg, a, meta, fc, dc = load("068")
    T_crit = meta["channels"]["fold"]["T_crit"]
    lam = meta["channels"]["fold"]["lambda_hat"]
    d = box(fc, R068).copy()
    d["edge"] = edges(d, a["w_net"].shape)
    d = d[d.edge >= 0]
    x0 = d[["u0", "v0"]].to_numpy()
    e1 = d[["e1_u", "e1_v"]].to_numpy()
    ctx = context(cfg, "068")
    r_fold, r_valid, p = depth_r_at(cfg, ctx, x0, e1)
    d["r_at_fold"] = np.where(r_valid, r_fold, np.nan)
    tree = cKDTree(dc[["u0", "v0"]].to_numpy())
    dd, ii = tree.query(x0, distance_upper_bound=2.0)
    has = np.isfinite(dd)
    d["depth_reason"] = np.where(has, dc.reason.to_numpy()[np.minimum(ii, len(dc) - 1)], "no depth candidate (2 px)")
    d["depth_r"] = np.where(has, dc.r.to_numpy()[np.minimum(ii, len(dc) - 1)], np.nan)
    rows = []
    for e, g in d.groupby("edge"):
        acc = g.reason.isin(ACC)
        rows.append({
            "edge": int(e), "v_median": float(g.v0.median()), "u_range": [float(g.u0.min()), float(g.u0.max())],
            "n": len(g), "reason": counts(g.reason),
            "convexity": counts(np.sign(g.convexity).map({1.0: "convex", -1.0: "concave", 0.0: "0"})),
            "dtheta_deg": dist(g.dtheta, 180 / math.pi),
            "T_over_lambda": dist(g.T_over_lambda), "share_T_over_lambda_gt_T_crit": float((g.T_over_lambda > T_crit).mean()),
            "w_c_accepted_or_rescued": dist(g.w_c[acc]), "n_accepted_or_rescued": int(acc.sum()),
            "depth_r_at_fold": dist(g.r_at_fold), "share_r_at_fold_gt_tau_low": float((g.r_at_fold > p.tau_low).mean()),
            "depth_nearest_reason": counts(g.depth_reason), "depth_nearest_r": dist(g.depth_r)})
    acc = d.reason.isin(ACC)
    summ = {"n": len(d), "n_edges": int(d.edge.nunique()), "reason": counts(d.reason),
            "lambda_hat": lam, "T_crit": T_crit, "T_gate": lam * T_crit,
            "dtheta_deg": dist(d.dtheta, 180 / math.pi),
            "dtheta_deg_by_reason": {k: dist(g.dtheta, 180 / math.pi) for k, g in d.groupby("reason")},
            "dtheta_gate_deg": dist(d.dtheta_gate, 180 / math.pi),
            "T_over_lambda": dist(d.T_over_lambda),
            "T_over_lambda_by_reason": {k: dist(g.T_over_lambda) for k, g in d.groupby("reason")},
            "share_gate": float(d.gate.mean()),
            "w_c_accepted": dist(d.w_c[d.reason == "accepted"]), "w_c_rescued": dist(d.w_c[d.reason == "rescued"]),
            "w_c_accepted_or_rescued": dist(d.w_c[acc]),
            "tau_low": p.tau_low, "depth_r_at_fold": dist(d.r_at_fold),
            "share_r_at_fold_gt_tau_low": float((d.r_at_fold > p.tau_low).mean()),
            "share_r_at_fold_valid": float(np.isfinite(d.r_at_fold).mean()),
            "depth_nearest_reason": counts(d.depth_reason), "depth_nearest_r": dist(d.depth_r),
            "rescue_support": counts(d.rescue_support.astype(str)) if "rescue_support" in d else {}}
    return {"summary": summ, "edges": rows}, d


# ---------------------------------------------------------------- 2. 070
def side_status(g, s):
    excl = g[f"excl_{s}"] > 0.2
    st = np.where(g[f"planar_{s}"] & (g[f"m_used_{s}"] == 8), "plane m8",
                  np.where(g[f"planar_{s}"], "plane m4 (m8 contaminated)",
                           np.where(g[f"fallback_{s}"], "dense fallback (m8, m4 contaminated)",
                                    np.where(excl, "excluded (> 20 % unusable)", "no value"))))
    return pd.Series(st, index=g.index)


def part2():
    cfg, a, meta, fc, dc = load("070")
    T_crit = meta["channels"]["fold"]["T_crit"]
    out = {"T_crit": T_crit, "lambda_hat": meta["channels"]["fold"]["lambda_hat"], "risers": {}}
    nhat = a["n_hat"].astype(np.float64)
    keep = {}
    for name, b in R070.items():
        d = box(fc, b).copy()
        d = d[d.e1_v.abs() >= E1V_MIN]
        d["fold"] = np.where(d.v0 < SPLIT070[name], "upper", "lower")
        res = {}
        for fold, g in d.groupby("fold"):
            sp, sm = side_status(g, "p"), side_status(g, "m")
            worst = np.where((sp == "plane m8") & (sm == "plane m8"), "both m8",
                             np.where(sp.str.startswith("dense") | sm.str.startswith("dense"), "a side on the dense fallback",
                                      np.where(sp.str.startswith("excl") | sm.str.startswith("excl") |
                                               sp.str.startswith("no") | sm.str.startswith("no"), "a side unusable",
                                               "a side m4")))
            gated = g[g.gate]
            res[fold] = {
                "n": len(g), "v_median": float(g.v0.median()), "k1": counts(g.k1), "width_px": dist(g.width),
                "contamination": counts(pd.Series(worst)),
                "side_p": counts(sp), "side_m": counts(sm),
                "valid": int(g.valid.sum()), "gate_pass": int(g.gate.sum()),
                "T_over_lambda": dist(g.T_over_lambda), "dtheta_deg": dist(g.dtheta, 180 / math.pi),
                "veto_among_gated": {"n_gated": len(gated), "decidable": int(gated.decidable.sum()),
                                     "veto_pass": int(gated.veto_pass.sum()),
                                     "undecidable": int(gated.veto_undecidable.sum()),
                                     "narrow_face": int(gated.narrow_face.sum()),
                                     "C": dist(gated.C), "C_min_eff": dist(gated.C_min_eff), "C_curv": dist(gated.C_curv),
                                     "d_next_p": dist(gated.d_next_p), "d_next_m": dist(gated.d_next_m)},
                "final_reason": counts(g.reason), "accepted_or_rescued_share": float(g.reason.isin(ACC).mean())}
        # face width: fold lines per column, and normals
        cols = np.arange(b[0] + 10, b[1] - 9)
        up = d[d.fold == "upper"]
        lo = d[d.fold == "lower"]
        wl = []
        for u in cols:
            a1 = up.v0[(up.u0 - u).abs() <= 1.0]
            a2 = lo.v0[(lo.u0 - u).abs() <= 1.0]
            if len(a1) and len(a2):
                wl.append(a2.median() - a1.median())
        v0, v1 = int(b[2]), int(b[3])
        n_r = np.nanmedian(nhat[int(up.v0.median()) + 4:int(lo.v0.median()) - 3, cols], axis=(0, 1))
        n_t = np.nanmedian(nhat[int(lo.v0.median()) + 5:int(lo.v0.median()) + 9, cols], axis=(0, 1))
        n_r, n_t = n_r / np.linalg.norm(n_r), n_t / np.linalg.norm(n_t)
        seg = nhat[v0:v1 + 1][:, cols]
        riser_px = (seg @ n_r > seg @ n_t).sum(0)
        res["face_width_px"] = {"fold_lines": dist(wl), "normals": dist(riser_px),
                                "riser_tread_angle_deg": float(np.degrees(np.arccos(np.clip(n_r @ n_t, -1, 1))))}
        out["risers"][name] = res
        keep[name] = d
    return out, keep


# ---------------------------------------------------------------- 3. 069
def shadow_069(a):
    Y = luminance(B2 / "stage_a", STEM.format("069"))
    Ys = ndimage.gaussian_filter(Y, 4.0)
    H, W = Y.shape
    vv, uu = np.mgrid[0:H, 0:W]
    wall = (uu >= R069[0]) & (uu <= R069[1]) & (vv >= R069[2]) & (vv <= np.minimum(R069[3], v_floor_069(uu)))
    lv = np.log(np.clip(Ys[wall], 1e-6, None))
    hist, be = np.histogram(lv, 256)
    c = (be[:-1] + be[1:]) / 2
    w0, w1 = np.cumsum(hist), np.cumsum(hist[::-1])[::-1]
    m0 = np.cumsum(hist * c) / np.maximum(w0, 1)
    m1 = (np.cumsum((hist * c)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1))[::-1]
    j = np.argmax(w0[:-1] * w1[1:] * (m0[:-1] - m1[1:]) ** 2)
    thr = float(np.exp(c[j]))
    dark = wall & (Ys < thr)
    lab, nl = ndimage.label(dark)
    sizes = ndimage.sum(dark, lab, range(1, nl + 1))
    sh = lab == (1 + int(np.argmax(sizes)))
    sh = ndimage.binary_fill_holes(sh)
    edge_band = ndimage.binary_dilation(sh, iterations=3) & ~ndimage.binary_erosion(sh, iterations=3)
    return Y, Ys, wall, sh, edge_band, thr


def part3():
    cfg, a, meta, fc, dc = load("069")
    Y, Ys, wall, sh, band, thr = shadow_069(a)
    shape = a["w_net"].shape
    d = box(fc, R069).copy()
    d = d[d.v0 <= v_floor_069(d.u0)]
    d["edge"] = edges(d, shape)
    d = d[d.edge >= 0]
    u = np.clip(np.rint(d.u0).astype(int), 0, shape[1] - 1)
    v = np.clip(np.rint(d.v0).astype(int), 0, shape[0] - 1)
    d["shadow"] = np.where(band[v, u], "boundary", np.where(sh[v, u], "inside", "outside"))
    x0 = d[["u0", "v0"]].to_numpy()
    e1 = d[["e1_u", "e1_v"]].to_numpy()
    d["amp_deg"] = normal_profile_amp(a["n_hat"].astype(np.float64), a["normal_valid"].astype(bool), x0, e1)
    tau = meta["detector"]["tau"]
    d["q_over_tau"] = q_over_tau(a["laplacian_q"].astype(np.float64), tau, x0, e1)
    d["acc"] = d.reason.isin(ACC)
    d["Y"] = Ys[v, u]

    def stats(g):
        return {"n": len(g), "amp_deg": dist(g.amp_deg), "r_n_deg_per_px": dist(g.r_n, 180 / math.pi),
                "dtheta_deg": dist(g.dtheta, 180 / math.pi), "q_over_tau": dist(g.q_over_tau),
                "share_q_gt_tau": float((g.q_over_tau > 1).mean()) if len(g) else float("nan"),
                "accept_rate": float(g.acc.mean()) if len(g) else float("nan"),
                "accepted_share": float((g.reason == "accepted").mean()) if len(g) else float("nan"),
                "rescued_share": float((g.reason == "rescued").mean()) if len(g) else float("nan"),
                "reason": counts(g.reason), "T_over_lambda": dist(g.T_over_lambda),
                "luminance_smoothed": dist(g.Y)}
    # every column along each lap line (also where no candidate fired)
    fc_wall = box(fc, R069)
    fc_wall = fc_wall[fc_wall.v0 <= v_floor_069(fc_wall.u0)]
    ls = line_samples(fc_wall, d, a, tau)
    lu = np.clip(np.rint(ls.u).astype(int), 0, shape[1] - 1)
    lv = np.clip(np.rint(ls.v).astype(int), 0, shape[0] - 1)
    ls["shadow"] = np.where(band[lv, lu], "boundary", np.where(sh[lv, lu], "inside", "outside"))

    def col_stats(g):
        n = len(g)
        return {"columns": n, "amp_deg": dist(g.amp_deg), "q_over_tau": dist(g.q_over_tau),
                "share_q_gt_tau": float((g.q_over_tau > 1).mean()) if n else float("nan"),
                "share_columns_with_candidate": float(g.has_cand.mean()) if n else float("nan"),
                "share_columns_accepted_or_rescued": float(g.has_acc.mean()) if n else float("nan")}

    rows = []
    for e, g in d.groupby("edge"):
        c = ls[ls.edge == e]
        r = {"edge": int(e), "v_median": float(g.v0.median()), "u_range": [float(g.u0.min()), float(g.u0.max())]}
        for wh in ("inside", "outside"):
            r[wh] = {"candidates": stats(g[g["shadow"] == wh]), "line": col_stats(c[c["shadow"] == wh])}
        r["n_boundary"] = int((g["shadow"] == "boundary").sum())
        rows.append(r)
    both = [r for r in rows if r["inside"]["line"]["columns"] >= 10 and r["outside"]["line"]["columns"] >= 10]
    paired = {k: dist([r["inside"]["line"][k]["P50"] - r["outside"]["line"][k]["P50"] for r in both])
              for k in ("amp_deg", "q_over_tau")}
    for k in ("share_columns_with_candidate", "share_columns_accepted_or_rescued", "share_q_gt_tau"):
        paired[k] = dist([r["inside"]["line"][k] - r["outside"]["line"][k] for r in both])
    both_c = [r for r in rows if r["inside"]["candidates"]["n"] >= 5 and r["outside"]["candidates"]["n"] >= 5]
    paired["accept_rate_of_candidates"] = dist([r["inside"]["candidates"]["accept_rate"] -
                                                r["outside"]["candidates"]["accept_rate"] for r in both_c])
    summ = {"shadow_threshold_Y": thr, "shadow_share_of_wall": float(sh[wall].mean()), "tau": tau,
            "lambda_hat": meta["channels"]["fold"]["lambda_hat"], "T_crit": meta["channels"]["fold"]["T_crit"],
            "n_edges": len(rows),
            "inside": {"candidates": stats(d[d["shadow"] == "inside"]), "line": col_stats(ls[ls["shadow"] == "inside"])},
            "outside": {"candidates": stats(d[d["shadow"] == "outside"]),
                        "line": col_stats(ls[ls["shadow"] == "outside"])},
            "n_boundary": int((d["shadow"] == "boundary").sum()),
            "edges_crossing_the_shadow (>= 10 columns each side)": len(both),
            "per_edge_inside_minus_outside": paired,
            "edges_with_candidates_both_sides (>= 5 each)": len(both_c)}
    return {"summary": summ, "edges": rows}, d, (sh, wall)


def figure(d1, k2, d3, shw):
    fig, ax = plt.subplots(1, 3, figsize=(24, 11))
    col = {"accepted": "lime", "rescued": "cyan", "gate": "red", "veto": "orange", "veto_undecidable": "magenta",
           "invalid_window": "blue", "component": "white"}
    for i, (s, d) in enumerate([("068", d1), ("070", pd.concat(k2.values())), ("069", d3)]):
        im = plt.imread(B2 / "stage_a" / "srgb" / f"{STEM.format(s)}.png")
        ax[i].imshow(im, extent=(-0.5, 377.5, 503.5, -0.5))
        if s == "069":
            sh, wall = shw
            ax[i].contour(sh, levels=[0.5], colors="yellow", linewidths=1)
            ax[i].contour(wall, levels=[0.5], colors="white", linewidths=0.6)
        for r, c in col.items():
            g = d[d.reason == r]
            ax[i].scatter(g.u0, g.v0, s=2, c=c, label=f"{r} {len(g)}", linewidths=0)
        if "edge" in d:
            for e, g in d.groupby("edge"):
                ax[i].text(g.u0.max() + 2, g.v0.median(), str(e), color="w", fontsize=6)
        ax[i].legend(fontsize=7, markerscale=4, loc="lower left")
        ax[i].set_title(f"{STEM.format(s)} (geom_v1_1): analysed candidates by stage" +
                        ("; yellow: shadow mask" if s == "069" else ""))
    fig.tight_layout()
    fig.savefig(OUT / "laps_regions.png", dpi=80)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r1, d1 = part1()
    r2, k2 = part2()
    r3, d3, shw = part3()
    rep = {"068_right_wall_laps": r1, "070_risers": r2, "069_laps_shadow": r3}
    (OUT / "laps.json").write_text(json.dumps(rep, indent=1, default=float))
    figure(d1, k2, d3, shw)
    print(json.dumps({"068": r1["summary"], "070": r2, "069": r3["summary"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
