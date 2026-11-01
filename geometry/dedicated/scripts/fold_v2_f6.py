"""TASK_FOLD_V2 F6: targets and diagnostics for the fold channel (fold_only configuration, multiplicative null).

1. Targets: rough polylines (upright network px) snapped onto the dense-normal edge; per target the candidates within
   TOL px by outcome, the accepted coverage, dense normal-angle profiles at 25/50/75% with a plateau-step test, a
   virtual candidate at the midpoint (dtheta(k1), dtheta(k2), C, gate statistics) and diag-fold on the nearest real
   candidate.
2. no_veto: which k2 samples are excluded and why (outside the image, sky, invalid normal, and why the normal is
   invalid: border band, sky band, sum W < min_weight_sum, condition number).
3. The C > 1.18 tail by class and location.

Writes baseline_output/diag/fold_v2/f6/ (profiles, diag-fold figures, f6.json) and prints the tables.
Run from geometry/dedicated:  env -u PYTHONPATH uv run python scripts/fold_v2_f6.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.diag_fold import diag_fold  # noqa: E402
from illum_baseline.geom.candidates import rounding_width  # noqa: E402
from illum_baseline.geom.fold import FoldChannel, FoldParams  # noqa: E402
from illum_baseline.geom.sampling import bilinear  # noqa: E402
from illum_baseline.geom.twosided import null_projection, quad_form, two_sided_fit  # noqa: E402
from illum_baseline.stage_b import run_dir, run_geometry  # noqa: E402

OUT = ROOT / "baseline_output" / "diag" / "fold_v2" / "f6"
TOL = 3.0
# rough polylines (upright network px), snapped below
ROUGH = {
    "079 plinth edge": ("079", "narrow", [(242, 297), (206, 325)]),
    "076 plinth top edge L": ("076", "narrow", [(95, 292), (190, 348)]),
    "076 plinth bottom edge L": ("076", "narrow", [(92, 302), (190, 366)]),
    "076 plinth top edge R": ("076", "narrow", [(192, 344), (253, 305)]),
    "076 plinth bottom edge R": ("076", "narrow", [(197, 362), (260, 310)]),
    "090 plinth edge upper": ("090", "narrow", [(190, 363), (285, 262)]),
    "090 plinth edge lower": ("090", "narrow", [(200, 367), (297, 267)]),
    "080 side face edge L": ("080", "narrow", [(233, 92), (213, 200), (192, 295)]),
    "080 side face edge R": ("080", "narrow", [(260, 95), (238, 200), (226, 290)]),
    "070 wall seam u227": ("070", "crease", [(227, 15), (227, 190)]),
    "070 wall seam u272": ("070", "crease", [(272, 15), (272, 190)]),
    "083 collar groove": ("083", "crease", [(148, 186), (190, 199), (228, 186)]),
}


def canon(e):
    e = np.asarray(e, float) / np.linalg.norm(e)
    return e if (e[0] > 1e-12 or (abs(e[0]) <= 1e-12 and e[1] > 0)) else -e


def profile(ctx, x, e, L=30.0, step=0.25):
    s = np.arange(-L, L + 1e-9, step)
    pts = x[None] + s[:, None] * e[None]
    n, ok = bilinear(ctx.n_hat, ctx.normal_valid, pts[:, 0], pts[:, 1])
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    i0 = np.nonzero(ok)[0]
    if not i0.size:
        return s, np.full(len(s), np.nan)
    ref = n[i0[0]]
    ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(n, ref), axis=-1), n @ ref))
    return s, np.where(ok, ang, np.nan)


def plateau_step(s, ang, a, L=24.0):
    """step = |median(+plateau) - median(-plateau)|; flatness = the larger range (P90 - P10) of the two plateaus,
    plateaus at |s| in [a, L]."""
    lo = ang[(s <= -a) & (s >= -L)]
    hi = ang[(s >= a) & (s <= L)]
    lo, hi = lo[np.isfinite(lo)], hi[np.isfinite(hi)]
    if lo.size < 8 or hi.size < 8:
        return float("nan"), float("nan")
    rng = max(np.percentile(lo, 90) - np.percentile(lo, 10), np.percentile(hi, 90) - np.percentile(hi, 10))
    return float(abs(np.median(hi) - np.median(lo))), float(rng)


def snap(ctx, pts, every=10.0, search=5.0):
    """Move a rough polyline onto the dense-normal edge: along each segment every `every` px, the point of maximum
    |d angle / ds| within +-search px along the segment normal; refit each segment through them (PCA)."""
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        t = (b - a) / np.linalg.norm(b - a)
        e = canon([-t[1], t[0]])
        hits = []
        for c in np.linspace(0, np.linalg.norm(b - a), max(int(np.linalg.norm(b - a) / every), 2) + 1):
            x = a + c * t
            s, ang = profile(ctx, x, e, L=search + 2, step=0.25)
            d = np.abs(np.gradient(ang, s))
            m = (np.abs(s) <= search) & np.isfinite(d)
            if m.any():
                hits.append(x + s[m][np.argmax(d[m])] * e)
        if len(hits) < 2:
            out.append((a, b))
            continue
        h = np.array(hits)
        cen = h.mean(0)
        _, _, vt = np.linalg.svd(h - cen)
        dvec = vt[0]
        proj = lambda p: cen + ((p - cen) @ dvec) * dvec  # noqa: E731
        out.append((proj(a), proj(b)))
    # join consecutive segments at the average of the shared endpoints
    poly = [out[0][0]]
    for i in range(len(out) - 1):
        poly.append(0.5 * (out[i][1] + out[i + 1][0]))
    poly.append(out[-1][1])
    return np.array(poly)


def poly_samples(poly, step=0.5):
    out = [poly[:1]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.concatenate(out)


def dist_to_poly(x, poly):
    S = poly_samples(poly)
    d = np.sqrt(((x[:, None, :] - S[None]) ** 2).sum(-1))
    return d.min(1)


def point_at(poly, frac):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    c = np.concatenate([[0], np.cumsum(seg)])
    t = frac * c[-1]
    i = min(np.searchsorted(c, t, side="right") - 1, len(seg) - 1)
    a, b = poly[i], poly[i + 1]
    tan = (b - a) / seg[i]
    return a + (t - c[i]) * tan, canon([-tan[1], tan[0]])


def setup(stem, cfg):
    meta_a = json.loads((ROOT / "baseline_output/stage_a" / f"{stem}.json").read_text())
    c = np.load(ROOT / "baseline_output/stage_a" / f"{stem}.npz")
    ctx, results, _ = run_geometry(cfg, c["D"], c["sky_prob"].astype(np.float64), meta_a)
    return ctx, results[0]


def virtual(ctx, res, p, x, e):
    """The fold measure of a virtual candidate at x with direction e (no contamination check at k1 either: these
    are diagnostics of the geometry, the channel's own outcome is in the candidate table)."""
    width, _ = rounding_width(ctx, x[None], e[None], p.profile_half_len, p.width_min, p.width_max)
    k1 = int(max(p.k_min, math.ceil(width[0] / 2) + p.k_margin))
    kw = dict(m_min=p.m_min, max_excluded=p.max_excluded, fallback=False)
    r1 = two_sided_fit(ctx, x[None], e[None], k1, p.m, p.w_t, rms_max=None, **kw)
    r2 = two_sided_fit(ctx, x[None], e[None], 2 * k1, p.m, p.w_t, rms_max=None, **kw)
    sc = res.scalars
    T_add = float("nan")
    if r1.valid[0]:
        d1, S1 = null_projection(r1.p.nhat, r1.m.nhat, r1.p.Sigma_n, r1.m.Sigma_n)
        T_add = float(quad_form(d1, S1, p.lambda_n, sc["sigma_geo"] ** 2)[0])
    return {"k1": k1, "width": float(width[0]), "valid_k1": bool(r1.valid[0]), "valid_k2": bool(r2.valid[0]),
            "dtheta_k1": math.degrees(r1.dtheta[0]), "dtheta_k2": math.degrees(r2.dtheta[0]),
            "C": float(r1.dtheta[0] / r2.dtheta[0]) if r2.valid[0] else float("nan"),
            "T_over_lambda": float(r1.T[0] / sc["lambda_hat"]), "T_add": T_add, "T_crit": sc["T_crit"],
            "excl_k2": [float(r2.p.excl[0]), float(r2.m.excl[0])], "rms_rel_k1": [float(r1.p.rms_rel[0]),
                                                                                  float(r1.m.rms_rel[0])]}


def invalid_cause(ctx, p_dense):
    """(H, W) int: 0 valid normal, 1 border band, 2 sky / sky band, 3 sum W < min, 4 cond, 5 other."""
    H, W = ctx.shape
    R = p_dense.R_n
    cause = np.zeros((H, W), int)
    nv = ctx.normal_valid
    border = np.zeros((H, W), bool)
    border[:R, :] = border[-R:, :] = border[:, :R] = border[:, -R:] = True
    from illum_baseline.detector import dilate

    skyband = ~ctx.omega | dilate(~ctx.omega, R)
    low = ctx.dense.m0 < p_dense.min_weight_sum
    cond = np.nan_to_num(ctx.dense.cond, nan=0) > p_dense.cond_max
    for code, m in ((5, ~nv), (4, ~nv & cond), (3, ~nv & low), (2, ~nv & skyband), (1, ~nv & border)):
        cause[m] = code
    return cause


CAUSES = {0: "ok", 1: "border band", 2: "sky", 3: "sumW<min (step band)", 4: "cond", 5: "other", 6: "outside image"}


def k2_exclusion(ctx, cause, row, p):
    """Per side of the k2 windows: excluded share and the causes of the excluded samples (worst corner)."""
    x0, e1 = np.array([row.u0, row.v0]), np.array([row.e1_u, row.e1_v])
    et = np.array([-e1[1], e1[0]])
    H, W = ctx.shape
    out = {}
    for sign, tag in ((1, "p"), (-1, "m")):
        s = sign * (row.k2 + np.arange(p.m + 1.0))
        t = np.arange(-p.w_t, p.w_t + 1.0)
        pts = (x0 + s[:, None, None] * e1 + t[None, :, None] * et).reshape(-1, 2)
        u0, v0 = np.floor(pts[:, 0]).astype(int), np.floor(pts[:, 1]).astype(int)
        inside = (u0 >= 0) & (u0 <= W - 2) & (v0 >= 0) & (v0 <= H - 2)
        uc, vc = np.clip(u0, 0, W - 2), np.clip(v0, 0, H - 2)
        corners = np.stack([cause[vc, uc], cause[vc, uc + 1], cause[vc + 1, uc], cause[vc + 1, uc + 1]], -1)
        worst = np.where(inside, corners.max(-1), 6)
        exc = worst > 0
        rows_t = exc.reshape(len(s), len(t))
        out[tag] = {"excl": float(exc.mean()),
                    "causes": {CAUSES[c]: int((worst == c).sum()) for c in np.unique(worst[exc])},
                    "excluded_t_columns": int(rows_t.any(0).sum()), "excluded_s_rows": int(rows_t.any(1).sum())}
    return out


def cand_class(df, D, p):
    """Per candidate: fold / rim / limb / texture (fold_channel.md section 6.3)."""
    cls = []
    for r in df.itertuples():
        e = np.array([r.e1_u, r.e1_v])
        half = r.k1 + p.m / 2
        s = np.linspace(-half, half, int(4 * half) + 1)
        pts = np.array([r.u0, r.v0]) + s[:, None] * e
        z = D[np.clip(np.rint(pts[:, 1]).astype(int), 0, D.shape[0] - 1),
              np.clip(np.rint(pts[:, 0]).astype(int), 0, D.shape[1] - 1)]
        step = bool((np.abs(np.diff(np.log(z))) > 0.05).any())
        both, one = bool(r.planar_p and r.planar_m), bool(r.planar_p or r.planar_m)
        cls.append(("rim" if one else "limb") if step else ("fold" if both else "texture"))
    return np.array(cls)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "candidate_fold_only.yaml")
    p = FoldParams.from_dict(dataclasses.asdict(cfg.geometry.fold))
    from illum_baseline.config import dense_fit_params

    pd_ = dense_fit_params(cfg)
    rd = run_dir(cfg)
    report = {"targets": {}, "no_veto": {}, "C_tail": {}}
    stems = sorted({v[0] for v in ROUGH.values()} | {"079", "083"})
    cache = {}
    for s in stems:
        cache[s] = setup(f"taiwei_cui_{s}", cfg)

    # 1. targets
    for name, (s, kind, pts) in ROUGH.items():
        ctx, res = cache[s]
        df = res.candidates
        poly = snap(ctx, pts)
        d = dist_to_poly(df[["u0", "v0"]].to_numpy(), poly)
        near = df[d <= TOL]
        acc = near[near["accepted"]]
        S = poly_samples(poly)
        if len(acc):
            dd = np.sqrt(((S[:, None, :] - acc[["u0", "v0"]].to_numpy()[None]) ** 2).sum(-1)).min(1)
            coverage = float((dd <= TOL).mean())
        else:
            coverage = 0.0
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.6), constrained_layout=True)
        prof = []
        for i, frac in enumerate((0.25, 0.5, 0.75)):
            x, e = point_at(poly, frac)
            sv, ang = profile(ctx, x, e)
            w, _ = rounding_width(ctx, x[None], e[None], p.profile_half_len, p.width_min, p.width_max)
            a = max(w[0] / 2 + 1, 4.0)
            stp, rng = plateau_step(sv, ang, a)
            prof.append({"x": x.round(1).tolist(), "step_deg": stp, "plateau_range_deg": rng, "width": float(w[0])})
            ax[i].plot(sv, ang, "k-")
            ax[i].axvspan(-a, a, color="tab:orange", alpha=0.2)
            ax[i].set_title(f"({x[0]:.0f}, {x[1]:.0f}) step {stp:.1f} deg, plateau range {rng:.1f}", fontsize=9)
            ax[i].set_xlabel("s along e1 [px]")
        ax[0].set_ylabel("dense normal angle [deg]")
        fig.suptitle(f"{name} (taiwei_cui_{s})")
        fig.savefig(OUT / f"profile_{name.replace(' ', '_')}.png", dpi=80)
        plt.close(fig)
        xm, em = point_at(poly, 0.5)
        virt = virtual(ctx, res, p, xm, em)
        fig_path = None
        if len(near):
            j = np.argmin(dist_to_poly(near[["u0", "v0"]].to_numpy(), np.array([xm, xm + 1e-6])))
            r = near.iloc[j]
            fig_path = str(diag_fold(cfg, rd, f"taiwei_cui_{s}", float(r.u0), float(r.v0)).relative_to(ROOT))
        report["targets"][name] = {
            "stem": s, "kind": kind, "polyline": poly.round(1).tolist(), "length_px": float(np.linalg.norm(
                np.diff(poly, axis=0), axis=1).sum()), "n_near": int(len(near)),
            "reasons": near["reason"].value_counts().to_dict(), "no_veto_accepted": int((near.accepted & near.no_veto).sum()),
            "accepted_coverage": coverage,
            "dtheta_k1_median_deg": float(np.degrees(near.loc[near.valid, "dtheta"].median())) if near.valid.any() else float("nan"),
            "C_median": float(near.loc[near.gate, "C"].median()) if near.gate.any() else float("nan"),
            "profiles": prof, "virtual_mid": virt, "diag_fold": fig_path}

    # 2. no_veto
    for s in ("083", "079"):
        ctx, res = cache[s]
        df = res.candidates
        cause = invalid_cause(ctx, pd_)
        nv = df[df.accepted & df.no_veto]
        agg = {}
        rows = []
        for r in nv.itertuples():
            ex = k2_exclusion(ctx, cause, r, p)
            for tag in ("p", "m"):
                if ex[tag]["excl"] > p.max_excluded:
                    for c, n in ex[tag]["causes"].items():
                        agg[c] = agg.get(c, 0) + n
            rows.append({"u": r.u, "v": r.v, "k2": r.k2, "e1": [round(r.e1_u, 2), round(r.e1_v, 2)],
                         "excl_p": ex["p"]["excl"], "excl_m": ex["m"]["excl"],
                         "causes_p": ex["p"]["causes"], "causes_m": ex["m"]["causes"],
                         "t_cols_p": ex["p"]["excluded_t_columns"], "t_cols_m": ex["m"]["excluded_t_columns"]})
        # the locations named in the task: 083 bollard limbs, 079 box left silhouette
        named = {"083": [(157, 250), (157, 300), (228, 250), (228, 300), (159, 220)],
                 "079": [(116, 150), (116, 200), (117, 240), (118, 130), (114, 175)]}[s]
        named_rows = []
        for (u, v) in named:
            j = int(np.argmin((df.u0 - u) ** 2 + (df.v0 - v) ** 2))
            r = df.iloc[j]
            ex = k2_exclusion(ctx, cause, r, p) if r.gate else None
            named_rows.append({"at": [u, v], "cand": [round(r.u0, 1), round(r.v0, 1)], "reason": r.reason,
                               "no_veto": bool(r.no_veto), "C": float(r.C), "dtheta_deg": float(np.degrees(r.dtheta)),
                               "k2_excl": None if ex is None else [ex["p"]["excl"], ex["m"]["excl"]],
                               "fig": str(diag_fold(cfg, rd, f"taiwei_cui_{s}", float(r.u0), float(r.v0)).relative_to(ROOT))})
        # a few real no_veto cases with figures
        figs = []
        for r in nv.iloc[np.linspace(0, len(nv) - 1, min(4, len(nv))).astype(int)].itertuples():
            figs.append(str(diag_fold(cfg, rd, f"taiwei_cui_{s}", float(r.u0), float(r.v0)).relative_to(ROOT)))
        report["no_veto"][s] = {"n": int(len(nv)), "excluded_sample_causes": agg, "cases": rows[:12],
                                "named_locations": named_rows, "figures": figs,
                                "border_dist_px": [float(np.min([nv.u.min(), nv.v.min(), ctx.shape[1] - 1 - nv.u.max(),
                                                                  ctx.shape[0] - 1 - nv.v.max()]))] if len(nv) else []}

    # 2b. all images: no_veto excluded-sample causes and distance to the border
    allc = {}
    for f in sorted(rd.glob("*_fold_candidates.parquet")):
        s = f.name[11:14]
        if s not in cache:
            cache[s] = setup(f"taiwei_cui_{s}", cfg)
        ctx, res = cache[s]
        df = res.candidates
        cause = invalid_cause(ctx, pd_)
        nv = df[df.accepted & df.no_veto]
        H, W = ctx.shape
        bd = np.minimum.reduce([nv.u0, nv.v0, W - 1 - nv.u0, H - 1 - nv.v0]) if len(nv) else np.array([])
        agg = {}
        for r in nv.itertuples():
            ex = k2_exclusion(ctx, cause, r, p)
            for tag in ("p", "m"):
                if ex[tag]["excl"] > p.max_excluded:
                    for c, n in ex[tag]["causes"].items():
                        agg[c] = agg.get(c, 0) + n
        allc[s] = {"n": int(len(nv)), "border_dist_median": float(np.median(bd)) if len(bd) else float("nan"),
                   "share_within_30px_of_border": float((bd <= 30).mean()) if len(bd) else float("nan"), "causes": agg}

        # 3. C tail
        g = df[df.gate & ~df.no_veto]
        tail = g[g.C > 1.18]
        cls = cand_class(tail, ctx.D.astype(float), p) if len(tail) else np.array([])
        bdt = np.minimum.reduce([tail.u0, tail.v0, W - 1 - tail.u0, H - 1 - tail.v0]) if len(tail) else np.array([])
        report["C_tail"][s] = {
            "n_gate_with_k2": int(len(g)), "n_tail": int(len(tail)),
            "classes": {c: int((cls == c).sum()) for c in ("fold", "rim", "limb", "texture")},
            "accepted_share": float(tail.accepted.mean()) if len(tail) else float("nan"),
            "dtheta_k1_median": float(np.degrees(tail.dtheta.median())) if len(tail) else float("nan"),
            "dtheta_k2_median": float(np.degrees(tail.dtheta_k2.median())) if len(tail) else float("nan"),
            "border_dist_median": float(np.median(bdt)) if len(bdt) else float("nan"),
            "fallback_k1_share": float((tail.fallback_p | tail.fallback_m).mean()) if len(tail) else float("nan"),
            "depth_jump_share": float(tail.depth_jump.mean()) if len(tail) else float("nan")}
    report["no_veto_all"] = allc
    (OUT / "f6.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
