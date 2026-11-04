"""TASK_FOLD_V2 F7: calibrate the capped veto's C_min_eff = 1 - a (1 - C_curv) and its decidability limit.

Synthetic, noise 5e-4 on rho before the trilateral filter:
- smooth curvature: gate-passed candidates of the box + cylinders scene whose k1 and 2 k1 windows lie entirely on one
  cylinder at least R_s + 2 px from its rim and limb (deep interior);
- folds: candidates within 2.5 px of the fold line of single folds (30-120 deg, 4 orientations), both faces wide.
For every candidate, C = dtheta(k1) / dtheta(k2+, k2-) over a grid of skips k1 < k2+-, <= 2 k1 (symmetric and
asymmetric), against C_curv = (2 k1 + m) / (k2+ + k2- + m). Per C_curv bin: cylinder C P95 / max and fold C P5 / min;
per coefficient a: the margins of C_min_eff to both, and the largest C_curv with both margins positive.
Writes baseline_output/diag/fold_v2/f7/calibrate_veto.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from illum_baseline.detector import Params  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.fold import FoldChannel, FoldParams  # noqa: E402
from illum_baseline.geom.synthetic import F_PX, fold_scene, scene_objects  # noqa: E402
from illum_baseline.geom.twosided import two_sided_fit  # noqa: E402

TRI = Params(prefilter="trilateral")
K = Intrinsics.centered(F_PX, F_PX, 504, 378)
FP = FoldParams()
NOISE = 5e-4
R_S = 9
OUT = ROOT / "baseline_output" / "diag" / "fold_v2" / "f7"


def noisy(D, rel, seed):
    return 1.0 / ((1.0 / D) * (1.0 + rel * np.random.default_rng(seed).normal(size=D.shape)))


def at(a, x):
    return a[np.clip(np.rint(x[:, 1]).astype(int), 0, a.shape[0] - 1), np.clip(np.rint(x[:, 0]).astype(int), 0, a.shape[1] - 1)]


def skip_grid(k1):
    ks = range(k1 + 1, 2 * k1 + 1)
    return [(a, b) for a in ks for b in ks if a >= b]  # + side >= - side; by symmetry of the scenes this suffices


def ratios(ctx, x0, e1, k1):
    """[(C_curv, C)] for every skip pair, per candidate (k1 per candidate)."""
    out = []
    kw = dict(rms_max=None, fallback=False, m_min=FP.m_min, max_excluded=FP.max_excluded)
    r1 = two_sided_fit(ctx, x0, e1, k1, FP.m, FP.w_t, **kw)
    for kk in np.unique(k1):
        sel = k1 == kk
        for a, b in skip_grid(int(kk)):
            for kp, km in ((a, b), (b, a)):
                r2 = two_sided_fit(ctx, x0[sel], e1[sel], kp, FP.m, FP.w_t, k_m=km, **kw)
                ok = r1.valid[sel] & r2.valid
                C = r1.dtheta[sel][ok] / r2.dtheta[ok]
                cc = (2 * kk + FP.m) / (kp + km + FP.m)
                out += [(cc, c) for c in C]
    return np.array(out)


def cylinders():
    D, sid, nrm, names = scene_objects(radii=(0.5, 1.0, 2.0))
    ctx = GeometryContext(noisy(D, NOISE, 0), None, K, TRI)
    df = FoldChannel().run(ctx).candidates
    g = df[df["gate"]]
    x0, e1, k1 = g[["u0", "v0"]].to_numpy(), g[["e1_u", "e1_v"]].to_numpy(), g["k1"].to_numpy()
    et = np.stack([-e1[:, 1], e1[:, 0]], -1)
    deep = np.zeros(len(g), dtype=bool)
    for nm, i in names.items():
        if not nm.startswith("cyl"):
            continue
        dist = ndimage.distance_transform_edt(sid == i)
        dmin = np.full(len(g), np.inf)
        for sign in (1, -1):
            for s in range(0, int(2 * k1.max() + FP.m) + 1):
                for t in range(-FP.w_t, FP.w_t + 1):
                    m = s <= 2 * k1 + FP.m
                    d = at(dist, x0 + sign * s * e1 + t * et)
                    dmin = np.where(m & (s >= 0), np.minimum(dmin, d), dmin)
        deep |= dmin >= R_S + 2
    return ratios(ctx, x0[deep], e1[deep], k1[deep]), int(deep.sum())


def folds():
    out, n = [], 0
    for dth, phi, cvx in ((30, 0, True), (60, 135, True), (90, 30, False), (120, 90, False)):
        D, sid, ns, sky = fold_scene(dth, phi, cvx)
        ctx = GeometryContext(noisy(D, NOISE, 3), sky.astype(float), K, TRI)
        df = FoldChannel().run(ctx).candidates
        ph = math.radians(phi)
        dist = (df[["u0", "v0"]].to_numpy() - np.array([K.c_x, K.c_y])) @ np.array([-math.sin(ph), math.cos(ph)])
        line = (np.abs(dist) < 2.5) & df["valid"].to_numpy() & ~sky[df["v"], df["u"]]
        d = df[line]
        n += len(d)
        out.append(ratios(ctx, d[["u0", "v0"]].to_numpy(), d[["e1_u", "e1_v"]].to_numpy(), d["k1"].to_numpy()))
    return np.concatenate(out), n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cyl, n_cyl = cylinders()
    fol, n_fol = folds()
    bins = np.round(np.arange(0.55, 1.0001, 0.05), 2)
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        c = cyl[(cyl[:, 0] >= lo) & (cyl[:, 0] < hi), 1]
        f = fol[(fol[:, 0] >= lo) & (fol[:, 0] < hi), 1]
        table.append({"C_curv": f"[{lo:.2f}, {hi:.2f})", "n_cyl": int(c.size), "n_fold": int(f.size),
                      "cyl_P95": float(np.percentile(c, 95)) if c.size else None, "cyl_max": float(c.max()) if c.size else None,
                      "fold_P5": float(np.percentile(f, 5)) if f.size else None, "fold_min": float(f.min()) if f.size else None})
    coef = {}
    for a in (0.2, 0.3, 0.4, 0.5, 0.6):
        rows, limit = [], None
        for t in table:
            if t["n_cyl"] == 0 or t["n_fold"] == 0:
                continue
            cc = float(t["C_curv"][1:5]) + 0.025
            thr = 1 - a * (1 - cc)
            mc, mf = thr - t["cyl_max"], t["fold_min"] - thr
            rows.append({"C_curv": cc, "threshold": thr, "margin_cyl_max": mc, "margin_fold_min": mf})
            if mc > 0 and mf > 0:
                limit = cc
        coef[str(a)] = {"rows": rows, "largest_C_curv_with_both_margins": limit}
    res = {"n_cylinder_candidates": n_cyl, "n_fold_candidates": n_fol, "bins": table, "coef": coef}
    (OUT / "calibrate_veto.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"n": [n_cyl, n_fol], "bins": table}, indent=1))
    for a, v in coef.items():
        print(a, "limit", v["largest_C_curv_with_both_margins"],
              [(round(r["C_curv"], 3), round(r["margin_cyl_max"], 3), round(r["margin_fold_min"], 3)) for r in v["rows"]])


if __name__ == "__main__":
    main()
