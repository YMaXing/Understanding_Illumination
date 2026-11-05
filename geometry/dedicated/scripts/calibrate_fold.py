"""Calibrate the fold channel's two data-derived constants on synthetic scenes (TASK_FOLD_CHANNEL F2 decisions 2, 4).

1. b_tri(k, m): the rms(residual)/alpha that the trilateral filter's bias alone puts into a one-sided window at skip k
   and length m next to a feature. Noise-free single folds (15-120 deg, 4 orientations, convex and concave) and single
   occlusion steps (depth ratio 1.1-3, three tilt pairs, 2 orientations); windows at the channel's own candidates
   (subpixel x0, structure-tensor e1), no contamination check, no fallback. Only windows whose samples (all four
   bilinear corners) lie on one surface and at least R_s + 1 px from the image border count: the bias next to a
   feature, not windows straddling it or the odd-reflect padding mirroring the feature back near the border.
   b_tri(k, m) = max over those windows for k <= K_MAX, then made non-increasing in k (constant beyond K_MAX). Written to illum_baseline/geom/data/b_tri.json.
2. lambda_n: the noise-only inflation of T (Monte-Carlo E[T]/2 over the null sample) on noisy planes, per k_min arm and
   noise level. Written to illum_baseline/geom/data/lambda_n.json; the FoldParams default is set from it by hand.

Run from geometry/dedicated:  env -u PYTHONPATH uv run python scripts/calibrate_fold.py
TASK_GEOM_V2 W1b: `--add-m 5 6 7` computes b_tri for those window lengths in the same way and adds them to b_tri.json,
leaving the existing entries (m = 8, 4) and lambda_n untouched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from illum_baseline.detector import Params  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.candidates import fold_candidates  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.fold import FoldChannel, FoldParams  # noqa: E402
from illum_baseline.geom.synthetic import F_PX, fold_scene, rays, step_scene  # noqa: E402
from illum_baseline.geom.twosided import B_TRI_PATH, two_sided_fit  # noqa: E402

TRI = Params(prefilter="trilateral")
K = Intrinsics.centered(F_PX, F_PX, 504, 378)
FP = FoldParams()
K_MAX = 16  # largest k the pipeline uses: k2 = 2 * (ceil(12 / 2) + 2)
R_S = 9      # trilateral support


def reference_scenes():
    for dth in (15, 30, 60, 90, 120):
        for phi in (0, 30, 90, 135):
            for convex in (True, False):
                D, sid, _, sky = fold_scene(dth, phi, convex)
                yield f"fold {dth} phi {phi} {'cvx' if convex else 'ccv'}", D, sky.astype(float), sid
    for ratio in (1.1, 1.5, 3.0):
        for tilts in ((20, -20), (0, 40), (-30, 10)):
            for phi in (0, 60):
                D, sid, _ = step_scene(ratio, *tilts, phi)
                yield f"step {ratio} {tilts} phi {phi}", D, None, sid


def single_surface(sid: np.ndarray, x0: np.ndarray, e1: np.ndarray, k: int, m: int, sign: int) -> np.ndarray:
    """(N,) bool: every bilinear corner of every sample of the window lies on one surface (> 0), R_s + 1 px or more
    inside the image border."""
    H, W = sid.shape
    et = np.stack([-e1[:, 1], e1[:, 0]], -1)
    s = sign * (k + np.arange(m + 1.0))
    t = np.arange(-FP.w_t, FP.w_t + 1.0)
    pts = (x0[:, None, None, :] + s[None, :, None, None] * e1[:, None, None, :]
           + t[None, None, :, None] * et[:, None, None, :]).reshape(len(x0), -1, 2)
    u0, v0 = np.floor(pts[..., 0]).astype(int), np.floor(pts[..., 1]).astype(int)
    b = R_S + 1
    inside = ((u0 >= b) & (u0 <= W - 2 - b) & (v0 >= b) & (v0 <= H - 2 - b)).all(1)
    u0, v0 = np.clip(u0, 0, W - 2), np.clip(v0, 0, H - 2)
    corners = np.stack([sid[v0, u0], sid[v0, u0 + 1], sid[v0 + 1, u0], sid[v0 + 1, u0 + 1]], -1).reshape(len(x0), -1)
    return inside & (corners == corners[:, :1]).all(1) & (corners[:, 0] > 0)


def calibrate_b_tri(ms=(FP.m, FP.m // 2)) -> dict:
    b = {m: np.zeros(K_MAX + 1) for m in ms}
    worst = {m: [""] * (K_MAX + 1) for m in b}
    n_scenes = 0
    for name, D, sky, sid in reference_scenes():
        ctx = GeometryContext(D, sky, K, TRI)
        c = fold_candidates(ctx, FP.cand_sigma_d, FP.cand_sigma_T, FP.cand_rate_factor)
        for m in b:
            for k in range(K_MAX + 1):
                r = two_sided_fit(ctx, c.x0, c.e1, k, m, FP.w_t, rms_max=None, fallback=False,
                                  max_excluded=FP.max_excluded)
                okp = r.p.planar & single_surface(sid, c.x0, c.e1, k, m, +1)
                okm = r.m.planar & single_surface(sid, c.x0, c.e1, k, m, -1)
                rr = np.concatenate([r.p.rms_rel[okp], r.m.rms_rel[okm]])
                if rr.size and rr.max() > b[m][k]:
                    b[m][k], worst[m][k] = float(rr.max()), name
        n_scenes += 1
    env = {m: np.maximum.accumulate(v[::-1])[::-1] for m, v in b.items()}
    return {"tri": [TRI.tri_sigma_s, TRI.tri_sigma_r, TRI.tri_sigma_g], "w_t": FP.w_t, "k_max": K_MAX,
            "n_scenes": n_scenes, "b": {str(m): env[m].tolist() for m in env},
            "raw_max": {str(m): b[m].tolist() for m in b}, "worst_scene": {str(m): worst[m] for m in worst}}


def noisy(D, rel, seed):
    return 1.0 / ((1.0 / D) * (1.0 + rel * np.random.default_rng(seed).normal(size=D.shape)))


def calibrate_lambda_n() -> dict:
    r = rays(504, 378)
    planes = {}
    for name, n, d in (("slanted", [0.2, -0.3, 1.0], 3.0), ("fronto", [0.0, 0.0, 1.0], 4.0),
                       ("oblique", [-0.45, 0.25, 1.0], 2.5)):
        n = np.asarray(n, float) / np.linalg.norm(n)
        planes[name] = d / (r @ n)
    out = {}
    for k_min in (6, 11):
        for noise in (2e-4, 5e-4, 1e-3):
            T = []
            for name, D in planes.items():
                for seed in (0, 1):
                    res = FoldChannel().run(GeometryContext(noisy(D, noise, 10 * seed + 1), None, K, TRI),
                                            {"k_min": k_min})
                    t = res.tables["null"]
                    T.append(t["T"][t["valid"]].to_numpy())
            T = np.concatenate(T)
            T = T[np.isfinite(T)]
            out[f"k{k_min}_noise{noise:g}"] = {"lambda_mean": float(T.mean() / 2),
                                              "lambda_median": float(np.median(T) / (2 * np.log(2))), "n": int(T.size)}
            print(f"  lambda_n k_min={k_min} noise={noise:g}: E[T]/2 = {T.mean() / 2:.1f}, "
                  f"median/(2 ln 2) = {np.median(T) / (2 * np.log(2)):.1f} ({T.size} null windows)")
    for k_min in (6, 11):
        v = [out[f"k{k_min}_noise{n:g}"]["lambda_mean"] for n in (2e-4, 5e-4, 1e-3)]
        out[f"k{k_min}"] = float(np.median(v))
    return out


if __name__ == "__main__":
    t0 = time.time()
    if "--add-m" in sys.argv:
        ms = tuple(int(a) for a in sys.argv[sys.argv.index("--add-m") + 1:])
        old = json.loads(B_TRI_PATH.read_text())
        new = calibrate_b_tri(ms)
        assert new["tri"] == old["tri"] and new["w_t"] == old["w_t"] and new["k_max"] == old["k_max"]
        for key in ("b", "raw_max", "worst_scene"):
            for m in ms:
                if str(m) in old[key]:
                    raise SystemExit(f"b_tri.json already has m = {m}")
                old[key][str(m)] = new[key][str(m)]
        B_TRI_PATH.write_text(json.dumps(old, indent=1))
        for m in ms:
            print(f"b_tri m={m}: " + " ".join(f"k{k}:{v:.1e}" for k, v in enumerate(old["b"][str(m)]) if 4 <= k <= 14))
        print(f"b_tri: added m = {ms}, {time.time() - t0:.0f} s -> {B_TRI_PATH}")
        sys.exit(0)
    tab = calibrate_b_tri()
    B_TRI_PATH.parent.mkdir(exist_ok=True)
    B_TRI_PATH.write_text(json.dumps(tab, indent=1))
    for m in tab["b"]:
        print(f"b_tri m={m}: " + " ".join(f"k{k}:{v:.1e}" for k, v in enumerate(tab["b"][m]) if 4 <= k <= 14))
        print(f"  worst: " + "; ".join(f"k{k} {tab['worst_scene'][m][k]}" for k in (6, 8, 10, 11)))
    print(f"b_tri: {tab['n_scenes']} scenes, {time.time() - t0:.0f} s -> {B_TRI_PATH}")
    from illum_baseline.geom import twosided
    twosided._b_tri_table.cache_clear()
    lam = calibrate_lambda_n()
    (B_TRI_PATH.parent / "lambda_n.json").write_text(json.dumps(lam, indent=1))
    print(f"lambda_n: k_min 6 -> {lam['k6']:.1f}, k_min 11 -> {lam['k11']:.1f}; total {time.time() - t0:.0f} s")
