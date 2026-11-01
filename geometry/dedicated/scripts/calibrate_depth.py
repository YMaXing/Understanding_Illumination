"""Calibrate the depth channel's data-derived constants (TASK_DEPTH_CHANNEL D2). Writes
illum_baseline/geom/data/depth_calibration.json; the DepthParams default sigma_w_max is set from it by hand.

1. lambda_hat_Delta_rho on synthetic noise: noisy planes (three slants, two seeds; relative noise 2e-4, 5e-4, 1e-3 on
   rho), the null grid (stride 4, each pixel's own n, x0 = the pixel centre) through the depth channel's two-sided fit
   at x0 (samples on Omega, no fallback, the contamination check) at k in {11, 13, 16}:
   lambda_median = median(z^2) / median(chi^2_1) (what the channel uses) and the Monte-Carlo inflation mean(z^2),
   z = Delta rho* / SE.
2. The real-image null: lambda_hat_Delta_rho per image at k = k_min (the channel's own per-image estimate).
3. sigma_w_max: P95 of sigma_w_c over unambiguous occlusions on the 10 real images. First pass with the prior
   (sigma_w_prior in the detrending) but without the smear bound (s_hi = inf, so the offset bound is off too; lobes
   searched to +-8 px): otherwise the P95 is capped by the prior itself. Unambiguous occlusion: passed D-3, both sides
   planar at k_c (m = 8, contamination check), r = Delta rho*/rho_hat_- > 0.2, significance
   |Delta rho*| / (SE sqrt(lambda_hat)) > 2 z, and the candidate holds the largest g_n on its normal line within
   +-lobe_half_len (not a crossing on the flank of a larger doublet). Sensitivity rows are recorded too.

Run from geometry/dedicated:  env -u PYTHONPATH uv run python scripts/calibrate_depth.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from illum_baseline.config import dense_fit_params, detector_params, load_config  # noqa: E402
from illum_baseline.detector import Params  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.candidates import depth_candidates, depth_field  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.depth import (CALIBRATION_PATH, DepthParams, depth_null, measure_jump,  # noqa: E402
                                       relative_jump, significance)
from illum_baseline.geom.sampling import bilinear  # noqa: E402
from illum_baseline.geom.synthetic import F_PX, rays  # noqa: E402

TRI = Params(prefilter="trilateral")
STAGE_A = ROOT / "baseline_output" / "stage_a"
CFG = ROOT / "configs" / "candidate_fold_v2.yaml"  # tri_veto4 detector (trilateral rho_f); geometry channels unused
STEMS = [f"taiwei_cui_{s}" for s in ("070", "075", "076", "079", "080", "082", "083", "084", "088", "090")]
OUT = ROOT / "baseline_output" / "diag" / "depth_v1" / "d2"


def noisy(D, rel, seed):
    return 1.0 / ((1.0 / D) * (1.0 + rel * np.random.default_rng(seed).normal(size=D.shape)))


def calibrate_lambda_synthetic(p: DepthParams) -> dict:
    K = Intrinsics.centered(F_PX, F_PX, 504, 378)
    r = rays(504, 378)
    planes = {}
    for name, n, d in (("slanted", [0.2, -0.3, 1.0], 3.0), ("fronto", [0.0, 0.0, 1.0], 4.0),
                       ("oblique", [-0.45, 0.25, 1.0], 2.5)):
        n = np.asarray(n, float) / np.linalg.norm(n)
        planes[name] = d / (r @ n)
    out = {}
    for k in (11, 13, 16):
        for noise in (2e-4, 5e-4, 1e-3):
            z2 = []
            for name, D in planes.items():
                for seed in (0, 1):
                    ctx = GeometryContext(noisy(D, noise, 10 * seed + 1), None, K, TRI)
                    z2.append(depth_null(ctx, p, depth_field(ctx, p), k=k).z2)
            z2 = np.concatenate(z2)
            rec = {"lambda_median": float(np.median(z2) / 0.45493642311957283), "lambda_mean": float(z2.mean()),
                   "n": int(z2.size)}
            out[f"k{k}_noise{noise:g}"] = rec
            print(f"  synthetic k={k} noise={noise:g}: median(z^2)/0.455 = {rec['lambda_median']:.2f}, "
                  f"mean(z^2) = {rec['lambda_mean']:.2f} ({z2.size} null windows)")
    return out


def real_context(stem: str, cfg) -> tuple[GeometryContext, dict]:
    meta = json.loads((STAGE_A / f"{stem}.json").read_text())
    z = np.load(STAGE_A / f"{stem}.npz")
    ctx = GeometryContext(z["D"].astype(np.float64), z["sky_prob"].astype(np.float64), Intrinsics.from_stage_a(meta),
                          detector_params(cfg), dense_fit_params(cfg))
    return ctx, meta


def dominant(f, x0: np.ndarray, n: np.ndarray, half_len: float, step: float = 0.25) -> np.ndarray:
    """(N,) bool: g_n at x0 is the largest g_n on x0 + s n, |s| <= half_len (samples outside the image ignored)."""
    s = np.arange(-half_len, half_len + step / 2, step)
    pts = x0[:, None, :] + s[None, :, None] * n[:, None, :]
    g, ok = bilinear(f.g_n, np.ones(f.g_n.shape, dtype=bool), pts[..., 0], pts[..., 1])
    g0, _ = bilinear(f.g_n, np.ones(f.g_n.shape, dtype=bool), x0[:, 0], x0[:, 1])
    gmax = np.nanmax(np.where(ok, g, -np.inf), axis=1)
    return g0 >= gmax - 1e-12 * np.abs(gmax)


def strong_table(ctx: GeometryContext, p: DepthParams, cal_pass: bool):
    """Candidates passing D-3 with their jump at k_c: a dict of per-candidate columns plus the null."""
    pp = dataclasses.replace(p, sigma_w_max=math.inf) if cal_pass else p
    f = depth_field(ctx, pp)
    c = depth_candidates(ctx, pp, field=f)
    nul = depth_null(ctx, pp, f)
    sel = c.passed
    x0, n, kc = c.x0[sel], c.n[sel], c.k_c[sel]
    res = measure_jump(ctx, pp, x0, n, kc, nul)
    r = relative_jump(res)
    sig = significance(res.d_rho, res.d_rho_se, nul.lambda_hat)
    planar = res.p.planar & res.m.planar
    dom = dominant(f, x0, n, p.lobe_half_len)
    return {"x0": x0, "n": n, "k_c": kc, "sigma_wc": c.sigma_wc[sel], "s_sep": c.s_sep[sel], "r": r, "sig": sig,
            "planar": planar, "dominant": dom, "d_rho": res.d_rho, "d_rho_se": res.d_rho_se,
            "rho_m": res.rho_m_gap, "rho_p": res.rho_p_gap}, c, nul


def calibrate_real(p: DepthParams) -> dict:
    cfg = load_config(CFG)
    out = {"per_image": {}, "pooled": {}}
    pools = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        ctx, _ = real_context(stem, cfg)
        t, c, nul = strong_table(ctx, p, cal_pass=True)
        with np.errstate(invalid="ignore"):
            base = t["planar"] & (t["r"] > 0.2)
            sets = {"main (r>0.2, sig>2z, dominant)": base & (t["sig"] > 2 * p.z) & t["dominant"],
                    "sig>z, dominant": base & (t["sig"] > p.z) & t["dominant"],
                    "sig>4z, dominant": base & (t["sig"] > 4 * p.z) & t["dominant"],
                    "sig>2z, not dominance-filtered": base & (t["sig"] > 2 * p.z)}
        rec = {"lambda_raw": nul.lambda_raw, "lambda_hat": nul.lambda_hat, "eps_hat": nul.eps_hat,
               "q_null": nul.q_null, "n_null": int(nul.z2.size), "n_passed_cal": int(c.counts["n_passed"]),
               "n_main": int(sets["main (r>0.2, sig>2z, dominant)"].sum())}
        sw = t["sigma_wc"][sets["main (r>0.2, sig>2z, dominant)"]]
        rec["sigma_wc_P50_P95"] = [float(np.percentile(sw, 50)), float(np.percentile(sw, 95))] if sw.size else None
        out["per_image"][stem[-3:]] = rec
        for name, m in sets.items():
            pools.setdefault(name, []).append(t["sigma_wc"][m])
        np.savez_compressed(OUT / f"cal_{stem[-3:]}.npz", **t)
        print(f"  {stem[-3:]}: lambda_hat {nul.lambda_hat:.1f} ({nul.z2.size} null), D-3 passed (cal) "
              f"{c.counts['n_passed']}, unambiguous {rec['n_main']}, sigma_wc P50/P95 {rec['sigma_wc_P50_P95']}")
    for name, arrs in pools.items():
        a = np.concatenate(arrs)
        out["pooled"][name] = {"n": int(a.size), "P50": float(np.percentile(a, 50)), "P90": float(np.percentile(a, 90)),
                               "P95": float(np.percentile(a, 95)), "P99": float(np.percentile(a, 99)),
                               "share_at_lobe_limit": float((a > 7.5).mean())}
        print(f"  pooled [{name}]: n {a.size}, sigma_wc P50 {np.percentile(a, 50):.2f} P90 {np.percentile(a, 90):.2f} "
              f"P95 {np.percentile(a, 95):.2f} P99 {np.percentile(a, 99):.2f}")
    out["sigma_w_max"] = out["pooled"]["main (r>0.2, sig>2z, dominant)"]["P95"]
    return out


if __name__ == "__main__":
    t0 = time.time()
    p = DepthParams()
    res = {"params": {"sigma": p.sigma, "sigma_w_prior": p.sigma_w_prior, "k_g": p.k_g, "k_min": p.k_min, "m": p.m,
                      "w_t": p.w_t, "z": p.z, "gap_mode": p.gap_mode, "disparity": p.disparity}}
    print("sigma_w_max and the real null:")
    res["real"] = calibrate_real(p)
    res["sigma_w_max"] = res["real"]["sigma_w_max"]
    print("lambda_hat_Delta_rho on synthetic noise:")
    res["lambda_synthetic"] = calibrate_lambda_synthetic(p)
    CALIBRATION_PATH.parent.mkdir(exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(res, indent=1))
    print(f"sigma_w_max = {res['sigma_w_max']:.2f} px; {time.time() - t0:.0f} s -> {CALIBRATION_PATH}")
