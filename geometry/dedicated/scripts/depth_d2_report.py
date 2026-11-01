"""TASK_DEPTH_CHANNEL D2 report: per-image candidate counts per stage, the 070 floor ablation (rho_f vs raw rho), and
the DA3-vs-RGB edge alignment diagnostic. Writes baseline_output/diag/depth_v1/d2/{d2_report.json, alignment.parquet}.

Alignment (diagnostic only; changes nothing): at strong depth candidates (passed D-3 with the calibrated parameters,
both sides planar at k_c, r > 0.2, significance > 2 z) and along the fold ROIs (the fold line targets of
scripts/fold_null_rois.py, plus the D0 fold ROIs, marked provisional), search s in [-8, 8] px along n for the maximum
|d Y / dn|, Y the luminance of the upright sRGB image area-resized to network resolution exactly as Stage A feeds DA3
(linearized sRGB, Rec. 709 weights; Gaussian derivative sigma = 1 px; parabola-refined). The signed offset is
s* along n (depth candidates: n toward the near side; fold ROIs: the polyline normal with n_u > 0, or (0, 1)), and
(dx, dy) = s* n. SE is the naive std/sqrt(N) (samples along a contour are correlated: treat it as a lower bound);
the per-image means give a second, conservative SE.

Run from geometry/dedicated:  env -u PYTHONPATH uv run python scripts/depth_d2_report.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_depth import CFG, OUT, STAGE_A, STEMS, real_context  # noqa: E402
from fold_null_rois import TARGETS, _poly_samples  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.detector import gaussian_gradient  # noqa: E402
from illum_baseline.geom.candidates import DEPTH_CAND_STAGES, depth_candidates, depth_field  # noqa: E402
from illum_baseline.geom.depth import (DepthParams, depth_null, measure_jump, relative_jump,  # noqa: E402
                                       significance)
from illum_baseline.geom.sampling import bilinear  # noqa: E402

D0_ROIS = ROOT / "baseline_output" / "diag" / "depth_v1" / "d0" / "rois.json"
FLOOR_ROI = ROOT / "baseline_output" / "diag" / "070" / "roi_mask.npy"
STAGE_KEYS = ("n_cross", "n_floor", "n_cand", "n_polarity", "n_height", "n_sharpness", "n_centring", "n_passed")


def luminance(stem: str, meta: dict) -> np.ndarray:
    """Y at network resolution: the upright sRGB PNG area-resized as in Stage A (dither none), linearized."""
    bgr = cv2.imread(str(STAGE_A / "srgb" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    W, H = meta["network"]["W_n"], meta["network"]["H_n"]
    net = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float64) / 255.0
    lin = np.where(net <= 0.04045, net / 12.92, ((net + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def rgb_offset(Y: np.ndarray, x0: np.ndarray, n: np.ndarray, half: float = 8.0, step: float = 0.25) -> np.ndarray:
    """(N,) s* in [-half, half]: argmax over s of |grad Y . n| at x0 + s n (parabola-refined); NaN if a sample
    leaves the image."""
    gu, gv = gaussian_gradient(Y, 1.0)
    s = np.arange(-half, half + step / 2, step)
    pts = x0[:, None, :] + s[None, :, None] * n[:, None, :]
    ok_all = np.ones(Y.shape, dtype=bool)
    a, ok1 = bilinear(gu, ok_all, pts[..., 0], pts[..., 1])
    b, ok2 = bilinear(gv, ok_all, pts[..., 0], pts[..., 1])
    d = np.abs(a * n[:, None, 0] + b * n[:, None, 1])
    j = np.argmax(np.where(ok1 & ok2, d, -np.inf), axis=1)
    r = np.arange(len(x0))
    jm, jp = np.clip(j - 1, 0, len(s) - 1), np.clip(j + 1, 0, len(s) - 1)
    y0, ym, yp = d[r, j], d[r, jm], d[r, jp]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = ym - 2 * y0 + yp
        dl = np.where((j > 0) & (j < len(s) - 1) & (c < 0), 0.5 * (ym - yp) / c, 0.0)
    out = s[j] + np.clip(dl, -0.5, 0.5) * step
    return np.where((ok1 & ok2).all(1), out, np.nan)


def stats(a: np.ndarray) -> dict:
    a = a[np.isfinite(a)]
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "se": float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1
            else float("nan"), "P10": float(np.percentile(a, 10)), "P50": float(np.percentile(a, 50)),
            "P90": float(np.percentile(a, 90))}


def fold_rois(stem: str) -> list[tuple[str, np.ndarray, bool]]:
    """(name, polyline, provisional) for the fold-class line targets of this image."""
    out = [(k, np.asarray(v[2], float), False) for k, v in TARGETS.items() if stem.endswith(v[0])]
    d0 = json.loads(D0_ROIS.read_text())["polylines"]
    out += [(k, np.asarray(v["polyline"], float), True) for k, v in d0.items()
            if v["class"] == "fold" and stem.endswith(v["stem"])]
    return out


def poly_points(poly: np.ndarray, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    pts = _poly_samples(poly, step)
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.stack([-t[:, 1], t[:, 0]], -1)
    flip = (n[:, 0] < 0) | ((n[:, 0] == 0) & (n[:, 1] < 0))
    n[flip] *= -1
    return pts, n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(CFG)
    p = DepthParams()
    floor = np.load(FLOOR_ROI).astype(bool)
    rows, abl, al = [], {}, []
    for stem in STEMS:
        ctx, meta = real_context(stem, cfg)
        t0 = time.perf_counter()
        f = depth_field(ctx, p)
        c = depth_candidates(ctx, p, field=f)
        t_c = time.perf_counter() - t0
        nul = depth_null(ctx, p, f)
        row = {"stem": stem[-3:], **{k: c.counts[k] for k in STAGE_KEYS}, "valid_frac": float(f.valid.mean()),
               "valid_frac_omega": float(f.valid.sum() / ctx.omega.sum()), "s_q": f.s_q, "sigma_g": f.sigma_g,
               "lambda_hat": nul.lambda_hat, "t_cand_s": t_c, "normals_built": "_normals" in ctx.__dict__}
        sel = c.passed
        res = measure_jump(ctx, p, c.x0[sel], c.n[sel], c.k_c[sel], nul)
        r = relative_jump(res)
        sig = significance(res.d_rho, res.d_rho_se, nul.lambda_hat)
        with np.errstate(invalid="ignore"):
            strong = res.p.planar & res.m.planar & (r > 0.2) & (sig > 2 * p.z)
        row["n_strong"] = int(strong.sum())
        rows.append(row)

        # 070 floor ablation: rho_f vs raw rho, candidates with x0 in the floor ROI and over the whole image
        if stem.endswith("070"):
            for disp in ("rho_f", "rho"):
                pd_ = dataclasses.replace(p, disparity=disp)
                cc = c if disp == "rho_f" else depth_candidates(ctx, pd_)
                ij = np.clip(cc.ij, 0, [ctx.shape[1] - 1, ctx.shape[0] - 1])
                inroi = floor[ij[:, 1], ij[:, 0]]
                st = {"n_cand": int(inroi.sum())}
                alive = inroi.copy()
                for stage in DEPTH_CAND_STAGES:
                    alive &= cc.reason != stage
                    st[f"after_{stage}"] = int(alive.sum())
                abl[disp] = {"floor_roi": st, "whole_image": {k: cc.counts[k] for k in STAGE_KEYS},
                             "s_q": cc.field.s_q, "sigma_g": cc.field.sigma_g,
                             "valid_frac_floor": float(cc.field.valid[floor].mean())}

        # alignment
        Y = luminance(stem, meta)
        xs, ns = c.x0[sel][strong], c.n[sel][strong]
        off = rgb_offset(Y, xs, ns)
        ang = np.degrees(np.arctan2(ns[:, 1], ns[:, 0]))
        al.append(pd.DataFrame({"stem": stem[-3:], "group": "depth_strong", "roi": "", "provisional": False,
                                "u": xs[:, 0], "v": xs[:, 1], "n_u": ns[:, 0], "n_v": ns[:, 1], "angle": ang,
                                "s": off, "dx": off * ns[:, 0], "dy": off * ns[:, 1]}))
        for name, poly, prov in fold_rois(stem):
            pts, nn = poly_points(poly)
            o = rgb_offset(Y, pts, nn)
            al.append(pd.DataFrame({"stem": stem[-3:], "group": "fold_roi", "roi": name, "provisional": prov,
                                    "u": pts[:, 0], "v": pts[:, 1], "n_u": nn[:, 0], "n_v": nn[:, 1],
                                    "angle": np.degrees(np.arctan2(nn[:, 1], nn[:, 0])), "s": o, "dx": o * nn[:, 0],
                                    "dy": o * nn[:, 1]}))
        print(f"{stem[-3:]}: " + " > ".join(str(row[k]) for k in STAGE_KEYS) + f" | strong {row['n_strong']} | valid "
              f"{row['valid_frac_omega']:.3f} of Omega | s_q {f.s_q:.2e} | lambda {nul.lambda_hat:.0f} | {t_c:.2f} s")

    df = pd.DataFrame(rows)
    A = pd.concat(al, ignore_index=True)
    A.to_parquet(OUT / "alignment.parquet")
    rep = {"counts": df.to_dict(orient="records"), "ablation_070": abl, "alignment": {}}
    for g, sub in A.groupby("group"):
        bins = ((sub["angle"].to_numpy() + 22.5) // 45 % 8 * 45).astype(int)  # bin centres 0, 45, ..., 315 deg
        per_img = {s: {k: stats(x[k].to_numpy()) for k in ("s", "dx", "dy")} for s, x in sub.groupby("stem")}
        img_means = np.array([v["s"]["mean"] for v in per_img.values() if v["s"]["n"]])
        rep["alignment"][g] = {
            "all": {k: stats(sub[k].to_numpy()) for k in ("s", "dx", "dy")},
            "se_of_image_means_s": float(img_means.std(ddof=1) / math.sqrt(len(img_means))) if len(img_means) > 1
            else float("nan"),
            "per_image": per_img,
            "per_orientation_bin": {int(b): {k: stats(sub.loc[bins == b, k].to_numpy()) for k in ("s", "dx", "dy")}
                                    for b in sorted(set(bins))}}
        if g == "fold_roi":
            rep["alignment"][g]["per_roi"] = {f"{r}{' (provisional)' if pv else ''}": stats(x["s"].to_numpy())
                                              for (r, pv), x in sub.groupby(["roi", "provisional"])}
    (OUT / "d2_report.json").write_text(json.dumps(rep, indent=1))

    pd.set_option("display.width", 250)
    print("\nper-image counts:\n" + df.round(4).to_string(index=False))
    print("\n070 floor ablation:", json.dumps(abl, indent=1))
    for g, v in rep["alignment"].items():
        print(f"\nalignment [{g}] all: " + "; ".join(f"{k} " + ", ".join(f"{a} {b:.3f}" if isinstance(b, float) else
                                                                     f"{a} {b}" for a, b in v["all"][k].items())
                                                 for k in ("s", "dx", "dy")))
        print(f"  SE over image means (s): {v['se_of_image_means_s']:.3f}")
        print("  per image (s mean/P50, n): " + "; ".join(f"{s} {x['s'].get('mean', float('nan')):+.2f}/"
                                                      f"{x['s'].get('P50', float('nan')):+.2f} ({x['s']['n']})"
                                                      for s, x in v["per_image"].items()))
        print("  per orientation bin of n (s mean, P10/P50/P90, n): " + "; ".join(
            f"{b}: {x['s']['mean']:+.2f} [{x['s']['P10']:+.1f}/{x['s']['P50']:+.1f}/{x['s']['P90']:+.1f}] ({x['s']['n']})"
            for b, x in v["per_orientation_bin"].items() if x["s"]["n"]))
        if "per_roi" in v:
            print("  per ROI (s mean, P50, n): " + "; ".join(f"{r}: {x['mean']:+.2f}/{x['P50']:+.2f} ({x['n']})"
                                                         for r, x in v["per_roi"].items() if x["n"]))


if __name__ == "__main__":
    main()
