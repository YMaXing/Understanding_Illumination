"""Pre-D6 checks (TASK_DEPTH_CHANNEL, after D5; diagnostics only, no channel changes).

a. 079 box bottom edge (box-plinth contact) and 075 kiosk base: at 4 points each, diag-depth plots plus
   Delta rho* at x0 and min-span, planes_cross, r, Delta Z (run depth_v1_add, the additive default), the fold
   channel's Delta theta at the same place (nearest fold_v2 candidate, and the fold measure of a virtual candidate at
   x0 with the depth n), and a model selection on the rho_f samples inside +-k_c (|t| <= w_t, samples within 2 px of
   the transition excluded): a step (two independent planes) vs a continuous two-plane kink, each with its best
   transition offset in [-2, 2] px; BIC = n ln(RSS/n) + p ln n.
b. (in scripts/depth_d5_report.py: 080 side face edge R evaluated as step.)
c. 070 sky boundary: band pixels with w_depth >= 0.5 classified as (iii) corner / junction, (ii) gap between accepted
   candidates (no accepted candidate within 2.5 px of the nearest sky-boundary pixel), (i) beyond the band (the
   boundary there is covered, the pixel is > b px from the nearest accepted edge); counts and an overlay.
Writes baseline_output/diag/depth_v1/pre_d6/{pre_d6.json, sky_band_070.png}; diag-depth plots in
stage_b/depth_v1_add/diag_depth/.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_depth import real_context  # noqa: E402
from depth_rois import sky_band_070  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.diag_depth import diag_depth  # noqa: E402
from illum_baseline.geom.fold import FoldChannel, FoldParams  # noqa: E402
from illum_baseline.geom.sampling import bilinear  # noqa: E402
from illum_baseline.stage_b import load_upright  # noqa: E402

SB = ROOT / "baseline_output" / "stage_b"
RUN = "depth_v1_add"
OUT = ROOT / "baseline_output" / "diag" / "depth_v1" / "pre_d6"
# (stem, component of accepted depth candidates in depth_v1_add, target points (u, v) to snap to it)
PROBES = {
    "079 box bottom edge": ("079", [(130, 285), (170, 270), (210, 258), (250, 245)]),
    "075 kiosk base": ("075", [(244, 276), (244, 292), (244, 303), (348, 280)]),
}


def model_selection(ctx, x0, n, k, w_t, excl=2.0):
    s = np.arange(-k, k + 1e-9, 0.5)
    t = np.arange(-w_t, w_t + 1, dtype=float)
    et = np.array([-n[1], n[0]])
    S, T = np.meshgrid(s, t, indexing="ij")
    pts = x0[None, None] + S[..., None] * n + T[..., None] * et
    val, ok = bilinear(ctx.rho_f, ctx.omega, pts[..., 0], pts[..., 1])
    S, T, val = S[ok], T[ok], val[ok]
    best = {}
    for name in ("step", "kink"):
        rec = (np.inf, None, None)
        for s0 in np.arange(-2, 2.001, 0.25):
            m = np.abs(S - s0) >= excl
            ss, tt, vv = S[m] - s0, T[m], val[m]
            if name == "step":
                A = np.zeros((len(ss), 6))
                lo = ss < 0
                A[lo, 0], A[lo, 1], A[lo, 2] = 1, ss[lo], tt[lo]
                A[~lo, 3], A[~lo, 4], A[~lo, 5] = 1, ss[~lo], tt[~lo]
            else:
                A = np.stack([np.ones_like(ss), tt, np.minimum(ss, 0), np.maximum(ss, 0)], -1)
            coef, *_ = np.linalg.lstsq(A, vv, rcond=None)
            rss = float(((A @ coef - vv) ** 2).sum())
            if rss < rec[0]:
                rec = (rss, s0, coef)
        rss, s0, coef = rec
        nn = int((np.abs(S - s0) >= excl).sum())
        p_ = (6 if name == "step" else 4) + 1
        out = {"rss": rss, "rms_rel": math.sqrt(rss / nn) / float(np.median(val)), "s0": float(s0), "n": nn,
               "bic": nn * math.log(max(rss, 1e-300) / nn) + p_ * math.log(nn)}
        if name == "step":
            far, near = coef[0], coef[3]  # values at s0 (t = 0): side - (s < s0) is the far side
            out["jump_rel"] = float((near - far) / far)
        else:
            out["slope_m"], out["slope_p"] = float(coef[2]), float(coef[3])
        best[name] = out
    d_bic = best["kink"]["bic"] - best["step"]["bic"]
    verdict = ("step" if d_bic > 10 and best["step"]["jump_rel"] > 0.02 else
               "kink" if d_bic < -10 or abs(best["step"]["jump_rel"]) < 0.01 else "ambiguous")
    return {**{f"{m}_{k_}": v for m in best for k_, v in best[m].items()}, "dBIC_kink_minus_step": d_bic,
            "model": verdict}


def check_a(cfg) -> list[dict]:
    rows = []
    fold_cfg = load_config(ROOT / "configs" / "candidate_fold_v2.yaml")
    for name, (s, targets) in PROBES.items():
        stem = f"taiwei_cui_{s}"
        cand = pd.read_parquet(SB / RUN / f"{stem}_depth_candidates.parquet")
        acc = cand[cand["accepted"]].reset_index(drop=True)
        fold = pd.read_parquet(SB / "fold_v2" / f"{stem}_fold_candidates.parquet")
        ctx, _ = real_context(stem, fold_cfg)  # tri_veto4 detector + the dense-fit settings (normals for the fold)
        fp = FoldParams.from_dict(dataclasses.asdict(fold_cfg.geometry.fold))
        fres = FoldChannel().run(ctx, fp)
        from fold_v2_f6 import virtual
        for (u, v) in targets:
            i = int(np.argmin((acc["u0"] - u) ** 2 + (acc["v0"] - v) ** 2))
            r = acc.iloc[i]
            x0, n = np.array([r.u0, r.v0]), np.array([r.n_u, r.n_v])
            png = diag_depth(cfg, SB / RUN, stem, r.u0, r.v0)
            dfold = np.hypot(fold["u0"] - r.u0, fold["v0"] - r.v0)
            j = int(np.argmin(dfold))
            e = n if n[0] > 0 or (n[0] == 0 and n[1] > 0) else -n
            virt = virtual(ctx, fres, fp, x0, e)
            ms = model_selection(ctx, x0, n, int(r.k_c), 4)
            rows.append({"probe": name, "stem": s, "u0": float(r.u0), "v0": float(r.v0), "n": [float(n[0]), float(n[1])],
                         "d_rho_at_x0": float(r.d_rho), "d_rho_min_span": float(r.d_rho_min_span),
                         "planes_cross": bool(r.planes_cross), "r": float(r.r), "dZ_m": float(r.dZ),
                         "fit_mode": r.fit_mode, "signif_add": float(r.signif_add), "signif_mult": float(r.signif),
                         "fold_nearest_px": float(dfold[j]), "fold_nearest_dtheta_deg": math.degrees(fold["dtheta"][j]),
                         "fold_nearest_accepted": bool(fold["accepted"][j]), "fold_nearest_reason": fold["reason"][j],
                         "fold_virtual_dtheta_k1_deg": virt["dtheta_k1"], "fold_virtual_dtheta_k2_deg": virt["dtheta_k2"],
                         **ms, "diag_depth_png": str(png)})
    return rows


def check_c() -> tuple[dict, Path]:
    cfg = load_config(SB / RUN / "config_resolved.yaml")
    a, meta = load_upright(cfg, SB / RUN, "taiwei_cui_070")
    w = a["depth_w_net"].astype(float)
    sky = np.load(SB / "tri_veto4" / "taiwei_cui_070.npz")["sky_mask"]
    band = sky_band_070()
    H, W = band.shape
    bnd = sky & ~ndimage.binary_erosion(sky, structure=np.ones((3, 3), bool), border_value=1)
    bnd[:1, :] = bnd[-1:, :] = False
    bnd[:, :1] = bnd[:, -1:] = False
    bv, bu = np.nonzero(bnd)
    B = np.stack([bu, bv], -1).astype(float)
    cand = pd.read_parquet(SB / RUN / "taiwei_cui_070_depth_candidates.parquet")
    acc = cand[cand["accepted"]]
    X = acc[["u0", "v0"]].to_numpy()
    tX = cKDTree(X)
    nonsky = acc[~acc["sky_edge"]][["u0", "v0"]].to_numpy()
    tN = cKDTree(nonsky) if len(nonsky) else None
    tB = cKDTree(B)
    uv, uu = np.nonzero(band & (w >= 0.5))
    P = np.stack([uu, uv], -1).astype(float)
    _, ib = tB.query(P)
    b0 = B[ib]
    # local boundary linearity at b0 (PCA over boundary pixels within 4 px)
    lin = np.empty(len(P))
    for k_, q in enumerate(b0):
        nb = B[tB.query_ball_point(q, 4.0)]
        if len(nb) < 3:
            lin[k_] = 1.0
            continue
        ev = np.linalg.eigvalsh(np.cov((nb - nb.mean(0)).T))
        lin[k_] = ev[0] / max(ev[1], 1e-12)
    d_acc_b0, _ = tX.query(b0)
    d_acc_p, ia = tX.query(P)
    band_c = acc["band"].to_numpy()[ia]
    junction = tN.query(b0)[0] <= 4.0 if tN is not None else np.zeros(len(P), bool)
    corner = (lin > 0.15) | junction
    gap = ~corner & (d_acc_b0 > 2.5)
    beyond = ~corner & ~gap
    counts = {"band_px": int(band.sum()), "w_ge_0.5": int(len(P)), "share_w_ge_0.5": float(len(P) / band.sum()),
              "iii_corner_junction": int(corner.sum()), "of_which_junction": int(junction.sum()),
              "ii_gap": int(gap.sum()), "i_beyond_band": int(beyond.sum()),
              "i_beyond_band_dist_P50": float(np.median(d_acc_p[beyond])) if beyond.any() else float("nan"),
              "i_band_radius_P50": float(np.median(band_c[beyond])) if beyond.any() else float("nan")}
    # overlay
    sa = cfg.out_dir / meta["provenance"]["stage_a_json"]
    img = cv2.resize(cv2.imread(str(sa.parent / "srgb" / "taiwei_cui_070.png")), (W, H), interpolation=cv2.INTER_AREA)
    f = 3
    big = cv2.resize(img, (W * f, H * f), interpolation=cv2.INTER_NEAREST) // 2
    bandk = np.kron(band, np.ones((f, f), bool))
    big[bandk & (np.kron(w, np.ones((f, f))) < 0.5)] = (90, 90, 90)
    for mask, col in ((beyond, (0, 255, 255)), (gap, (0, 0, 255)), (corner, (255, 0, 255))):
        for (x, y) in P[mask].astype(int):
            big[y * f:(y + 1) * f, x * f:(x + 1) * f] = col
    for (x, y) in X:
        cv2.circle(big, (int(x * f), int(y * f)), 1, (0, 255, 0), -1)
    cv2.putText(big, "gray: band w<0.5 | yellow: (i) beyond band | red: (ii) gap | magenta: (iii) corner/junction | "
                "green: accepted", (5, 14), 0, 0.42, (255, 255, 255), 1)
    out = OUT / "sky_band_070.png"
    cv2.imwrite(str(out), big)
    return counts, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "candidate_depth_only.yaml")
    rows = check_a(cfg)
    counts, png = check_c()
    (OUT / "pre_d6.json").write_text(json.dumps({"a_probes": rows, "c_sky_band_070": counts, "c_overlay": str(png)},
                                                indent=1, default=float))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    df = pd.DataFrame(rows)
    print(df[["probe", "u0", "v0", "d_rho_at_x0", "d_rho_min_span", "planes_cross", "r", "dZ_m", "fit_mode",
              "fold_nearest_px", "fold_nearest_dtheta_deg", "fold_nearest_accepted", "fold_virtual_dtheta_k1_deg",
              "fold_virtual_dtheta_k2_deg"]].round(4).to_string(index=False))
    print(df[["probe", "u0", "v0", "step_rms_rel", "kink_rms_rel", "step_s0", "kink_s0", "step_jump_rel",
              "kink_slope_m", "kink_slope_p", "dBIC_kink_minus_step", "model"]].round(5).to_string(index=False))
    print("070 sky band:", json.dumps(counts, indent=1), png)


if __name__ == "__main__":
    main()
