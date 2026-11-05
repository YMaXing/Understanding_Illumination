"""Resolution probe (report only): the existing geometry code, unchanged (all px parameters as in geom_v1_1), on DA3 at
network long side 504 (the Stage A cache) and 1008 (scripts/probe_res1008_stage_a.py), five images.

Coordinates: a 504 network pixel centre x maps to 2 x + 0.5 at 1008 (both are area resizes of the same upright image).
Every ROI, lap line and riser box is defined at 504 (as in scripts/geom_v1_1_laps.py) and mapped, so both resolutions
are measured at the same places. "px" is the resolution's own pixel; "px@504" divides 1008 px by 2.
Measures:
  1. 070 middle riser (and the top / bottom risers): face width from the dense normals (per column, the pixels whose
     normal is closer to the riser's than to the tread's), px and metric (3-D distance between the first and last riser
     pixel of the column, back-projected with Z = f_n D / 300).
  2. 068 / 069 laps (lines fitted on geom_v1_1's fold candidates at 504): spacing between consecutive lap lines and
     between same-convexity lines (px and metric, at the shared middle column); DA3 ramp amplitude, the peak-to-peak
     rotation of the dense normal along the line normal within the same physical window (|s| <= 6 px@504); the depth
     channel's r across the line (measure_jump at its own k_min = 11 px, unchanged); 069 inside / outside the pole's
     shadow (the smoothed-luminance mask at 504, mapped).
  3. 7-px grid ripple on the bottom-third RANSAC plane ROI (illum_baseline/diag.floor_roi): relative residual of the
     raw disparity and of the trilateral-filtered one (current px parameters), column Welch spectra (nperseg 64 at 504,
     128 at 1008: the same physical frequency resolution), rms amplitude at lambda = 7.1 px (own grid) and at
     7.1 px@504 (= 14.2 px at 1008).
  4. Normal noise: the dense fit's sigma_theta (median, deg) and the angular scatter of n_hat about the ROI's median
     normal, in the 079 texture ROIs and the 083 bollard interior.
  5. Runtimes: DA3 (from timing.json) and the GeometryContext normals (dense fit) at both resolutions.
Writes baseline_output/diag/geom/res1008/probe.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from depth_rois import BOXES, box_mask  # noqa: E402
from fold_null_rois import rois as tex_rois  # noqa: E402
from geom_v1_1_laps import depth_r_at, normal_profile_amp, shadow_069  # noqa: E402
from geom_w2_hysteresis import lap_lines  # noqa: E402
from illum_baseline.config import dense_fit_params, detector_params, load_config  # noqa: E402
from illum_baseline.detector import compute_weights  # noqa: E402
from illum_baseline.diag import floor_roi, residual_on, welch_columns  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.sampling import bilinear  # noqa: E402

OUT = ROOT / "baseline_output" / "diag" / "geom" / "res1008"
STEMS = {"acharya_mrudula_068": "baseline_output_batch2", "acharya_mrudula_069": "baseline_output_batch2",
         "acharya_mrudula_070": "baseline_output_batch2", "taiwei_cui_079": "baseline_output",
         "taiwei_cui_083": "baseline_output"}
RISERS = {"top": (344, 382, 362.5), "middle": (386, 414, 400.5), "bottom": (443, 478, 460.5)}


def dist(x, scale=1.0):
    x = np.asarray(x, dtype=np.float64) * scale
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "P10": float(np.percentile(x, 10)), "P50": float(np.percentile(x, 50)),
            "P90": float(np.percentile(x, 90))}


def up(mask504: np.ndarray, c: int) -> np.ndarray:
    return mask504 if c == 1 else np.kron(mask504.astype(np.uint8), np.ones((c, c), np.uint8)).astype(bool)


def m(x, c):
    """504 coordinate -> resolution coordinate."""
    return c * np.asarray(x, dtype=np.float64) + (c - 1) / 2


def load(stem, res):
    batch_root = ROOT / STEMS[stem]
    sa = batch_root / "stage_a" if res == 504 else OUT / "stage_a"
    meta = json.loads((sa / f"{stem}.json").read_text())
    z = np.load(sa / f"{stem}.npz")
    cfg = load_config(batch_root / "stage_b" / "geom_v1_1" / "config_resolved.yaml")
    D = z["D"].astype(np.float64)
    sky = z["sky_prob"].astype(np.float64) if "sky_prob" in z else None
    K = Intrinsics.from_stage_a(meta)
    ctx = GeometryContext(z["D"], sky, K, detector_params(cfg), dense_fit_params(cfg))
    Z = meta["focal"]["f_n"] * D / 300.0
    return cfg, meta, ctx, K, Z, sky


def backproject(K, Z, u, v):
    zz, _ = bilinear(Z, np.ones(Z.shape, bool), u, v)
    return np.stack([(u - K.c_x) / K.f_x * zz, (v - K.c_y) / K.f_y * zz, zz], -1)


# ---------------------------------------------------------------- 1. risers
def risers(ctx, K, Z, c):
    nhat = ctx.n_hat.astype(np.float64)
    out = {}
    cols504 = np.arange(100, 281)
    for name, (v0, v1, sp) in RISERS.items():
        # fold rows at 504 (from geom_v1_1's candidate medians): the riser lies between sp - 7 and sp + 7 roughly;
        # use the box halves: upper fold ~ v0 + 8, lower fold ~ v1 - 6 (504 px)
        up_f, lo_f = {"top": (353, 374), "middle": (394, 408), "bottom": (450, 470)}[name]
        cols = np.unique(np.rint(m(cols504, c)).astype(int))
        r0, r1 = int(round(m(up_f + 4, c))), int(round(m(lo_f - 3, c)))
        t0, t1 = int(round(m(lo_f + 5, c))), int(round(m(lo_f + 9, c)))
        n_r = np.nanmedian(nhat[r0:r1, cols].reshape(-1, 3), axis=0)
        n_t = np.nanmedian(nhat[t0:t1, cols].reshape(-1, 3), axis=0)
        n_r, n_t = n_r / np.linalg.norm(n_r), n_t / np.linalg.norm(n_t)
        b0, b1 = int(round(m(v0, c))), int(round(m(v1, c)))
        seg = nhat[b0:b1 + 1][:, cols]
        is_r = (seg @ n_r) > (seg @ n_t)
        px = is_r.sum(0).astype(float)
        metric = []
        for j, u in enumerate(cols):
            idx = np.nonzero(is_r[:, j])[0]
            if idx.size < 2:
                metric.append(np.nan)
                continue
            P = backproject(K, Z, np.array([u, u], float), np.array([b0 + idx[0], b0 + idx[-1]], float))
            metric.append(float(np.linalg.norm(P[0] - P[1])))
        out[name] = {"width_px": dist(px), "width_px@504": dist(px / c), "width_m": dist(metric),
                     "riser_tread_angle_deg": float(np.degrees(np.arccos(np.clip(n_r @ n_t, -1, 1))))}
    return out


# ---------------------------------------------------------------- 2. laps
def laps(stem, cfg, ctx, K, Z, c, lines504, shadow504=None, conv504=None, fold_acc=None):
    nhat, nval = ctx.n_hat.astype(np.float64), ctx.normal_valid.astype(bool)
    rows, per_line = [], []
    conv = []
    pts = []
    for i, (b1, b0, lo, hi) in enumerate(lines504):
        u504 = np.arange(lo, hi + 1, 1.0 / c)
        v504 = b0 + b1 * u504
        nrm = np.array([-b1, 1.0]) / math.hypot(b1, 1.0)
        x0 = np.stack([m(u504, c), m(v504, c)], -1)
        pts.append((u504, v504, x0, np.broadcast_to(nrm, x0.shape).copy()))
    r_all, rv_all, _ = depth_r_at(cfg, ctx, np.concatenate([q[2] for q in pts]), np.concatenate([q[3] for q in pts]))
    r_all = np.where(rv_all, r_all, np.nan)
    offs = np.cumsum([0] + [len(q[2]) for q in pts])
    for i, (b1, b0, lo, hi) in enumerate(lines504):
        u504, v504, x0, e1 = pts[i]
        u, v = x0[:, 0], x0[:, 1]
        amp = normal_profile_amp(nhat, nval, x0, e1, half=6 * c)
        r = r_all[offs[i]:offs[i + 1]]
        L, okL = bilinear(ctx.L_over_rho_f, ctx.omega, u, v)
        conv.append(conv504[i] if conv504 is not None else (float(np.median(-np.sign(L[okL]))) if okL.any() else 0.0))
        sh = np.full(len(u), "all", dtype=object)
        if shadow504 is not None:
            mask, band = shadow504
            ui, vi = np.clip(np.rint(u504).astype(int), 0, 377), np.clip(np.rint(v504).astype(int), 0, 503)
            sh = np.where(band[vi, ui], "boundary", np.where(mask[vi, ui], "inside", "outside"))
        rot = two_sided_rotation(nhat, nval, x0, e1, c)
        acc = fold_acc(x0, 1.5 * c) if fold_acc is not None else np.full(len(u), np.nan)
        rows.append(pd.DataFrame({"line": i, "amp": amp, "rot": rot, "r": r, "shadow": sh, "fold_acc": acc}))
        per_line.append({"line": i, "v_mid504": float(b0 + b1 * (lo + hi) / 2), "convexity": conv[-1],
                         "amp_P50": float(np.nanmedian(amp)), "r_P50": float(np.nanmedian(r))})
    df = pd.concat(rows, ignore_index=True)
    # spacing at the shared middle column of consecutive lines (sorted by v at that column)
    order = sorted(range(len(lines504)), key=lambda i: per_line[i]["v_mid504"])
    sp_any, sp_same, sp_any_m, sp_same_m = [], [], [], []

    def spacing(i, j):
        bi, bj = lines504[i], lines504[j]
        lo, hi = max(bi[2], bj[2]), min(bi[3], bj[3])
        if hi - lo < 5:
            return None
        u504 = (lo + hi) / 2
        vi, vj = bi[1] + bi[0] * u504, bj[1] + bj[0] * u504
        P = backproject(K, Z, m(np.array([u504, u504]), c), m(np.array([vi, vj]), c))
        return abs(vj - vi) * c, float(np.linalg.norm(P[0] - P[1]))
    for a, b in zip(order[:-1], order[1:]):
        s = spacing(a, b)
        if s:
            sp_any.append(s[0])
            sp_any_m.append(s[1])
    for k, a in enumerate(order):
        nxt = next((b for b in order[k + 1:] if np.sign(per_line[b]["convexity"]) == np.sign(per_line[a]["convexity"])), None)
        if nxt is not None:
            s = spacing(a, nxt)
            if s:
                sp_same.append(s[0])
                sp_same_m.append(s[1])
    res = {"n_lines": len(lines504), "spacing_consecutive_px": dist(sp_any), "spacing_consecutive_px@504": dist(sp_any, 1 / c),
           "spacing_consecutive_m": dist(sp_any_m), "pitch_same_convexity_px": dist(sp_same),
           "pitch_same_convexity_px@504": dist(sp_same, 1 / c), "pitch_same_convexity_m": dist(sp_same_m)}
    groups = {"all": df} | ({"inside": df[df.shadow == "inside"], "outside": df[df.shadow == "outside"]}
                            if shadow504 is not None else {})
    for k, g in groups.items():
        res[k] = {"ramp_amplitude_deg": dist(g.amp), "two_sided_rotation_deg": dist(g.rot),
                  "share_columns_fold_accepted": float(np.nanmean(g.fold_acc)), "depth_r": dist(g.r), "share_r_gt_tau_low": float((g.r > 0.02).mean()),
                  "columns": len(g)}
    ctrl = []
    for a, b in zip(order[:-1], order[1:]):
        ba, bb = lines504[a], lines504[b]
        lo, hi = max(ba[2], bb[2]), min(ba[3], bb[3])
        if hi - lo < 10:
            continue
        um = np.arange(lo, hi + 1, 1.0 / c)
        va, vb = ba[1] + ba[0] * um, bb[1] + bb[0] * um
        if np.median(np.abs(vb - va)) < 20:
            continue
        b1 = 0.5 * (ba[0] + bb[0])
        x0 = np.stack([m(um, c), m(0.5 * (va + vb), c)], -1)
        e1 = np.broadcast_to(np.array([-b1, 1.0]) / math.hypot(b1, 1.0), x0.shape).copy()
        ctrl.append(pd.DataFrame({"amp": normal_profile_amp(nhat, nval, x0, e1, half=6 * c),
                                  "rot": two_sided_rotation(nhat, nval, x0, e1, c)}))
    if ctrl:
        cd = pd.concat(ctrl, ignore_index=True)
        res["control_mid_lap"] = {"ramp_amplitude_deg": dist(cd.amp), "two_sided_rotation_deg": dist(cd.rot),
                                  "columns": len(cd)}
    res["per_line"] = per_line
    return res


def two_sided_rotation(nhat, valid, x0, e1, c):
    """Angle (deg) between the mean dense normals on s in [-6, -2] and [2, 6] px@504 along e1 (noise-robust ramp)."""
    out = []
    for lo, hi in ((-6, -2), (2, 6)):
        s = np.arange(lo * c, hi * c + 0.5, 1.0)
        pts = x0[:, None, :] + s[None, :, None] * e1[:, None, :]
        comp = []
        for k in range(3):
            v, ok = bilinear(nhat[..., k], valid, pts[..., 0], pts[..., 1])
            comp.append(np.nanmean(np.where(ok, v, np.nan), axis=1))
        n = np.stack(comp, -1)
        out.append(n / np.linalg.norm(n, axis=-1, keepdims=True))
    return np.degrees(np.arccos(np.clip(np.sum(out[0] * out[1], -1), -1, 1)))


# ---------------------------------------------------------------- 3. ripple
def ripple(cfg, ctx, D, sky, c):
    det = compute_weights(D, sky, detector_params(cfg))
    sc = det["scalars"]
    q_s = det["q"] / sc["s_hat"]
    info = floor_roi(det["rho"], det["sky_mask"], q_s, sc["R"] + sc["r"], cfg.seed)
    roi = info["roi"]
    nper = 64 * c
    out = {"roi_px": int(roi.sum()), "roi_px@504": float(roi.sum() / c**2)}
    for tag, rho in (("raw", det["rho"]), ("trilateral", det["rho_filtered"])):
        e = residual_on(rho, roi)
        f, p, nseg = welch_columns(np.nan_to_num(e), roi, nper)
        if f.size == 0:
            out[tag] = {"n_segments": 0}
            continue
        df = f[1] - f[0]
        amp = lambda lam: float(np.sqrt(p[int(np.argmin(np.abs(1 / np.maximum(f, 1e-12) - lam)))] * df))  # noqa: E731
        lam_own, lam_phys = 64 / 9, 64 / 9 * c
        lam = 1 / np.maximum(f, 1e-12)

        def ratio(l0):
            i = int(np.argmin(np.abs(lam - l0)))
            nb = [j for j in range(max(3, i - 3), min(len(f), i + 4)) if abs(j - i) >= 2]
            return float(p[i] / np.median(p[nb])) if nb else float("nan")
        out[tag + "_psd_over_neighbours"] = {"at_7.1px_own": ratio(64 / 9), "at_7.1px@504": ratio(64 / 9 * c)}
        out[tag] = {"n_segments": nseg, "rms_e": float(np.sqrt(np.nanmean(e**2))),
                    "amp_at_7.1px_own": amp(lam_own), "amp_at_7.1px@504": amp(lam_phys),
                    "peak_wavelength_px": float(1 / f[3:][np.argmax(p[3:])]), "peak_amp": float(np.sqrt(p[3:].max() * df))}
    return out


# ---------------------------------------------------------------- 4. normal noise
def noise(stem, ctx, c):
    nhat = ctx.n_hat.astype(np.float64)
    st = np.degrees(ctx.sigma_theta.astype(np.float64))
    valid = ctx.normal_valid.astype(bool)
    masks = {}
    if stem.endswith("079"):
        masks = tex_rois(stem, (504, 378))
    if stem.endswith("083"):
        masks = {"083 bollard interior": box_mask((504, 378), BOXES["083 bollard interior"][2])}
    out = {}
    for name, m504 in masks.items():
        mk = up(m504, c) & valid
        nn = nhat[mk]
        mu = np.median(nn, 0)
        mu /= np.linalg.norm(mu)
        ang = np.degrees(np.arccos(np.clip(nn @ mu, -1, 1)))
        out[name] = {"sigma_theta_deg": dist(st[mk]), "scatter_about_median_deg": dist(ang), "px": int(mk.sum())}
    return out


def main():
    timing = json.loads((OUT / "timing.json").read_text())
    lines = lap_lines()
    _, _, _, sh, band, _ = shadow_069(None)
    rep = {}
    for stem in STEMS:
        rep[stem] = {}
        for res in (504, 1008):
            c = res // 504
            cfg, meta, ctx, K, Z, sky = load(stem, res)
            t0 = time.perf_counter()
            ctx.normals
            t_norm = time.perf_counter() - t0
            r = {"network_HxW": list(ctx.shape), "f_n": meta["focal"]["f_n"], "Z_median_m": float(np.median(Z)),
                 "da3": timing[stem][str(res)], "context_normals_s": t_norm}
            fold_acc = None
            if stem.endswith(("068", "069", "070")):
                import dataclasses

                from scipy.spatial import cKDTree

                from illum_baseline.geom.fold import FoldChannel, FoldParams
                t1 = time.perf_counter()
                fr = FoldChannel().run(ctx, FoldParams.from_dict(dataclasses.asdict(cfg.geometry.fold)))
                r["fold_channel_s"] = time.perf_counter() - t1
                fc = fr.candidates
                acc = fc[fc.accepted]
                r["fold"] = {"lambda_hat": fr.scalars["lambda_hat"], "n_cand": fr.scalars["n_cand"],
                             "n_accepted": fr.scalars["n_accepted"], "n_rescued": fr.scalars.get("n_rescued")}
                tree = cKDTree(acc[["u0", "v0"]].to_numpy()) if len(acc) else None

                def fold_acc(x0, tol, tree=tree):
                    if tree is None:
                        return np.zeros(len(x0))
                    return np.isfinite(tree.query(x0, distance_upper_bound=tol)[0]).astype(float)
            if stem.endswith("070"):
                r["risers"] = risers(ctx, K, Z, c)
                for name, vf in (("middle upper", 394), ("middle lower", 408), ("top upper", 353), ("bottom lower", 470)):
                    u504 = np.arange(100, 281, 1.0 / c)
                    x0 = np.stack([m(u504, c), np.full(len(u504), m(vf, c))], -1)
                    r["risers"].setdefault("fold_accepted_share", {})[name] = float(fold_acc(x0, 2.0 * c).mean())
            if stem in lines:
                conv504 = None if res == 504 else [pl["convexity"] for pl in rep[stem]["504"]["laps"]["per_line"]]
                r["laps"] = laps(stem, cfg, ctx, K, Z, c, lines[stem], (sh, band) if stem.endswith("069") else None,
                                 conv504, fold_acc)
            r["ripple"] = ripple(cfg, ctx, np.load((ROOT / STEMS[stem] / "stage_a" if res == 504 else OUT / "stage_a")
                                                   / f"{stem}.npz")["D"].astype(np.float64), sky, c)
            r["normal_noise"] = noise(stem, ctx, c)
            rep[stem][str(res)] = r
            print(stem, res, "done", round(t_norm, 1), "s", flush=True)
        # D consistency: metric Z at 1008 downsampled vs 504
        z5 = rep[stem]["504"]["Z_median_m"]
        rep[stem]["Z_median_ratio_1008_over_504"] = rep[stem]["1008"]["Z_median_m"] / z5
    (OUT / "probe.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
