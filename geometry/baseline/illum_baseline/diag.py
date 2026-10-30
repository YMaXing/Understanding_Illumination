"""Stripe diagnostics for one image (TASK_070_STRIPES.md, T1). Read-only w.r.t. the pipeline outputs.

Writes out_dir/diag/<id>/: ROI mask + overlay, residual plots and spectra, code-transition test,
DA3 shift test, summary.json. Network resolution, upright frame, float64 throughout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage, signal  # noqa: E402

from .config import Config, detector_params, load_config  # noqa: E402
from .detector import compute_weights, disk  # noqa: E402
from .stage_a import prepare, stage_a_dir  # noqa: E402
from .stage_b import run_dir  # noqa: E402

SHIFT = 7
SUPP_SHIFT = 4
RANSAC_ITERS = 4000
INLIER_REL = 0.01
Q_EXCLUDE = 100.0
STRIPE_K = 3.0
NULL_SHIFTS = [s for s in range(-10, 11) if abs(s) >= 3]
SMOOTH_SIGMA = 2.0  # supplementary test: iso-code contours of the smoothed luma
MAX_LAG = 12


# ---------------------------------------------------------------- 1. ROI

def fit_plane(u: np.ndarray, v: np.ndarray, rho: np.ndarray) -> np.ndarray:
    A = np.stack([u, v, np.ones_like(u)], axis=1)
    return np.linalg.lstsq(A, rho, rcond=None)[0]


def eval_plane(coef: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    v, u = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    return coef[0] * u + coef[1] * v + coef[2]


def floor_roi(rho: np.ndarray, sky: np.ndarray, q_s: np.ndarray, erode_r: int, seed: int) -> dict[str, Any]:
    H, W = rho.shape
    v, u = np.mgrid[0:H, 0:W]
    cand = (v >= 2 * H // 3) & ~sky
    cu, cv, cr = u[cand].astype(float), v[cand].astype(float), rho[cand]
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(cr.size, 3, replace=False)
        A = np.stack([cu[idx], cv[idx], np.ones(3)], axis=1)
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        coef = np.linalg.solve(A, cr[idx])
        n = int((np.abs(cr - (coef[0] * cu + coef[1] * cv + coef[2])) < INLIER_REL * cr).sum())
        if n > best_n:
            best, best_n = coef, n
    # Least-squares refit on the bottom-third inliers, then inliers over the whole image.
    fit = best
    inl3 = np.abs(cr - (fit[0] * cu + fit[1] * cv + fit[2])) < INLIER_REL * cr
    fit = fit_plane(cu[inl3], cv[inl3], cr[inl3])
    plane = eval_plane(fit, rho.shape)
    inl = (np.abs(rho - plane) < INLIER_REL * np.abs(rho)) & ~sky & (rho > 0)
    lab, n = ndimage.label(inl, structure=np.ones((3, 3), bool))
    comp = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n else np.zeros_like(inl)
    eroded = ndimage.binary_erosion(comp, structure=disk(erode_r))  # border_value=0: also trims the image border
    roi = eroded & (q_s <= Q_EXCLUDE)
    # Final plane: least squares on the ROI itself (used for the residual e).
    fit_roi = fit_plane(u[roi].astype(float), v[roi].astype(float), rho[roi])
    return {
        "roi": roi, "component": comp, "fit_bottom_third": fit, "fit_roi": fit_roi,
        "ransac_inliers_bottom_third": int(inl3.sum()), "n_candidates": int(cand.sum()),
        "n_component": int(comp.sum()), "n_eroded": int(eroded.sum()), "n_roi": int(roi.sum()),
    }


# ---------------------------------------------------------------- 2. residual and spectra

def column_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int, int]]:
    """(u, v0, v1) for the longest contiguous run of each column with length >= min_len."""
    out = []
    for c in range(mask.shape[1]):
        col = mask[:, c]
        if not col.any():
            continue
        lab, n = ndimage.label(col)
        sizes = np.bincount(lab)[1:]
        k = int(np.argmax(sizes)) + 1
        if sizes[k - 1] >= min_len:
            idx = np.nonzero(lab == k)[0]
            out.append((c, int(idx[0]), int(idx[-1]) + 1))
    return out


def welch_columns(e: np.ndarray, mask: np.ndarray, nperseg: int) -> tuple[np.ndarray, np.ndarray, int]:
    acc, wsum = None, 0
    for c, v0, v1 in column_runs(mask, nperseg):
        x = e[v0:v1, c]
        f, p = signal.welch(x, nperseg=nperseg, noverlap=nperseg // 2, detrend="linear", window="hann")
        nseg = 1 + (x.size - nperseg) // (nperseg - nperseg // 2)
        acc = p * nseg if acc is None else acc + p * nseg
        wsum += nseg
    if acc is None:
        return np.array([]), np.array([]), 0
    return f, acc / wsum, wsum


def spectral_peaks(f: np.ndarray, p: np.ndarray, n: int = 3) -> list[dict[str, float]]:
    """Top-n local maxima above the 2 lowest nonzero bins (trend), by power."""
    if f.size == 0:
        return []
    keep = np.arange(f.size) >= 3
    idx, _ = signal.find_peaks(np.where(keep, p, 0.0))
    idx = sorted(idx, key=lambda i: -p[i])[:n]
    med = float(np.median(p[keep]))
    return [{"wavelength_px": float(1 / f[i]), "freq": float(f[i]), "psd": float(p[i]),
             "psd_over_median": float(p[i] / med), "amplitude_rms": float(np.sqrt(p[i] * (f[1] - f[0])))}
            for i in idx]


# ---------------------------------------------------------------- 3. code transitions

def stripe_crossings(lor_s: np.ndarray, roi: np.ndarray, k: float) -> list[tuple[int, float]]:
    """Zero crossings of L/rho along v inside the ROI whose two adjacent lobes both exceed k (units of s_hat).

    Returns (column, v position) with v interpolated between the two samples that straddle zero.
    """
    out = []
    H, W = lor_s.shape
    for c in range(W):
        lab, n = ndimage.label(roi[:, c])
        for j in range(1, n + 1):
            idx = np.nonzero(lab == j)[0]
            x = lor_s[idx, c]
            if x.size < 3:
                continue
            zc = np.nonzero(np.sign(x[:-1]) * np.sign(x[1:]) < 0)[0]  # crossing between i and i+1
            bounds = np.concatenate([[-1], zc, [x.size - 1]])
            lobe = [np.abs(x[bounds[i] + 1:bounds[i + 1] + 1]).max() for i in range(len(bounds) - 1)]
            for m, i in enumerate(zc):
                if min(lobe[m], lobe[m + 1]) > k:
                    pos = idx[i] + x[i] / (x[i] - x[i + 1])
                    out.append((c, float(pos)))
    return out


def code_transitions(net: np.ndarray) -> np.ndarray:
    """T[v, u]: the uint8 code changes between rows v and v+1 in any channel (position v + 0.5)."""
    return np.any(net[1:] != net[:-1], axis=2)


def smoothed_level_transitions(net: np.ndarray, sigma: float) -> np.ndarray:
    """Supplementary: rows where floor(smoothed luma) changes, i.e. the band edges a posterized smooth
    gradient would have once pixel noise (which flips codes almost everywhere) is averaged out."""
    y = ndimage.gaussian_filter(net.astype(np.float64) @ [0.2126, 0.7152, 0.0722], sigma)
    lev = np.floor(y)
    return lev[1:] != lev[:-1]


def transition_score(cross: list[tuple[int, float]], T: np.ndarray, shift: int = 0, tol: float = 1.0) -> float:
    if not cross:
        return float("nan")
    Ts = np.zeros_like(T)
    if shift > 0:
        Ts[shift:] = T[:-shift]
    elif shift < 0:
        Ts[:shift] = T[-shift:]
    else:
        Ts = T
    hit = 0
    for c, z in cross:
        lo, hi = int(np.ceil(z - tol - 0.5)), int(np.floor(z + tol - 0.5))
        lo, hi = max(lo, 0), min(hi, Ts.shape[0] - 1)
        hit += bool(Ts[lo:hi + 1, c].any()) if hi >= lo else False
    return hit / len(cross)


# ---------------------------------------------------------------- 4. shift test

def shift_input(net: np.ndarray, s: int) -> np.ndarray:
    """Reflect-pad top/left by s, crop bottom/right by s: content moves down/right by s px."""
    p = np.pad(net, ((s, 0), (s, 0), (0, 0)), mode="reflect")
    return np.ascontiguousarray(p[: net.shape[0], : net.shape[1]])


def shift_back(a: np.ndarray, s: int) -> np.ndarray:
    """Output of the shifted input mapped back to original coordinates; NaN where not covered."""
    out = np.full(a.shape, np.nan)
    out[: a.shape[0] - s, : a.shape[1] - s] = a[s:, s:]
    return out


def residual_on(rho: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """e = (rho - rho_fit)/rho_fit with a least-squares plane on mask; NaN outside."""
    v, u = np.nonzero(mask)
    coef = fit_plane(u.astype(float), v.astype(float), rho[mask])
    fit = eval_plane(coef, rho.shape)
    return np.where(mask, (rho - fit) / fit, np.nan)


def xcorr_v(a: np.ndarray, b: np.ndarray, max_lag: int) -> dict[int, float]:
    """Pooled normalized cross-correlation sum_v a[v] b[v+lag] over all columns (NaN = missing)."""
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[: a.shape[0] - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: b.shape[0] + lag]
        m = np.isfinite(x) & np.isfinite(y)
        out[lag] = float((x[m] * y[m]).sum() / np.sqrt((x[m] ** 2).sum() * (y[m] ** 2).sum()))
    return out


# ---------------------------------------------------------------- driver

def run_diag(cfg: Config, stem: str, run_name: str, gpu: bool = True) -> Path:
    rd = run_dir(cfg, run_name)
    rcfg = load_config(rd / "config_resolved.yaml")
    params = detector_params(rcfg)
    out = cfg.out_dir / "diag" / stem.rsplit("_", 1)[-1]
    out.mkdir(parents=True, exist_ok=True)

    sa = stage_a_dir(cfg)
    meta_a = json.loads((sa / f"{stem}.json").read_text())
    z = np.load(sa / f"{stem}.npz")
    D = z["D"].astype(np.float64)
    sky_prob = z["sky_prob"].astype(np.float64)
    det = compute_weights(D, sky_prob, params)
    sc = det["scalars"]
    s_hat = sc["s_hat"]
    S, rho, q_s = det["sky_mask"], det["rho"], det["q"] / s_hat
    lor_s = det["L_over_rho"] / s_hat

    # Sanity: recomputed w matches the saved baseline_v0 w_net (float16) for orientation 1.
    saved = np.load(rd / f"{stem}.npz")
    orient = meta_a["frames"]["orientation_applied"]
    w_check = float(np.abs(saved["w_net"].astype(np.float64) - det["w"]).max()) if orient == 1 else None

    # Network input exactly as DA3 saw it.
    item = prepare(cfg, cfg.data_dir / meta_a["file"])
    net = item["net"]
    assert net.shape[:2] == D.shape

    # 1. ROI
    R, r = sc["R"], sc["r"]
    roi_info = floor_roi(rho, S, q_s, R + r, cfg.seed)
    roi = roi_info["roi"]
    np.save(out / "roi_mask.npy", roi)
    plt.imsave(out / "roi_mask.png", roi, cmap="gray")
    ov = net.astype(float) / 255
    ov[roi] = 0.5 * ov[roi] + 0.5 * np.array([0.0, 1.0, 0.0])
    comp_only = roi_info["component"] & ~roi
    ov[comp_only] = 0.6 * ov[comp_only] + 0.4 * np.array([1.0, 1.0, 0.0])
    plt.imsave(out / "roi_overlay.png", np.clip(ov, 0, 1))

    # 2. Residual
    fit = eval_plane(roi_info["fit_roi"], rho.shape)
    e = np.where(roi, (rho - fit) / fit, np.nan)
    runs = column_runs(roi, 64)
    cols = [runs[int(i)][0] for i in np.linspace(0, len(runs) - 1, 5)] if len(runs) >= 5 else [c for c, _, _ in runs]
    fig, ax = plt.subplots(len(cols), 1, figsize=(10, 2.0 * len(cols)), sharex=True, constrained_layout=True)
    for a_, c in zip(np.atleast_1d(ax), cols):
        a_.plot(np.arange(rho.shape[0]), 100 * e[:, c], lw=0.8)
        a_.set_ylabel(f"e [%]  u={c}")
        a_.grid(alpha=0.3)
    np.atleast_1d(ax)[-1].set_xlabel("v [px] (network res, upright)")
    fig.suptitle(f"{stem}: e = (rho - rho_fit)/rho_fit on the floor ROI")
    fig.savefig(out / "residual_columns.png", dpi=100)
    plt.close(fig)

    f_all, p_all, nseg_all = welch_columns(np.nan_to_num(e), roi, 64)
    vv = np.nonzero(roi.any(axis=1))[0]
    edges = np.linspace(vv.min(), vv.max() + 1, 4).round().astype(int)
    bands = []
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), constrained_layout=True)
    ax.loglog(f_all[1:], p_all[1:], "k", lw=1.5, label=f"all v (nperseg 64, {nseg_all} seg)")
    for b in range(3):
        bm = roi.copy()
        bm[: edges[b]] = False
        bm[edges[b + 1]:] = False
        f_b, p_b, n_b = welch_columns(np.nan_to_num(e), bm, 32)
        rho_c = float(np.median(fit[bm])) if bm.any() else float("nan")
        bands.append({"v_range": [int(edges[b]), int(edges[b + 1])], "n_segments": n_b,
                      "rho_fit_median": rho_c, "peaks": spectral_peaks(f_b, p_b)})
        if n_b:
            ax.loglog(f_b[1:], p_b[1:], lw=1, label=f"band v {edges[b]}-{edges[b + 1]} (nperseg 32, {n_b} seg)")
    for lam in (14, 7):
        ax.axvline(1 / lam, color="0.6", ls=":", lw=1)
        ax.text(1 / lam, ax.get_ylim()[1], f"{lam} px", fontsize=8, ha="center", va="top")
    ax.set_xlabel("frequency along v [cycles/px]")
    ax.set_ylabel("PSD of e")
    ax.legend(fontsize=8)
    ax.set_title(f"{stem}: Welch spectrum of e along v, column-averaged")
    fig.savefig(out / "residual_spectrum.png", dpi=100)
    plt.close(fig)
    # A pattern fixed on the ground plane has an image wavelength proportional to rho^2 along v
    # (Z ~ 1/rho, rho affine in v, so dZ/dv ~ 1/rho^2). A grid-locked pattern keeps a constant wavelength.
    for b in bands:
        b["ground_locked_wavelength_ratio_vs_band0"] = (b["rho_fit_median"] / bands[0]["rho_fit_median"]) ** 2

    # 3. Code transitions
    T = code_transitions(net)
    cross = stripe_crossings(lor_s, roi, STRIPE_K)
    score = transition_score(cross, T)
    nulls = {s: transition_score(cross, T, shift=s) for s in NULL_SHIFTS}
    roi_pairs = roi[1:] & roi[:-1]
    t_density = float(T[roi_pairs].mean())
    Tsm = smoothed_level_transitions(net, SMOOTH_SIGMA)
    sm_score = transition_score(cross, Tsm)
    sm_nulls = np.array([transition_score(cross, Tsm, shift=s) for s in NULL_SHIFTS])
    lum = (net.astype(float) @ [0.2126, 0.7152, 0.0722])[roi]
    n_codes = {ch: int(np.unique(net[..., i][roi]).size) for i, ch in enumerate("RGB")}

    # 4. Shift test. Spec: input shifted by (7,7), xcorr of e along v. Supplementary (flagged): the same with
    # the band-passed stripe signal L/rho, and a (4,4) shift, because a 7-px-periodic grid pattern maps onto
    # itself under a 7-px shift (lag -7 == lag 0 mod 7).
    shift = None
    if gpu:
        from .depth import DepthEstimator
        from .stage_a import resolve_device

        dev = resolve_device(cfg.device)
        de = DepthEstimator(cfg.da3.model_id, cfg.da3.process_res, cfg.da3.process_res_method, dev, cfg.seed)
        D0 = de.predict([net])[0]["D"].astype(np.float64)
        rerun_identical = bool(np.array_equal(D0, z["D"].astype(np.float64)))
        H, W = rho.shape
        shift = {"rerun_unshifted_bit_identical_to_cache": rerun_identical, "shifts": {}}
        curves = {}
        arrays = {"D_rerun": D0.astype(np.float32)}
        for sft in (SHIFT, SUPP_SHIFT):
            pred = de.predict([shift_input(net, sft)])[0]
            Ds = pred["D"].astype(np.float64)
            arrays[f"D_shift{sft}"] = Ds.astype(np.float32)
            Db = shift_back(Ds, sft)
            overlap = roi & np.isfinite(Db)
            rec: dict[str, Any] = {"n_overlap": int(overlap.sum())}
            if sft == SHIFT:
                e_o = residual_on(rho, overlap)
                e_s = residual_on(1.0 / np.where(np.isfinite(Db), Db, 1.0), overlap)
                xe = xcorr_v(e_o, e_s, MAX_LAG)
                curves[f"e, shift {sft} (spec)"] = (xe, sft)
                f_s, p_s, _ = welch_columns(np.nan_to_num(e_s), overlap, 64)
                rec.update({
                    "e_xcorr": {str(k): v for k, v in xe.items()},
                    "e_xcorr_lag0": xe[0], "e_xcorr_lag_minus_shift": xe[-sft], "e_xcorr_lag_plus_shift": xe[sft],
                    "e_argmax_lag": max(xe, key=xe.get),
                    "rms_e_orig": float(np.sqrt(np.nanmean(e_o**2))), "rms_e_shift": float(np.sqrt(np.nanmean(e_s**2))),
                    "shifted_e_peaks": spectral_peaks(f_s, p_s),
                })
                arrays.update(e_orig=e_o.astype(np.float32), e_shift=e_s.astype(np.float32))
            # L/rho of the shifted run, computed in its own frame and mapped back; drop rows/cols whose kernel
            # support touches the shifted image's bottom/right border.
            det_s = compute_weights(Ds, pred["sky_prob"].astype(np.float64) if pred["sky_prob"] is not None else None,
                                    params)
            Ls = shift_back(det_s["L_over_rho"], sft)
            m = overlap.copy()
            m[H - sft - R:, :] = False
            m[:, W - sft - R:] = False
            xl = xcorr_v(np.where(m, det["L_over_rho"], np.nan), np.where(m, Ls, np.nan), MAX_LAG)
            curves[f"L/rho, shift {sft}"] = (xl, sft)
            grid_lags = sorted({-sft, (-sft) % 7 if (-sft) % 7 <= MAX_LAG // 2 else (-sft) % 7 - 7})
            rec.update({
                "L_xcorr": {str(k): v for k, v in xl.items()},
                "L_xcorr_lag0": xl[0], "L_xcorr_lag_minus_shift": xl[-sft],
                "L_grid_lags_period14_period7": grid_lags,
                "L_xcorr_at_grid_lags": {str(k): xl[k] for k in grid_lags},
                "L_argmax_lag": max(xl, key=xl.get),
                "s_hat_shifted": det_s["scalars"]["s_hat"],
            })
            shift["shifts"][str(sft)] = rec
        np.savez_compressed(out / "shift_test.npz", **arrays)

        fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
        for (lab_, (xc, sft)), mk in zip(curves.items(), ("o-", "s-", "^-")):
            lags = np.array(sorted(xc))
            ax.plot(lags, [xc[k] for k in lags], mk, ms=4, label=lab_)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.text(0, ax.get_ylim()[0], " scene-locked", fontsize=8, va="bottom")
        for x_, lab_ in [(-SHIFT, "grid, shift 7"), (-SUPP_SHIFT, "grid, shift 4 (14-px)"), (3, "grid, shift 4 (7-px)")]:
            ax.axvline(x_, color="0.6", ls=":", lw=1)
            ax.text(x_, ax.get_ylim()[1], lab_, fontsize=7, ha="center", va="top", rotation=90)
        ax.set_xlabel("lag along v [px]:  sum_v a_orig[v] a_shift_back[v+lag]")
        ax.set_ylabel("normalized xcorr")
        ax.set_title(f"{stem}: DA3 shift test on the floor ROI")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.savefig(out / "shift_xcorr.png", dpi=100)
        plt.close(fig)

    # Code-transition figure: a crop of the floor with transitions and stripe crossings.
    vs = vv.min() + (vv.max() - vv.min()) // 3
    v0, v1 = max(0, vs - 40), min(rho.shape[0], vs + 40)
    us = np.nonzero(roi[vs])[0]
    u0 = int(us[len(us) // 2] - 60) if us.size else 0
    u0, u1 = max(u0, 0), min(rho.shape[1], max(u0, 0) + 120)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    ax[0].imshow(net[v0:v1, u0:u1], extent=(u0 - 0.5, u1 - 0.5, v1 - 0.5, v0 - 0.5), interpolation="nearest")
    tv, tu = np.nonzero(T[v0:v1 - 1, u0:u1])
    ax[0].plot(tu + u0, tv + v0 + 0.5, ",", color="cyan")
    cc = np.array([(c, p) for c, p in cross if u0 <= c < u1 and v0 <= p < v1])
    if cc.size:
        ax[0].plot(cc[:, 0], cc[:, 1], ".", color="red", ms=3)
    ax[0].set_title("network input (uint8); cyan: code transitions, red: stripe zero crossings")
    im = ax[1].imshow(lor_s[v0:v1, u0:u1], cmap="RdBu_r", vmin=-20, vmax=20,
                      extent=(u0 - 0.5, u1 - 0.5, v1 - 0.5, v0 - 0.5))
    fig.colorbar(im, ax=ax[1], label="L/rho / s_hat")
    ax[1].set_title("signed L/rho / s_hat (same crop)")
    fig.savefig(out / "code_transitions_crop.png", dpi=110)
    plt.close(fig)

    null_vals = np.array(list(nulls.values()))
    summary = {
        "stem": stem, "run": run_name, "s_hat": s_hat, "R": R, "r": r,
        "w_net_check_max_abs_diff_vs_saved_float16": w_check,
        "roi": {k: (v.tolist() if isinstance(v, np.ndarray) and v.ndim == 1 else v)
                for k, v in roi_info.items() if k not in ("roi", "component")},
        "residual": {
            "rms_e": float(np.sqrt(np.nanmean(e**2))),
            "peaks_all_v": spectral_peaks(f_all, p_all), "n_segments_all": nseg_all,
            "bands": bands, "columns_plotted": cols,
        },
        "code_transition": {
            "n_stripe_crossings": len(cross),
            "score_within_1px": score,
            "null_shifts": NULL_SHIFTS,
            "null_scores": {str(k): v for k, v in nulls.items()},
            "null_mean": float(np.nanmean(null_vals)), "null_max": float(np.nanmax(null_vals)),
            "transition_density_in_roi": t_density,
            "n_codes_in_roi": n_codes,
            "roi_luma_range_uint8": [float(lum.min()), float(lum.max())],
            "saturated": bool(t_density > 0.5),
            "supplementary_smoothed_luma": {
                "sigma_px": SMOOTH_SIGMA,
                "transition_density_in_roi": float(Tsm[roi_pairs].mean()),
                "score_within_1px": sm_score,
                "null_mean": float(np.nanmean(sm_nulls)), "null_max": float(np.nanmax(sm_nulls)),
            },
        },
        "shift_test": shift,
    }
    summary["verdict"] = verdict(summary)
    summary["headline"] = headline(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return out


def verdict(s: dict[str, Any]) -> str:
    ct = s["code_transition"]
    sh = s["shift_test"]
    sm = ct["supplementary_smoothed_luma"]
    p_like = (sm if ct["saturated"] else ct)["score_within_1px"] > (sm if ct["saturated"] else ct)["null_max"] + 0.1
    parts = [f"code-transition {ct['score_within_1px']:.2f} vs null {ct['null_mean']:.2f}"
             + (" (saturated)" if ct["saturated"] else ""),
             f"smoothed-luma levels {sm['score_within_1px']:.2f} vs null {sm['null_mean']:.2f} (max {sm['null_max']:.2f})"]
    if sh is None:
        return "; ".join(parts) + "; shift test not run"
    s7, s4 = sh["shifts"][str(SHIFT)], sh["shifts"][str(SUPP_SHIFT)]
    grid = max(s4["L_xcorr_at_grid_lags"].values()) > s4["L_xcorr_lag0"]
    parts.append(f"e xcorr (shift 7) lag0 {s7['e_xcorr_lag0']:.2f} vs lag-7 {s7['e_xcorr_lag_minus_shift']:.2f}")
    parts.append(f"L/rho xcorr (shift 4) lag0 {s4['L_xcorr_lag0']:.2f} vs grid lags "
                 + ", ".join(f"{k}: {v:.2f}" for k, v in s4["L_xcorr_at_grid_lags"].items()))
    if grid:
        lean = "G (DA3 grid-locked)"
    elif p_like:
        lean = "P (posterization)"
    else:
        lean = "scene-locked but not tied to code transitions: T (texture) or P via a broader mechanism"
    return "; ".join(parts) + f" -> lean: {lean}"


def headline(s: dict[str, Any]) -> dict[str, Any]:
    """The T1 summary fields in one place (spec'd tests first, supplementary ones marked)."""
    ct, sh, res = s["code_transition"], s["shift_test"], s["residual"]
    h: dict[str, Any] = {
        "code_transition_score": ct["score_within_1px"],
        "code_transition_null_mean": ct["null_mean"],
        "code_transition_saturated": ct["saturated"],
        "supp_smoothed_luma_score": ct["supplementary_smoothed_luma"]["score_within_1px"],
        "supp_smoothed_luma_null_mean": ct["supplementary_smoothed_luma"]["null_mean"],
        "peak_wavelengths_px_all_v": [round(p["wavelength_px"], 2) for p in res["peaks_all_v"]],
        "peak_wavelengths_px_by_band": [[round(p["wavelength_px"], 2) for p in b["peaks"]] for b in res["bands"]],
        "ground_locked_wavelength_ratio_by_band": [round(b["ground_locked_wavelength_ratio_vs_band0"], 2)
                                                   for b in res["bands"]],
        "rms_e": res["rms_e"],
    }
    if sh is not None:
        s7, s4 = sh["shifts"][str(SHIFT)], sh["shifts"][str(SUPP_SHIFT)]
        h.update({
            "xcorr_e_shift7_lag0": s7["e_xcorr_lag0"], "xcorr_e_shift7_lag_minus7": s7["e_xcorr_lag_minus_shift"],
            "xcorr_e_shift7_lag_plus7": s7["e_xcorr_lag_plus_shift"],
            "supp_xcorr_L_shift7_lag0": s7["L_xcorr_lag0"], "supp_xcorr_L_shift7_lag_minus7": s7["L_xcorr_lag_minus_shift"],
            "supp_xcorr_L_shift4_lag0": s4["L_xcorr_lag0"], "supp_xcorr_L_shift4_lag_minus4": s4["L_xcorr_lag_minus_shift"],
            "rerun_bit_identical": sh["rerun_unshifted_bit_identical_to_cache"],
        })
    h["verdict"] = s["verdict"]
    return h


# ---------------------------------------------------------------- variant metrics (T2-T4)

STRIPE_WAVELENGTH = 64 / 9  # the 7.1-px Welch bin (nperseg 64) holding the T1 stripe peak


def _run_detector(rd: Path, stem: str) -> tuple[Config, dict[str, Any], np.ndarray]:
    """Recompute the detector in float64 from the run's own Stage A cache and parameters."""
    rcfg = load_config(rd / "config_resolved.yaml")
    z = np.load(stage_a_dir(rcfg) / f"{stem}.npz")
    sky = z["sky_prob"].astype(np.float64) if "sky_prob" in z.files else None
    return rcfg, compute_weights(z["D"].astype(np.float64), sky, detector_params(rcfg)), z["D"].astype(np.float64)


def run_metrics(cfg: Config, stem: str, run: str, roi: np.ndarray, edge_F: np.ndarray) -> dict[str, Any]:
    rd = run_dir(cfg, run)
    rcfg, det, D = _run_detector(rd, stem)
    sc = det["scalars"]
    s_hat, q, w, rho = sc["s_hat"], det["q"], det["w"], det.get("rho_filtered", det["rho"])  # what the Laplacian saw
    e = residual_on(rho, roi)
    f, p, _ = welch_columns(np.nan_to_num(e), roi, 64)
    pk = spectral_peaks(f, p, 1)
    i7 = int(np.argmin(np.abs(1 / np.maximum(f, 1e-12) - STRIPE_WAVELENGTH)))
    meta = json.loads((stage_a_dir(rcfg) / f"{stem}.json").read_text())
    sa_cfg = load_config(stage_a_dir(rcfg) / "config_resolved.yaml")  # what produced the Stage A cache
    net = prepare(sa_cfg, rcfg.data_dir / meta["file"])["net"]
    luma = net.astype(np.float64) @ [0.2126, 0.7152, 0.0722]
    saved = np.load(rd / f"{stem}.npz")
    return {
        "run": run,
        "variant": rcfg.paths.stage_a_variant,
        "exposure_mode": sa_cfg.preprocess.exposure_mode,
        "dither": sa_cfg.preprocess.dither,
        "roi_frac_w_lt_0.9": float((w[roi] < 0.9).mean()),
        "roi_frac_w_lt_0.5": float((w[roi] < 0.5).mean()),
        "roi_median_q_over_s": float(np.median(q[roi]) / s_hat),
        "roi_median_q_abs": float(np.median(q[roi])),
        "roi_p95_q_abs": float(np.percentile(q[roi], 95)),
        "roi_rms_e": float(np.sqrt(np.nanmean(e**2))),
        "roi_peak_wavelength_px": pk[0]["wavelength_px"] if pk else None,
        "roi_peak_amplitude": pk[0]["amplitude_rms"] if pk else None,
        "roi_amplitude_at_7px": float(np.sqrt(p[i7] * (f[1] - f[0]))),
        "edge_frac_w_lt_0.5": float((w[edge_F] < 0.5).mean()),
        "s_hat": s_hat,
        "global_frac_w_lt_0.5": sc["frac_w_lt_0.5"],
        "roi_net_luma_codes": int(np.unique(np.round(luma[roi])).size),
        "roi_net_luma_range": [float(luma[roi].min()), float(luma[roi].max())],
        "saved_w_net_max_abs_diff": float(np.abs(saved["w_net"].astype(np.float64) - w).max()),
        "_e": e, "_f": f, "_p": p,
    }


def compare_runs(cfg: Config, stem: str, runs: list[str], baseline_run: str, tag: str) -> Path:
    out = cfg.out_dir / "diag" / stem.rsplit("_", 1)[-1]
    roi = np.load(out / "roi_mask.npy")
    _, base_det, _ = _run_detector(run_dir(cfg, baseline_run), stem)
    r = base_det["scalars"]["r"]
    edge_F = (base_det["q"] / base_det["scalars"]["s_hat"] > Q_EXCLUDE) & ~ndimage.binary_dilation(roi, disk(r))
    rows = [run_metrics(cfg, stem, run, roi, edge_F) for run in [baseline_run] + [x for x in runs if x != baseline_run]]
    base_edge = rows[0]["edge_frac_w_lt_0.5"]
    for row in rows:
        row["edge_retention_ratio"] = row["edge_frac_w_lt_0.5"] / base_edge
        row["edge_retention_ok"] = row["edge_retention_ratio"] >= 0.95

    cols = [("run", "s"), ("exposure_mode", "s"), ("dither", "s"), ("roi_frac_w_lt_0.9", ".3f"),
            ("roi_frac_w_lt_0.5", ".3f"), ("roi_median_q_over_s", ".2f"), ("roi_median_q_abs", ".2e"),
            ("roi_p95_q_abs", ".2e"), ("roi_rms_e", ".2e"), ("roi_amplitude_at_7px", ".2e"),
            ("roi_peak_wavelength_px", ".1f"), ("roi_peak_amplitude", ".2e"), ("edge_frac_w_lt_0.5", ".3f"),
            ("edge_retention_ratio", ".3f"), ("s_hat", ".3e"), ("global_frac_w_lt_0.5", ".3f"),
            ("roi_net_luma_codes", "d")]
    lines = [f"# {tag}: {stem} variant metrics", "",
             f"- floor ROI: `{out / 'roi_mask.npy'}` ({int(roi.sum())} px, fixed from T1)",
             f"- edge set F: q/s_hat > {Q_EXCLUDE:g} in `{baseline_run}`, outside the ROI dilated by r={r} "
             f"({int(edge_F.sum())} px); retention must stay >= 0.95 x baseline",
             "- q in absolute units as well as q/s_hat, since input changes alter s_hat", "",
             "| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        lines.append("| " + " | ".join(format(row[c], f) if row[c] is not None else "-" for c, f in cols) + " |")
    lines.append("")

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for row in rows:
        ax.loglog(row["_f"][1:], row["_p"][1:], lw=1.2, label=row["run"])
    ax.axvline(1 / STRIPE_WAVELENGTH, color="0.6", ls=":", lw=1)
    ax.set_xlabel("frequency along v [cycles/px]")
    ax.set_ylabel("PSD of e (floor ROI)")
    ax.legend(fontsize=8)
    ax.set_title(f"{stem}: floor residual spectra ({tag})")
    fig.savefig(out / f"{tag}_spectra.png", dpi=100)
    plt.close(fig)

    # Best among runs that change something (a run identical to the baseline, e.g. A0, is not a candidate).
    changed = [row for row in rows[1:] if (row["s_hat"], row["roi_frac_w_lt_0.5"]) != (rows[0]["s_hat"], rows[0]["roi_frac_w_lt_0.5"])]
    ok = [row for row in changed if row["edge_retention_ok"]]
    best = min(ok, key=lambda x: x["roi_frac_w_lt_0.5"]) if ok else None
    if best is not None:
        a = plt.imread(run_dir(cfg, baseline_run) / "viz" / f"{stem}.png")
        b = plt.imread(run_dir(cfg, best["run"]) / "viz" / f"{stem}.png")
        wd = max(a.shape[1], b.shape[1])
        pad = lambda im: np.pad(im, ((0, 0), (0, wd - im.shape[1]), (0, 0)), constant_values=1.0)  # noqa: E731
        plt.imsave(out / f"{tag}_side_by_side.png", np.concatenate([pad(a), pad(b)], axis=0))
        lines += [f"Best changed variant by floor frac(w < 0.5) with edge retention ok: `{best['run']}`.",
                  f"Side-by-side sheets (top: {baseline_run}, bottom: {best['run']}): `{tag}_side_by_side.png`; "
                  f"spectra: `{tag}_spectra.png`.", ""]
    (out / f"{tag}_metrics.md").write_text("\n".join(lines))
    (out / f"{tag}_metrics.json").write_text(json.dumps(
        {"rows": [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows],
         "best": best["run"] if best else None, "n_edge_F": int(edge_F.sum())}, indent=2))
    return out / f"{tag}_metrics.md"


# ---------------------------------------------------------------- generalization (T5)

REGRESSION_DFRAC = 0.01  # global frac(w < 0.5) up by more than 1 percentage point
BORDER_BAND = 19         # px, as in the T3/T4 border analysis (14 + R)


def _band(shape: tuple[int, int], b: int) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[:b] = m[-b:] = True
    m[:, :b] = m[:, -b:] = True
    return m


def generalize(cfg: Config, runs: list[str], baseline_run: str, tag: str) -> Path:
    from .stage_b import image_stems

    out = cfg.out_dir / "diag" / tag
    out.mkdir(parents=True, exist_ok=True)
    stems = image_stems(run_dir(cfg, baseline_run))
    rows = []
    for stem in stems:
        _, base, _ = _run_detector(run_dir(cfg, baseline_run), stem)
        bs = base["scalars"]
        F = base["q"] / bs["s_hat"] > Q_EXCLUDE
        roi_path = cfg.out_dir / "diag" / stem.rsplit("_", 1)[-1] / "roi_mask.npy"
        if roi_path.exists():  # keep the T1-T4 definition for 070
            F &= ~ndimage.binary_dilation(np.load(roi_path), disk(bs["r"]))
        band = _band(base["w"].shape, BORDER_BAND)
        base_edge = float((base["w"][F] < 0.5).mean())
        for run in runs:
            _, det, _ = _run_detector(run_dir(cfg, run), stem)
            sc, w = det["scalars"], det["w"]
            edge = float((w[F] < 0.5).mean())
            row = {
                "stem": stem, "run": run,
                "frac_w_lt_0.5": sc["frac_w_lt_0.5"], "base_frac_w_lt_0.5": bs["frac_w_lt_0.5"],
                "d_frac_w_lt_0.5": sc["frac_w_lt_0.5"] - bs["frac_w_lt_0.5"],
                "d_border_w_lt_0.5": float((w[band] < 0.5).mean() - (base["w"][band] < 0.5).mean()),
                "d_interior_w_lt_0.5": float((w[~band] < 0.5).mean() - (base["w"][~band] < 0.5).mean()),
                "s_hat": sc["s_hat"], "s_hat_ratio": sc["s_hat"] / bs["s_hat"],
                "tau_min_over_tau_noise": sc.get("tau_min_over_tau_noise"),
                "n_F": int(F.sum()), "edge_ratio": edge / base_edge if base_edge > 0 else float("nan"),
            }
            row["regression"] = bool(row["d_frac_w_lt_0.5"] > REGRESSION_DFRAC or row["edge_ratio"] < 0.95)
            rows.append(row)
        # Weight rows (panel 5/6 + histogram + text) stacked: baseline, then each run.
        ims = [plt.imread(run_dir(cfg, r) / "viz" / f"{stem}.png") for r in [baseline_run] + runs]
        halves = [im[im.shape[0] // 2:] for im in ims]
        wd = max(h.shape[1] for h in halves)
        plt.imsave(out / f"{stem}_weights.png", np.concatenate(
            [np.pad(h, ((0, 0), (0, wd - h.shape[1]), (0, 0)), constant_values=1.0) for h in halves], axis=0))

    df_rows = rows
    lines = [f"# {tag}: generalization over {len(stems)} images", "",
             f"- baseline: `{baseline_run}`; runs: " + ", ".join(f"`{r}`" for r in runs),
             f"- edge set F per image: q/s_hat > {Q_EXCLUDE:g} in the baseline (070: outside its floor ROI dilated by r)",
             f"- regression: global frac(w<0.5) up by > {100 * REGRESSION_DFRAC:g} pp, or edge retention < 0.95",
             f"- border band: {BORDER_BAND} px from the image edge", ""]
    for run in runs:
        rr = [x for x in df_rows if x["run"] == run]
        lines += [f"## `{run}`", "",
                  "| image | w<0.5 base -> run | Δ w<0.5 | Δ border | Δ interior | ŝ ratio | τ_min/τ_noise | edge ratio | regression |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for x in rr:
            tm = "-" if x["tau_min_over_tau_noise"] is None else f"{x['tau_min_over_tau_noise']:.2f}"
            lines.append(f"| {x['stem'][-3:]} | {x['base_frac_w_lt_0.5']:.3f} -> {x['frac_w_lt_0.5']:.3f} | "
                         f"{x['d_frac_w_lt_0.5']:+.3f} | {x['d_border_w_lt_0.5']:+.3f} | {x['d_interior_w_lt_0.5']:+.3f} | "
                         f"{x['s_hat_ratio']:.2f} | {tm} | {x['edge_ratio']:.3f} | {'**yes**' if x['regression'] else 'no'} |")
        d = np.array([x["d_frac_w_lt_0.5"] for x in rr])
        e = np.array([x["edge_ratio"] for x in rr])
        lines += ["", f"median Δ w<0.5 {np.median(d):+.3f} (range {d.min():+.3f} .. {d.max():+.3f}); "
                      f"min edge ratio {e.min():.3f}; regressions {sum(x['regression'] for x in rr)}/{len(rr)}", ""]
    lines += [f"Weight-row stacks per image (top: baseline, then the runs in order): `<stem>_weights.png`.", ""]
    (out / f"{tag}_report.md").write_text("\n".join(lines))
    (out / f"{tag}_rows.json").write_text(json.dumps(df_rows, indent=2))
    return out / f"{tag}_report.md"
