"""2x4 diagnostic panels per image, upright frame, rendered only from saved outputs (spec M4).

Panels 1-5 and 7 are in network pixel coordinates (the full-res sRGB is drawn over the same extent);
panel 6 is at the original resolution. Detector quantities are in units of s_hat with fixed limits.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import Config  # noqa: E402
from .stage_a import stage_a_dir  # noqa: E402

LOR_LIM = 20.0
LOGQ_LIM = (-1.0, 3.0)
HIST_BINS = np.logspace(-2, 4, 121)


def render(cfg: Config, rd: Path, stem: str) -> Path:
    from .stage_b import load_upright

    a, meta = load_upright(cfg, rd, stem)
    det, foc, geo = meta["detector"], meta["focal"], meta["geometry"]
    prm = meta["params"]["detector"]
    k_tau, k_eps, w_min = prm["k_tau"], prm["k_eps"], prm["w_min"]
    s_hat = det["s_hat"]
    srgb = cv2.cvtColor(cv2.imread(str(stage_a_dir(cfg) / "srgb" / f"{stem}.png")), cv2.COLOR_BGR2RGB)

    S = a["sky_mask"]
    H_n, W_n = S.shape
    ext = (-0.5, W_n - 0.5, H_n - 0.5, -0.5)
    D = a["D"].astype(np.float64)
    Z = foc["f_n"] * D / foc["canonical_focal"]
    q_s = a["q"].astype(np.float64) / s_hat
    lor_s = a["L_over_rho"].astype(np.float64) / s_hat
    w_net = a["w_net"].astype(np.float64)
    w_full = a["w"].astype(np.float64)

    portrait = H_n > W_n
    fig, axes = plt.subplots(2, 4, figsize=(18, 12.5) if portrait else (22, 9.5), constrained_layout=True)
    ax = axes.ravel()
    fig.suptitle(f"{stem}   run={rd.name}   upright frame", fontsize=13)

    ax[0].imshow(srgb, extent=ext)
    if S.any():
        ax[0].contour(S.astype(float), levels=[0.5], colors="cyan", linewidths=1.0)
    ax[0].set_title(f"sRGB + sky contour (sky {100 * det['sky_frac']:.1f}%)")

    logz = np.ma.masked_where(S, np.log10(np.maximum(Z, 1e-6)))
    cm = plt.get_cmap("turbo").with_extremes(bad="0.6")
    im = ax[1].imshow(logz, cmap=cm, extent=ext)
    fig.colorbar(im, ax=ax[1], fraction=0.046, label="log10 Z [m]")
    ax[1].set_title(f"log10 Z (f_n = {foc['f_n']:.1f} px, {foc['mode']})")

    im = ax[2].imshow(lor_s, cmap="RdBu_r", vmin=-LOR_LIM, vmax=LOR_LIM, extent=ext)
    fig.colorbar(im, ax=ax[2], fraction=0.046, label="L / max(rho_s, rho_floor) / s_hat")
    ax[2].set_title("signed L/rho / s_hat (steps: ± doublets, creases: one sign)")

    with np.errstate(divide="ignore"):
        logq = np.clip(np.log10(q_s), *LOGQ_LIM)
    im = ax[3].imshow(logq, cmap="magma", vmin=LOGQ_LIM[0], vmax=LOGQ_LIM[1], extent=ext)
    ax[3].contour(logq, levels=[np.log10(k_tau), np.log10(k_tau + k_eps)], colors=["cyan", "lime"], linewidths=0.6)
    fig.colorbar(im, ax=ax[3], fraction=0.046, label="log10(q / s_hat)")
    ax[3].set_title(f"log10(q/s_hat), contours at {k_tau:g} (cyan), {k_tau + k_eps:g} (green)")

    im = ax[4].imshow(w_net, cmap="gray", vmin=w_min, vmax=1.0, extent=ext)
    fig.colorbar(im, ax=ax[4], fraction=0.046, label="w")
    ax[4].set_title("w (network resolution)")

    alpha = np.clip((1 - w_full) / (1 - w_min), 0, 1)[..., None]
    over = (srgb.astype(np.float64) * (1 - alpha) + np.array([255.0, 0, 0]) * alpha).astype(np.uint8)
    ax[5].imshow(over)
    ax[5].set_title(f"sRGB x red alpha = (1-w)/(1-w_min), original res {w_full.shape[1]}x{w_full.shape[0]}")

    vals = q_s[~S]
    n_zero = int((vals == 0).sum())
    ax[6].hist(np.clip(vals[vals > 0], HIST_BINS[0], HIST_BINS[-1]), bins=HIST_BINS, color="0.35")
    ax[6].set_xscale("log")
    ax[6].set_yscale("log")
    for x, c, lab in [(1, "k", "1"), (k_tau, "c", "k_tau"), (k_tau + k_eps, "g", "k_tau+k_eps")]:
        ax[6].axvline(x, color=c, ls="--", lw=1, label=f"{lab} = {x:g}")
    ax[6].legend(fontsize=8)
    ax[6].set_xlabel("q / s_hat (non-sky, clipped to [1e-2, 1e4])")
    ax[6].set_title(f"histogram of q/s_hat ({n_zero} px with q = 0 omitted)")

    txt = "\n".join([
        f"s_hat     {s_hat:.3e}" + ("  (clamped)" if det["s_hat_clamped"] else ""),
        f"tau, eps  {det['tau']:.3e}, {det['eps']:.3e}",
        f"rho_floor {det['rho_floor']:.3e}",
        f"r, R      {det['r']}, {det['R']}",
        f"sky_frac  {det['sky_frac']:.4f}",
        f"n_bad     {det['n_bad_depth']}",
        f"w<0.5     {det['frac_w_lt_0.5']:.3f}",
        f"w<0.9     {det['frac_w_lt_0.9']:.3f}",
        f"w<1       {det['frac_w_lt_1']:.3f}",
        "",
        f"f_active  {foc['f_active']:.1f} px ({foc['mode']})",
        f"f unc     {foc['focal_uncertainty']:.1f} px ({100 * foc['focal_rel_uncertainty']:.0f}%)",
        f"f_n       {foc['f_n']:.1f} px",
        f"vfov      {foc['vfov_deg']:.1f} deg",
        f"roll/pitch {foc['roll_deg']:.1f} / {foc['pitch_deg']:.1f} deg",
        f"median Z  {np.median(Z[~S]):.2f} m",
        "",
        f"k_tau {k_tau:g}  k_eps {k_eps:g}  w_min {w_min:g}",
        f"sigma {prm['sigma']:g}  sigma_w {prm['sigma_w_prior']:g}",
        f"s_x, s_y  {geo['s_x']:.5f}, {geo['s_y']:.5f}",
        f"orientation {geo['orientation_tag']}",
    ])
    ax[7].axis("off")
    ax[7].text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=10, transform=ax[7].transAxes)

    for i in (0, 1, 2, 3, 4, 5):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    out = rd / "viz" / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out
