"""Fold sheet: stage_b/<run>/viz_fold/<stem>.png, upright, 3x3 panels + a text box (TASK_FOLD_CHANNEL F4).

Rendered only from saved outputs (<stem>.npz, <stem>.json, <stem>_fold_candidates.parquet, <stem>_fold_null.parquet).
Panels in network pixel coordinates (the full-res sRGB is drawn over the same extent) except panel 9 (original res).
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from .config import Config  # noqa: E402

DTHETA_LIM = (0.0, 120.0)        # deg, colour limits for fold angles
SIGMA_LOG_LIM = (-2.0, 1.0)      # log10 deg
RATE_LOG_LIM = (-4.0, 0.0)       # log10 rad/px
HIST_BINS = np.logspace(-3, 4, 141)
STAT = {"multiplicative": ("T_over_lambda", "T / lambda_hat"), "additive": ("T_add", "T_add (global sigma_geo)"),
        "additive_local": ("T_add_local", "T_add_local (tile sigma_geo)"),
        "additive_cand": ("T_add_cand", "T_add_cand (candidate sigma_geo; null: global T_add)")}
NULL_STAT = {"multiplicative": "T", "additive": "T_add", "additive_local": "T_add_local", "additive_cand": "T_add"}


def _segments(df: pd.DataFrame, half: float) -> np.ndarray:
    """Short segments along each candidate's fold line (perpendicular to e1), centred at x0."""
    t = np.stack([-df["e1_v"].to_numpy(), df["e1_u"].to_numpy()], -1)
    c = df[["u0", "v0"]].to_numpy()
    return np.stack([c - half * t, c + half * t], 1)


def render_fold(cfg: Config, rd: Path, stem: str) -> Path:
    from .stage_b import load_upright

    a, meta = load_upright(cfg, rd, stem)
    fo = meta["channels"]["fold"]
    prm = meta["params"]["geometry"]["fold"]
    foc, geo = meta["focal"], meta["geometry"]
    cand = pd.read_parquet(rd / f"{stem}_fold_candidates.parquet")
    null = pd.read_parquet(rd / f"{stem}_fold_null.parquet")
    sa_json = cfg.out_dir / meta["provenance"]["stage_a_json"]
    srgb = cv2.cvtColor(cv2.imread(str(sa_json.parent / "srgb" / f"{stem}.png")), cv2.COLOR_BGR2RGB)

    w_fold = a["fold_w_net"].astype(np.float64)
    H_n, W_n = w_fold.shape
    ext = (-0.5, W_n - 0.5, H_n - 0.5, -0.5)
    w_min = prm["w_min"]
    model = fo.get("null_model", prm.get("null_model", "multiplicative"))
    stat_col, stat_name = STAT[model]
    T_crit = fo.get("T_crit", 2 * math.log(1 / prm["p_fa"]))

    portrait = H_n > W_n
    fig = plt.figure(figsize=(21, 19) if portrait else (25, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.62])
    ax = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(9)]
    axt = fig.add_subplot(gs[:, 3])
    fig.suptitle(f"{stem}   run={rd.name}   fold channel   upright frame   null: {model}", fontsize=14)

    # 1. sRGB with accepted segments coloured by dtheta, veto-rejected gray, no_veto dashed
    ax[0].imshow(srgb, extent=ext)
    acc = cand[cand["accepted"]]
    vetoed = cand[cand["gate"] & ~cand["veto_pass"]]
    norm = plt.Normalize(*DTHETA_LIM)
    if len(vetoed):
        ax[0].add_collection(LineCollection(_segments(vetoed, 0.7), colors="0.6", linewidths=1.2))
    capped = "narrow_face" in cand.columns  # TASK_FOLD_V2 capped veto
    if capped:  # dotted: accepted as narrow_face (veto undecidable, both k1 sides strictly planar)
        solid, dashed = acc[~acc["narrow_face"]], acc[acc["narrow_face"]]
        und = cand[cand["veto_undecidable"]]
        if len(und):
            ax[0].add_collection(LineCollection(_segments(und, 0.7), colors="0.3", linewidths=1.0))
    else:
        solid, dashed = acc[~acc["no_veto"]], acc[acc["no_veto"]]
    lc = LineCollection(_segments(solid, 0.7), cmap="plasma", norm=norm, linewidths=1.6)
    lc.set_array(np.degrees(solid["dtheta"].to_numpy()))
    ax[0].add_collection(lc)
    if len(dashed):
        lcd = LineCollection(_segments(dashed, 1.2), cmap="plasma", norm=norm, linewidths=1.6,
                             linestyles=":" if capped else "--")
        lcd.set_array(np.degrees(dashed["dtheta"].to_numpy()))
        ax[0].add_collection(lcd)
    fig.colorbar(lc, ax=ax[0], fraction=0.046, label="dtheta(k1) [deg]")
    if capped:
        ax[0].set_title(f"accepted folds by dtheta ({len(acc)}; dotted: narrow_face {len(dashed)})\n"
                        f"gray: vetoed {len(vetoed)} (dark: veto_undecidable {len(und)})")
    else:
        ax[0].set_title(f"accepted folds by dtheta ({len(acc)}; dashed: no_veto {len(dashed)})\ngray: vetoed {len(vetoed)}")

    # 2. normal map
    n = a["n_hat"].astype(np.float64)
    nv = a["normal_valid"]
    rgb = np.where(nv[..., None], np.clip((n + 1) / 2, 0, 1), 0.5)
    ax[1].imshow(rgb, extent=ext)
    ax[1].set_title("normal map RGB = (n+1)/2; gray invalid\nn toward camera; x right, y down, z forward")

    # 3. sigma_theta
    with np.errstate(divide="ignore", invalid="ignore"):
        st = np.log10(np.degrees(a["sigma_theta"].astype(np.float64)))
    im = ax[2].imshow(np.ma.masked_where(~nv | ~np.isfinite(st), st), cmap="viridis", vmin=SIGMA_LOG_LIM[0],
                      vmax=SIGMA_LOG_LIM[1], extent=ext)
    fig.colorbar(im, ax=ax[2], fraction=0.046, label="log10 sigma_theta [deg]")
    ax[2].set_title(f"sigma_theta (median {np.nanmedian(np.degrees(a['sigma_theta'][nv].astype(float))):.3f} deg)")

    # 4. rotation rate with all candidates
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log10(a["fold_r_n"].astype(np.float64))
    im = ax[3].imshow(np.ma.masked_invalid(lr), cmap="magma", vmin=RATE_LOG_LIM[0], vmax=RATE_LOG_LIM[1], extent=ext)
    ax[3].scatter(cand["u0"], cand["v0"], s=0.4, c="cyan", linewidths=0)
    fig.colorbar(im, ax=ax[3], fraction=0.046, label="log10 r_n [rad/px]")
    ax[3].set_title(f"rotation rate r_n, all {len(cand)} candidates (threshold {fo['rate_threshold']:.2e})")

    # 5. dtheta at candidates on a dark background
    ax[4].set_facecolor("black")
    vc = cand[cand["valid"]]
    sc = ax[4].scatter(vc["u0"], vc["v0"], c=np.degrees(vc["dtheta"]), s=1.2, cmap="plasma", norm=norm, linewidths=0)
    ax[4].set_xlim(ext[0], ext[1])
    ax[4].set_ylim(ext[2], ext[3])
    ax[4].set_aspect("equal")
    fig.colorbar(sc, ax=ax[4], fraction=0.046, label="dtheta(k1) [deg]")
    ax[4].set_title(f"dtheta(k1) at the {len(vc)} valid candidates")

    # 6. histograms of the gate statistic: null vs candidates, chi2_2 density, gate line
    tn = null[NULL_STAT[model]][null["valid"]].to_numpy(dtype=float)
    if model == "multiplicative":
        tn = tn / fo["lambda_hat"]
    tc = cand[stat_col][cand["valid"]].to_numpy(dtype=float)
    for x, lab, col in ((tn, f"null ({tn.size})", "0.4"), (tc, f"candidates ({tc.size})", "tab:red")):
        x = np.clip(x[np.isfinite(x)], HIST_BINS[0], HIST_BINS[-1])
        ax[5].hist(x, bins=HIST_BINS, density=True, histtype="step", color=col, lw=1.5, label=lab)
    xs = np.logspace(-3, 2, 200)
    ax[5].plot(xs, 0.5 * np.exp(-xs / 2), "k:", lw=1, label="chi2_2 density")
    ax[5].axvline(T_crit, color="tab:green", ls="--", label=f"gate {T_crit:.1f}")
    ax[5].set_xscale("log")
    ax[5].set_yscale("log")
    ax[5].set_ylim(1e-7, 10)
    ax[5].set_xlabel(stat_name)
    ax[5].legend(fontsize=8)
    ax[5].set_title(f"gate statistic ({model} null)")

    # 7. veto: C vs dtheta (fixed veto) or C / C_min_eff vs dtheta (capped veto)
    if capped:
        dec = cand[cand["gate"] & cand["decidable"] & ~cand["no_veto"]]
        ratio = dec["C"] / dec["C_min_eff"]
        ax[6].scatter(np.degrees(dec["dtheta"]), ratio, s=3, c=np.where(dec["veto_pass"], "tab:blue", "0.5"),
                      linewidths=0, label=f"decided ({len(dec)})")
        for col, key, mk in (("tab:green", "narrow_face", "^"), ("tab:red", "veto_undecidable", "x")):
            u_ = cand[cand[key]]
            ax[6].scatter(np.degrees(u_["dtheta"]), np.full(len(u_), 0.05), s=8, c=col, marker=mk, linewidths=0.6,
                          label=f"{key} ({len(u_)}, at 0.05)")
        hi = dec[dec["C_high"]]
        ax[6].axhline(1.0, color="tab:red", ls="--", label="C = C_min_eff")
        ax[6].set_ylabel("C / C_min_eff")
        ax[6].set_title(f"capped veto: {len(dec)} decided, C_high {len(hi)}\n"
                        f"narrow_face {int(cand['narrow_face'].sum())}, veto_undecidable {int(cand['veto_undecidable'].sum())}")
    else:
        g = cand[cand["gate"] & ~cand["no_veto"]]
        ax[6].scatter(np.degrees(g["dtheta"]), g["C"], s=3, c=np.where(g["veto_pass"], "tab:blue", "0.5"), linewidths=0)
        ax[6].axhline(prm["C_min"], color="tab:red", ls="--", label=f"C_min {prm['C_min']:g}")
        ax[6].set_ylabel("C = dtheta(k1) / dtheta(k2)")
        ax[6].set_title(f"concentration veto: {len(g)} gate-passed with k2 windows\n({int(cand['no_veto'].sum())} no_veto)")
    ax[6].set_xlabel("dtheta(k1) [deg]")
    ax[6].set_ylim(0, 1.5)
    ax[6].legend(fontsize=7)

    # 8. w_fold
    im = ax[7].imshow(w_fold, cmap="gray", vmin=w_min, vmax=1.0, extent=ext)
    fig.colorbar(im, ax=ax[7], fraction=0.046, label="w_fold")
    ax[7].set_title(f"w_fold (share < 0.5: {(w_fold < 0.5).mean():.3f})")

    # 9. combined w over sRGB
    w_full = a["w"].astype(np.float64)
    w_clip = meta["combined"]["w_min_clip"]
    alpha = np.clip((1 - w_full) / (1 - w_clip), 0, 1)[..., None]
    ax[8].imshow((srgb.astype(np.float64) * (1 - alpha) + np.array([255.0, 0, 0]) * alpha).astype(np.uint8))
    ax[8].set_title(f"combined w ({' + '.join(meta['combined']['channels'])}) as red alpha, original res")

    for i in (0, 1, 2, 3, 4, 7, 8):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    for a_ in ax:
        a_.title.set_fontsize(9)

    sig_loc = (f"{math.degrees(fo['sigma_geo_local_min']):.2f}-{math.degrees(fo['sigma_geo_local_max']):.2f} deg "
               f"({fo['sigma_geo_local_tiles']} tiles)" if fo.get("sigma_geo_local_tiles") else "-")
    txt = "\n".join([
        f"run         {rd.name}",
        f"null model  {model} (gate arm)",
        f"lambda_hat  {fo['lambda_hat']:.1f} (raw {fo['lambda_hat_raw']:.1f})",
        f"sigma_geo   {math.degrees(fo['sigma_geo']):.2f} deg (global)",
        f"sigma_geo   {sig_loc} (local)",
        f"lambda_n    {prm['lambda_n']:g}",
        f"eps_hat     {fo['eps_hat']:.2e}",
        f"null rms Q  {fo['null_rms_q']:.2e}",
        f"null        {fo['n_null_valid']}/{fo['n_null']} valid",
        "",
        "candidates by stage",
        f"  cand      {fo['n_cand']}",
        f"  valid     {fo['n_valid']}",
        f"  gate      {fo['n_gate']}  (mult {fo['n_gate_mult']}, add {fo['n_gate_add']},",
        f"            add_local {fo.get('n_gate_add_local', '-')})",
        *([f"  veto      {fo['n_veto']}  (k2 invalid {fo['n_no_veto']}: not accepted",
           f"            on that ground; narrow_face rule)",
           f"  veto mode capped: narrow_face {fo['n_narrow_face']},",
           f"            veto_undecidable {fo['n_veto_undecidable']}, C_high {fo['n_C_high']}"]
          if fo.get("veto_mode") == "capped" else [f"  veto      {fo['n_veto']}  (no_veto {fo['n_no_veto']})"]),
        f"  component {fo['n_accepted']}",
        f"  fallback  {fo['n_fallback']}",
        f"  depth_jump (accepted) {fo['n_accepted_depth_jump']}",
        "",
        f"w_fold < 0.5  {fo['frac_w_lt_0.5']:.3f}",
        f"w_fold < 0.9  {fo['frac_w_lt_0.9']:.3f}",
        f"w < 0.5 (combined) {meta['combined']['frac_w_lt_0.5']:.3f}",
        "",
        "parameters",
        f"  k_min {prm['k_min']}  m {prm['m']}  w_t {prm['w_t']}",
        f"  p_fa {prm['p_fa']:g}  C_min {prm['C_min']:g}" + (
            f"  veto capped coef {prm['veto_curv_coef']:g} max {prm['veto_curv_max']:g}"
            if prm.get("veto_mode") == "capped" else ""),
        f"  J_min {prm['J_min']:g}  J0 {prm['J0']:g}  w_min {w_min:g}",
        f"  b_tri_margin {prm['b_tri_margin']:g}",
        f"  dense fit g_init {meta['params']['geometry']['dense_fit']['g_init']}",
        "",
        f"f_active    {foc['f_active']:.1f} px",
        f"f unc       {foc['focal_uncertainty']:.1f} px ({100 * foc['focal_rel_uncertainty']:.0f}%)",
        f"f_n         {foc['f_n']:.1f} px",
        f"orientation {geo['orientation_tag']}",
        f"runtime     {fo.get('context_s', float('nan')):.1f} s context + {fo.get('runtime_s', float('nan')):.1f} s fold",
    ])
    axt.axis("off")
    axt.text(0.02, 0.99, txt, va="top", ha="left", family="monospace", fontsize=10.5, transform=axt.transAxes)

    out = rd / "viz_fold" / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out
