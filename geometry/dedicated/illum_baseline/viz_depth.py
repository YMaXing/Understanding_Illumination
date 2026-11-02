"""Depth sheet: stage_b/<run>/viz_depth/<stem>.png, upright, 3x3 panels + a text box (TASK_DEPTH_CHANNEL D5).

From the saved outputs (<stem>.npz, <stem>.json, <stem>_depth_candidates.parquet, <stem>_depth_null.parquet), except
panel 4 (|grad Dd| and n), which recomputes the depth field from the Stage A cache (the field is not saved).
Panels in network pixel coordinates (the full-res sRGB is drawn over the same extent) except panel 9 (original res).
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config, dense_fit_params, detector_params  # noqa: E402

DZ_LOG_LIM = (-2.0, 1.5)   # log10 m
Q_LIM = 20.0
CHI2_1_MEDIAN = 0.45493642311957283


def depth_field_for(cfg: Config, stem: str):
    """(context, DepthField) recomputed from the Stage A cache with the run's parameters."""
    from .geom.camera import Intrinsics
    from .geom.candidates import depth_field
    from .geom.context import GeometryContext
    from .geom.depth import DepthParams
    from .stage_a import stage_a_dir

    sa = stage_a_dir(cfg)
    meta_a = json.loads((sa / f"{stem}.json").read_text())
    cache = np.load(sa / f"{stem}.npz")
    sky = cache["sky_prob"].astype(np.float64) if "sky_prob" in cache else None
    ctx = GeometryContext(cache["D"], sky, Intrinsics.from_stage_a(meta_a), detector_params(cfg), dense_fit_params(cfg))
    p = DepthParams.from_dict(dataclasses.asdict(cfg.geometry.depth))
    return ctx, depth_field(ctx, p), p


def null_significance(null: pd.DataFrame, sc: dict, prm: dict, model: str) -> np.ndarray:
    ok = null["planar_p"] & null["planar_m"] & (null["d_rho_se"] > 0)
    d, se = null.loc[ok, "d_rho"].to_numpy(float), null.loc[ok, "d_rho_se"].to_numpy(float)
    if model == "additive":
        return np.sqrt(d**2 / (prm["lambda_n"] * se**2 + sc["sigma_geo_rho"] ** 2))
    return np.abs(d) / (se * math.sqrt(sc["lambda_hat"]))


def render_depth(cfg: Config, rd: Path, stem: str) -> Path:
    from .stage_b import load_upright

    a, meta = load_upright(cfg, rd, stem)
    sc = meta["channels"]["depth"]
    prm = meta["params"]["geometry"]["depth"]
    foc, geo = meta["focal"], meta["geometry"]
    cand = pd.read_parquet(rd / f"{stem}_depth_candidates.parquet")
    null = pd.read_parquet(rd / f"{stem}_depth_null.parquet")
    sa_json = cfg.out_dir / meta["provenance"]["stage_a_json"]
    srgb = cv2.cvtColor(cv2.imread(str(sa_json.parent / "srgb" / f"{stem}.png")), cv2.COLOR_BGR2RGB)
    model = sc.get("null_model", prm.get("null_model", "multiplicative"))
    sig_col = "signif_add" if model == "additive" else "signif"

    w = a["depth_w_net"].astype(np.float64)
    H_n, W_n = w.shape
    ext = (-0.5, W_n - 0.5, H_n - 0.5, -0.5)
    portrait = H_n > W_n
    fig = plt.figure(figsize=(21, 19) if portrait else (25, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.62])
    ax = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(9)]
    axt = fig.add_subplot(gs[:, 3])
    fig.suptitle(f"{stem}   run={rd.name}   depth channel   upright frame   null: {model}", fontsize=14)

    # 1. sRGB with accepted edges by log10 dZ (sky edges cyan), gate-rejected D-3 candidates gray
    acc = cand[cand["accepted"]]
    rej = cand[~cand["accepted"] & ~cand["reason"].isin(["polarity", "height", "sharpness", "centring", "symmetry"])]
    ax[0].imshow(srgb, extent=ext)
    ax[0].scatter(rej["u0"], rej["v0"], s=0.5, c="0.6", linewidths=0)
    fin = acc[np.isfinite(acc["dZ"])]
    sky = acc[acc["sky_edge"]]
    s = ax[0].scatter(fin["u0"], fin["v0"], s=1.2, c=np.log10(fin["dZ"].clip(lower=1e-3)), cmap="plasma",
                      vmin=DZ_LOG_LIM[0], vmax=DZ_LOG_LIM[1], linewidths=0)
    ax[0].scatter(sky["u0"], sky["v0"], s=1.2, c="cyan", linewidths=0)
    fig.colorbar(s, ax=ax[0], fraction=0.046, label="log10 Delta Z [m]")
    ax[0].set_title(f"accepted edges by log10 Delta Z ({len(acc)}; cyan: sky {len(sky)}, inf)\n"
                    f"gray: rejected at the gates ({len(rej)})")

    # 2. log10 Z
    Z = foc["f_n"] * a["D"].astype(np.float64) / prm["canonical_focal"]
    sky_mask = a["sky_mask"]
    im = ax[1].imshow(np.ma.masked_where(sky_mask, np.log10(Z)), cmap="viridis", extent=ext)
    fig.colorbar(im, ax=ax[1], fraction=0.046, label="log10 Z [m]")
    ax[1].set_title("log10 Z = log10(f_n D / 300) (sky masked)")

    # 3. signed q / s_q
    q = a["depth_q_over_s"].astype(np.float64)
    im = ax[2].imshow(q, cmap="RdBu_r", vmin=-Q_LIM, vmax=Q_LIM, extent=ext)
    fig.colorbar(im, ax=ax[2], fraction=0.046, label="q / s_q")
    ax[2].set_title(f"q / s_q (s_q {sc['s_q']:.2e})")

    # 4. |grad Dd| (log) with n arrows (recomputed field)
    _, f, _ = depth_field_for(cfg, stem)
    with np.errstate(divide="ignore"):
        lg = np.log10(np.where(f.grad_norm > 0, f.grad_norm, np.nan))
    im = ax[3].imshow(lg, cmap="magma", extent=ext, vmin=np.nanpercentile(lg, 5), vmax=np.nanpercentile(lg, 99.5))
    st = 12
    vv, uu = np.mgrid[st // 2:H_n:st, st // 2:W_n:st]
    ok = f.valid[vv, uu]
    ax[3].quiver(uu[ok], vv[ok], f.n[vv[ok], uu[ok], 0], -f.n[vv[ok], uu[ok], 1], color="cyan", scale=40, width=0.002)
    fig.colorbar(im, ax=ax[3], fraction=0.046, label="log10 |grad Dd|")
    ax[3].set_title(f"|grad Dd| and n (toward the near side) on the gradient-floor valid pixels\n"
                    f"(valid {f.valid.mean():.2f} of the image)")

    # 5. s_sep histogram
    reached = cand[~cand["reason"].isin(["polarity", "height"])]
    ax[4].hist(reached["s_sep"].dropna(), bins=np.arange(0, 16.25, 0.25), color="tab:blue")
    ax[4].axvline(sc["s_lo"], color="tab:red", ls="--", label=f"s_lo {sc['s_lo']:.2f}")
    ax[4].axvline(sc["s_hi"], color="tab:red", ls="-", label=f"s_hi {sc['s_hi']:.2f}")
    ax[4].set_yscale("log")
    ax[4].set_xlabel("s_sep [px]")
    ax[4].legend(fontsize=8)
    ax[4].set_title(f"lobe separation of the {len(reached)} candidates past polarity and height")

    # 6. r histogram
    rr = cand.loc[cand["reason"].isin(["accepted", "rel", "component", "signif"]), "r"].to_numpy(float)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size:
        ax[5].hist(np.clip(rr, 1e-4, 1e3), bins=np.logspace(-4, 3, 71), color="tab:blue")
    ax[5].axvline(prm["tau_low"], color="tab:red", ls="--", label=f"tau_low {prm['tau_low']:g}")
    ax[5].axvline(prm["tau_high"], color="tab:red", label=f"tau_high {prm['tau_high']:g}")
    ax[5].set_xscale("log")
    ax[5].set_yscale("log")
    ax[5].set_xlabel("r = Delta Z / Z_near")
    ax[5].legend(fontsize=8)
    ax[5].set_title("relative jump (finite r, candidates reaching the gates)")

    # 7. significance: null vs candidates
    sn = null_significance(null, sc, prm, model)
    scd = cand.loc[cand["reason"].isin(["accepted", "rel", "component", "signif"]) & ~cand["sky_edge"]
                   & ~cand["thin"], sig_col].to_numpy(float)
    bins = np.logspace(-3, 4, 141)
    for x, lab, col in ((sn, f"null ({sn.size})", "0.4"), (scd, f"candidates ({scd.size})", "tab:red")):
        x = np.clip(x[np.isfinite(x)], bins[0], bins[-1])
        ax[6].hist(x, bins=bins, density=True, histtype="step", color=col, lw=1.5, label=lab)
    ax[6].axvline(prm["z"], color="tab:green", ls="--", label=f"gate z = {prm['z']:g}")
    ax[6].set_xscale("log")
    ax[6].set_yscale("log")
    ax[6].set_xlabel(f"{sig_col} = " + ("|Delta rho*| / sqrt(lambda_n SE^2 + sigma_geo^2)" if model == "additive"
                                        else "|Delta rho*| / (SE sqrt(lambda_hat))"))
    ax[6].legend(fontsize=8)
    ax[6].set_title(f"significance ({model} null; plane fits)")

    # 8. w_depth
    im = ax[7].imshow(w, cmap="gray", vmin=prm["w_min"], vmax=1.0, extent=ext)
    fig.colorbar(im, ax=ax[7], fraction=0.046, label="w_depth")
    ax[7].set_title(f"w_depth (share < 0.5: {(w < 0.5).mean():.3f}, < 0.9: {(w < 0.9).mean():.3f})")

    # 9. combined w over sRGB
    w_full = a["w"].astype(np.float64)
    w_clip = meta["combined"]["w_min_clip"]
    alpha = np.clip((1 - w_full) / (1 - w_clip), 0, 1)[..., None]
    ax[8].imshow((srgb.astype(np.float64) * (1 - alpha) + np.array([255.0, 0, 0]) * alpha).astype(np.uint8))
    ax[8].set_title(f"combined w ({' + '.join(meta['combined']['channels'])}) as red alpha, original res")

    for i in (0, 1, 2, 3, 7, 8):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    for a_ in ax:
        a_.title.set_fontsize(9)

    fu = foc["focal_uncertainty"]
    txt = "\n".join([
        f"run          {rd.name}",
        f"null model   {model}",
        f"s_q          {sc['s_q']:.2e}",
        f"sigma_g      {sc['sigma_g']:.2e} (floor x{prm['k_g']:g})",
        f"lambda_hat   {sc['lambda_hat']:.1f} (raw {sc['lambda_hat_raw']:.1f})",
        f"sigma_geo,rho {sc['sigma_geo_rho']:.2e} (additive; lambda_n {prm['lambda_n']:g})",
        f"sigma_w,max  {sc['sigma_w_max']:.2f} px  (s_lo {sc['s_lo']:.2f}, s_hi {sc['s_hi']:.2f})",
        f"null         {sc['n_null_valid']}/{sc['n_null']} valid",
        "",
        "candidates by stage",
        f"  crossings  {sc['n_cross']}",
        f"  floor      {sc['n_floor']}",
        f"  merged     {sc['n_cand']}",
        f"  polarity   {sc['n_polarity']}",
        f"  height     {sc['n_height']}",
        f"  sharpness  {sc['n_sharpness']}",
        f"  centring   {sc['n_centring']}",
        f"  sky_near   -{sc['n_sky_near']}",
        f"  sign       {sc['n_sign_ok']}",
        f"  signif     {sc['n_signif_ok']}  (additive {sc['n_signif_ok_add']})",
        f"  rel        {sc['n_rel_ok']}",
        f"  component  {sc['n_accepted']}",
        f"  fit modes  m8 {sc['n_fit_m8']}, m4 {sc['n_fit_m4']},",
        f"             const {sc['n_fit_const']}, doublet {sc['n_fit_doublet']}",
        f"  accepted: thin {sc['n_thin_accepted']}, sky {sc['n_sky_edge_accepted']},",
        f"            lower bound {sc['n_lower_bound_accepted']}",
        "",
        f"r_gate P10/50/90 {sc['r_gate_P10']:.3f}/{sc['r_gate_P50']:.3f}/{sc['r_gate_P90']:.3f}",
        f"signif binding   {sc['share_signif_binding']:.2f}",
        "",
        f"w_depth < 0.5  {sc['frac_w_lt_0.5']:.3f}",
        f"w_depth < 0.9  {sc['frac_w_lt_0.9']:.3f}",
        f"w < 0.5 (combined) {meta['combined']['frac_w_lt_0.5']:.3f}",
        "",
        "parameters",
        f"  sigma {prm['sigma']:g}  k_min {prm['k_min']}  m {prm['m']}  w_t {prm['w_t']}",
        f"  tau {prm['tau_low']:g}/{prm['tau_high']:g}  z {prm['z']:g}  centre_tol {prm['centre_tol']:g}",
        f"  dZ0 {prm['dZ0']:g} m  w_min {prm['w_min']:g}",
        "",
        f"f_active    {foc['f_active']:.1f} px",
        f"f unc       {fu:.1f} px ({100 * foc['focal_rel_uncertainty']:.0f}%)",
        f"f_n         {foc['f_n']:.1f} px",
        f"orientation {geo['orientation_tag']}",
        f"runtime     {sc.get('context_s', float('nan')):.1f} s context + {sc.get('runtime_s', float('nan')):.1f} s depth",
    ])
    axt.axis("off")
    axt.text(0.02, 0.99, txt, va="top", ha="left", family="monospace", fontsize=10.5, transform=axt.transAxes)

    out = rd / "viz_depth" / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out
