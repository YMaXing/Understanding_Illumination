"""Fold profile tool (TASK_FOLD_CHANNEL F4): `python -m illum_baseline diag-fold --stem S --u U --v V`.

Recomputes the context and the fold channel of one image with the run's configuration, takes the candidate nearest
(U, V) (upright network px) and plots: the dense normal-angle profile along e1 with the rounding width, k1 and k2;
the k1 and k2 windows on the image with the two-sided normals n+- (image-plane components) as arrows; and the
candidate's numbers (dtheta(k1), dtheta(k2), C, T / lambda_hat, T_add, T_add_local, J, w_c, Delta rho*, reason).
Output: stage_b/<run>/diag_fold/<stem>_u<U>_v<V>.png.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from .config import Config, dense_fit_params, detector_params  # noqa: E402


def _window_corners(x0, e1, k, m, w_t, sign):
    et = np.array([-e1[1], e1[0]])
    s0, s1 = sign * k, sign * (k + m)
    return np.array([x0 + s0 * e1 - w_t * et, x0 + s1 * e1 - w_t * et, x0 + s1 * e1 + w_t * et,
                     x0 + s0 * e1 + w_t * et])


def diag_fold(cfg: Config, rd: Path, stem: str, u: float, v: float, half_crop: int = 40) -> Path:
    from .geom.camera import Intrinsics
    from .geom.context import GeometryContext
    from .geom.fold import FoldChannel, FoldParams
    from .geom.sampling import bilinear
    from .geom.twosided import two_sided_fit
    from .stage_a import stage_a_dir

    sa = stage_a_dir(cfg)
    meta_a = json.loads((sa / f"{stem}.json").read_text())
    cache = np.load(sa / f"{stem}.npz")
    sky = cache["sky_prob"].astype(np.float64) if "sky_prob" in cache else None
    ctx = GeometryContext(cache["D"], sky, Intrinsics.from_stage_a(meta_a), detector_params(cfg), dense_fit_params(cfg))
    import dataclasses

    p = FoldParams.from_dict(dataclasses.asdict(cfg.geometry.fold))
    res = FoldChannel().run(ctx, p)
    df, sc = res.candidates, res.scalars
    if not len(df):
        raise ValueError(f"{stem}: no fold candidates")
    i = int(np.argmin((df["u0"] - u) ** 2 + (df["v0"] - v) ** 2))
    r = df.iloc[i]
    x0, e1 = np.array([r.u0, r.v0]), np.array([r.e1_u, r.e1_v])
    k1, k2 = int(r.k1), int(r.k2)

    # dense normal-angle profile along e1
    L = k2 + p.m + 4
    s = np.arange(-L, L + 0.001, 0.25)
    pts = x0[None] + s[:, None] * e1[None]
    n, ok = bilinear(ctx.n_hat, ctx.normal_valid, pts[:, 0], pts[:, 1])
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    ref = n[np.nonzero(ok)[0][0]] if ok.any() else np.array([0, 0, -1.0])
    ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(n, ref), axis=-1), n @ ref))
    r2 = two_sided_fit(ctx, x0[None], e1[None], k2, p.m, p.w_t, rms_max=None, fallback=False,
                       m_min=p.m_min, max_excluded=p.max_excluded)

    fig = plt.figure(figsize=(17, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 0.75])
    ax0, ax1, axt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    fig.suptitle(f"{stem}  run={rd.name}  fold candidate nearest ({u:g}, {v:g}): x0 = ({r.u0:.1f}, {r.v0:.1f}), "
                 f"e1 = ({r.e1_u:.2f}, {r.e1_v:.2f}); side + is +e1", fontsize=12)

    ax0.plot(s, ang, "k-", lw=1.2, label="angle(n(s), n(-L)) [deg]")
    ax0.axvspan(-r.width / 2, r.width / 2, color="tab:orange", alpha=0.25, label=f"rounding width {r.width:.1f} px")
    for sign in (1, -1):
        ax0.axvspan(sign * k1, sign * (k1 + p.m), color="tab:blue", alpha=0.15,
                    label=f"k1 windows [{k1}, {k1 + p.m}]" if sign == 1 else None)
        ax0.axvspan(sign * k2, sign * (k2 + p.m), color="tab:green", alpha=0.12,
                    label=f"k2 windows [{k2}, {k2 + p.m}]" if sign == 1 else None)
    ax0.axhline(math.degrees(r.dtheta), color="tab:red", ls="--", lw=1, label=f"dtheta(k1) {math.degrees(r.dtheta):.1f}")
    if np.isfinite(r.dtheta_k2):
        ax0.axhline(math.degrees(r.dtheta_k2), color="tab:green", ls=":", lw=1,
                    label=f"dtheta(k2) {math.degrees(r.dtheta_k2):.1f}")
    ax0.set_xlabel("s along e1 [px]")
    ax0.set_ylabel("dense normal angle [deg]")
    ax0.legend(fontsize=8, loc="best")
    ax0.set_title("dense normal-angle profile through the candidate")

    srgb = cv2.cvtColor(cv2.imread(str(sa / "srgb" / f"{stem}.png")), cv2.COLOR_BGR2RGB)
    H, W = ctx.shape
    srgb = cv2.resize(srgb, (W, H), interpolation=cv2.INTER_AREA)
    ax1.imshow(srgb, extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    for k, col in ((k1, "tab:blue"), (k2, "tab:green")):
        for sign in (1, -1):
            ax1.add_patch(Polygon(_window_corners(x0, e1, k, p.m, p.w_t, sign), closed=True, fill=False, ec=col, lw=1.3))
    ax1.plot(*x0, "r+", ms=10)
    for sign, tag, col in ((1, "p", "tab:red"), (-1, "m", "tab:purple")):
        nn = np.array([r[f"n_{tag}_x"], r[f"n_{tag}_y"], r[f"n_{tag}_z"]])
        c = x0 + sign * (k1 + p.m / 2) * e1
        ax1.arrow(c[0], c[1], 12 * nn[0], 12 * nn[1], color=col, width=0.4, head_width=2.0,
                  length_includes_head=True)
        ax1.annotate(f"n{'+' if sign > 0 else '-'} = ({nn[0]:.2f}, {nn[1]:.2f}, {nn[2]:.2f})", c, color=col, fontsize=8,
                     xytext=(4, 4 * sign), textcoords="offset points")
    ax1.set_xlim(x0[0] - half_crop, x0[0] + half_crop)
    ax1.set_ylim(x0[1] + half_crop, x0[1] - half_crop)
    ax1.set_title("k1 (blue) and k2 (green) windows; n+- image-plane components x 12 px")

    deg = math.degrees
    txt = "\n".join([
        f"reason         {r.reason}",
        f"dtheta(k1)     {deg(r.dtheta):.2f} deg",
        f"dtheta(k2)     {deg(r.dtheta_k2):.2f} deg" if np.isfinite(r.dtheta_k2) else "dtheta(k2)     - (not gate-passed)",
        f"C              {r.C:.3f}" + ("  (no_veto)" if r.no_veto else ""),
        f"T / lambda_hat {r.T_over_lambda:.2f}   (gate {sc['T_crit']:.1f})",
        f"T_add          {r.T_add:.2f}",
        f"T_add_local    {r.T_add_local:.2f}",
        f"null model     {sc['null_model']}",
        f"J              {r.J:.3f}  [{r.J_lo:.3f}, {r.J_hi:.3f}] (focal)",
        f"w_c            {r.w_c:.3f}  band {r.band}",
        f"k1, k2         {k1}, {k2}",
        f"width          {r.width:.2f} px",
        f"sides          + {'planar m=%d' % r.m_used_p if r.planar_p else ('fallback' if r.fallback_p else 'invalid')},"
        f" - {'planar m=%d' % r.m_used_m if r.planar_m else ('fallback' if r.fallback_m else 'invalid')}",
        f"sigma_theta +- {deg(r.sigma_theta_p):.3f}, {deg(r.sigma_theta_m):.3f} deg",
        f"Delta rho*     {r.d_rho:.3e} (SE {r.d_rho_se:.1e})",
        f"dZ/Z_near      {r.dZ_rel:.4f}  planes_cross {bool(r.planes_cross)}",
        f"depth_jump     {bool(r.depth_jump)}",
        f"convexity      {int(r.convexity):+d}",
        "",
        f"k2 normals     + ({r2.p.nhat[0, 0]:.2f}, {r2.p.nhat[0, 1]:.2f}, {r2.p.nhat[0, 2]:.2f})",
        f"               - ({r2.m.nhat[0, 0]:.2f}, {r2.m.nhat[0, 1]:.2f}, {r2.m.nhat[0, 2]:.2f})",
        "",
        f"lambda_hat {sc['lambda_hat']:.1f}  sigma_geo {deg(sc['sigma_geo']):.2f} deg",
        f"sigma_geo_local here {deg(r.sigma_geo_local):.2f} deg",
    ])
    axt.axis("off")
    axt.text(0.0, 0.99, txt, va="top", ha="left", family="monospace", fontsize=10, transform=axt.transAxes)
    out = rd / "diag_fold" / f"{stem}_u{u:g}_v{v:g}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out
