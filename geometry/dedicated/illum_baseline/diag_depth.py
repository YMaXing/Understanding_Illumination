"""Depth profile tool (TASK_DEPTH_CHANNEL D5): `python -m illum_baseline diag-depth --stem S --u U --v V`.

Recomputes the context and the depth channel of one image with the run's configuration, takes the D-3 candidate
nearest (U, V) (upright network px) and plots: the rho_f and q profiles along n through it with the lobes marked and
the two side planes extrapolated to x0; the windows actually used on the image; and the candidate's numbers
(Delta rho*, r, Delta Z, Delta rho_doublet / Delta rho*, significance under both nulls, fit modes, reason).
Output: stage_b/<run>/diag_depth/<stem>_u<U>_v<V>.png.
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

from .config import Config  # noqa: E402
from .viz_depth import depth_field_for  # noqa: E402


def _strip(x0, n, s0, s1, w_t, sign):
    et = np.array([-n[1], n[0]])
    a, b = sign * s0, sign * s1
    return np.array([x0 + a * n - w_t * et, x0 + b * n - w_t * et, x0 + b * n + w_t * et, x0 + a * n + w_t * et])


def diag_depth(cfg: Config, rd: Path, stem: str, u: float, v: float, half_crop: int = 32) -> Path:
    from .geom.candidates import depth_candidates
    from .geom.depth import DepthChannel, depth_null, jump
    from .geom.sampling import bilinear

    ctx, f, p = depth_field_for(cfg, stem)
    res = DepthChannel().run(ctx, p)
    df = res.candidates
    if not len(df):
        raise ValueError(f"{stem}: no depth candidates")
    i = int(np.argmin((df["u0"] - u) ** 2 + (df["v0"] - v) ** 2))
    r = df.iloc[i]
    x0, n = np.array([r.u0, r.v0]), np.array([r.n_u, r.n_v])

    # the jump for this candidate alone, with the channel's obstacles (side planes and their slopes)
    c = depth_candidates(ctx, p, field=f)
    nul = depth_null(ctx, p, f)
    P = c.passed
    with np.errstate(invalid="ignore", divide="ignore"):
        r_d = p.doublet_coef * 0.5 * (c.h_p + np.abs(c.h_m)) * c.s_sep**2 / c.rho_s0
    obst = c.x0[P & (r_d > p.tau_low)]
    assert np.allclose(c.x0[i], x0)  # the candidate table follows the D-3 candidate order
    J = jump(ctx, p, x0[None], n[None], np.array([int(r.k_c)]), np.array([r.q_p]), np.array([r.q_m]),
             c.h_p[i:i + 1], c.h_m[i:i + 1], np.array([r.s_sep]), np.array([r.rho_s0]), f.s_q, nul, obst)

    L = int(r.k_c) + p.m + 4
    s = np.arange(-L, L + 0.001, 0.25)
    pts = x0[None] + s[:, None] * n[None]
    one = np.ones(ctx.shape, dtype=bool)
    rho, _ = bilinear(ctx.rho_f, one, pts[:, 0], pts[:, 1])
    qs, _ = bilinear(f.q, one, pts[:, 0], pts[:, 1])

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    a1, a2, a3 = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[:, 2])
    a1.plot(s, rho, "k-", lw=1.5, label="rho_f along n")
    for sgn, key, col in ((1, "p", "tab:red"), (-1, "m", "tab:blue")):
        val = J[f"rho_{key}"][0]
        beta = J[f"beta_{key}"][0]
        if np.isfinite(val):
            ss = np.linspace(0, sgn * L, 50)
            a1.plot(ss, val + (beta @ n) * ss, "--", color=col,
                    label=f"side {'+' if sgn > 0 else '-'} ({J['mode_' + key][0] or 'none'}) to x0: {val:.5g}")
    a1.axvline(0, color="0.5", lw=0.8)
    for kk in (r.k_c, r.k_c + p.m):
        for sg in (1, -1):
            a1.axvline(sg * kk, color="0.8", lw=0.8, ls=":")
    a1.set_ylabel("rho_f")
    a1.legend(fontsize=8)
    a1.set_title(f"{stem}  candidate {i} at ({r.u0:.1f}, {r.v0:.1f}), n = ({n[0]:.2f}, {n[1]:.2f}) toward the near "
                 f"side; k_c {int(r.k_c)} (dotted: window spans)")
    a2.plot(s, qs / f.s_q, "k-", lw=1.2)
    a2.axhline(0, color="0.5", lw=0.8)
    for sp, qv, col in ((r.s_p, r.q_p, "tab:red"), (r.s_m, r.q_m, "tab:blue")):
        if np.isfinite(sp):
            a2.plot([sp], [qv / f.s_q], "o", color=col)
    for lim in (-p.k_h, p.k_h):
        a2.axhline(lim, color="tab:green", ls="--", lw=0.8)
    a2.set_xlim(-p.lobe_half_len - 2, p.lobe_half_len + 2)
    a2.set_xlabel("s along n [px]")
    a2.set_ylabel("q / s_q")
    a2.set_title(f"q profile: lobes at s_p {r.s_p:.2f}, s_m {r.s_m:.2f} (s_sep {r.s_sep:.2f}; s_lo {res.scalars['s_lo']:.2f}, "
                 f"s_hi {res.scalars['s_hi']:.2f}); centring {r.centring:.2f}; dashed: +-k_h")

    # image crop with the windows used
    sa_json = cfg.out_dir / json.loads((rd / f"{stem}.json").read_text())["provenance"]["stage_a_json"] \
        if (rd / f"{stem}.json").exists() else None
    H, W = ctx.shape
    if sa_json is not None:
        img = cv2.cvtColor(cv2.imread(str(sa_json.parent / "srgb" / f"{stem}.png")), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    else:
        img = np.repeat((255 * (ctx.rho_f / ctx.rho_f.max()))[..., None], 3, -1).astype(np.uint8)
    u0, v0 = int(round(r.u0)), int(round(r.v0))
    a3.imshow(img, extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    for sgn, key, col in ((1, "p", "tab:red"), (-1, "m", "tab:blue")):
        mode = J[f"mode_{key}"][0]
        dn = J[f"d_next_{key}"][0]
        if mode in ("plane8", "plane4", ""):
            mm = p.m if mode != "plane4" else p.m_min
            corners = _strip(x0, n, r.k_c, r.k_c + mm, p.w_t, sgn)
        else:
            s_a = math.ceil(res.scalars["s_hi"] / 2) + 1
            corners = _strip(x0, n, s_a, min(dn - s_a, r.k_c + p.m) if np.isfinite(dn) else r.k_c + p.m_min,
                             p.w_t, sgn)
        a3.add_patch(Polygon(corners, closed=True, fill=False, ec=col, lw=1.5, ls="-" if mode else ":"))
    a3.plot([r.u0], [r.v0], "y+", ms=12)
    a3.arrow(r.u0, r.v0, 4 * n[0], 4 * n[1], color="yellow", width=0.3)
    a3.set_xlim(u0 - half_crop, u0 + half_crop)
    a3.set_ylim(v0 + half_crop, v0 - half_crop)
    txt = (f"reason {r.reason}   fit {r.fit_mode} (+ {r.mode_p or '-'}, - {r.mode_m or '-'})\n"
           f"Delta rho* {r.d_rho:.4g} +- {r.d_rho_se:.2g}\n"
           f"r {r.r:.4g}  (r_gate {r.r_gate:.3g})\n"
           f"Delta Z {r.dZ:.4g} m   [{r.dZ_lo:.3g}, {r.dZ_hi:.3g}]\n"
           f"doublet / Delta rho* {r.doublet_ratio:.3f}\n"
           f"signif {r.signif:.3g} (mult)  {r.signif_add:.3g} (add)\n"
           f"lambda_hat {res.scalars['lambda_hat']:.1f}  sigma_geo,rho {res.scalars['sigma_geo_rho']:.2e}\n"
           f"w_c {r.w_c:.3f}  band {r.band}  component px {r.component_px}")
    a3.set_title("windows used (red: + near side, blue: - far side)", fontsize=9)
    a3.text(0.02, -0.02, txt, transform=a3.transAxes, va="top", family="monospace", fontsize=9)

    out = rd / "diag_depth" / f"{stem}_u{int(round(u))}_v{int(round(v))}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out
