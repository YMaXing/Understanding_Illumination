"""Depth-edge channel (TASK_DEPTH_CHANNEL; docs/depth_channel.md): depth discontinuities found Canny-style in the
disparity, their metric jump Delta Z, and the weight w_depth.

The field (kernels, detrended orientation, q, gradient floor) and the candidates (D-3) are geom/candidates.py:
depth_field, depth_candidates. Here: the parameters and the pieces of D-4 / D-5 shared with the calibration
(scripts/calibrate_depth.py): the null grid, the contamination threshold, the two-sided fit at x0 and lambda_hat.
The jump retries, gates, magnitude, weights (D-4 ... D-7) and DepthChannel follow in D3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .candidates import DepthField
from .context import GeometryContext
from .twosided import TwoSidedResult, b_tri, two_sided_fit, window_rms

CALIBRATION_PATH = Path(__file__).with_name("data") / "depth_calibration.json"
CHI2_1_MEDIAN = 0.45493642311957283  # median of chi^2_1


@dataclass(frozen=True)
class DepthParams:
    # D-K, D-1, D-2: field
    disparity: str = "rho_f"          # rho_f (the context's trilateral-filtered disparity) | rho (ablation)
    sigma: float = 1.25               # px, Hessian scale (as the Laplacian channel)
    truncate: float = 4.0
    sigma_w_prior: float = 1.0        # px, DA3 boundary-smear prior; sigma_tot = sqrt(sigma^2 + sigma_w_prior^2)
    sigma_L_factor: float = 4.0       # sigma_L = sigma_L_factor sigma_tot (detrending scale)
    k_g: float = 3.0                  # valid: g_n > k_g sigma_g, sigma_g the Rayleigh scale of g_n over Omega (D2)
    # D-3: candidates
    lobe_half_len: float = 8.0        # px, lobe search along n
    lobe_step: float = 0.25           # px
    k_h: float = 3.0                  # two-sided height: min(|q_p|, |q_m|) > k_h s_q
    s_lo_factor: float = 0.8          # s_lo = s_lo_factor 2 sigma
    sigma_w_max: float = 0.83         # px, P95 of sigma_w_c at unambiguous real occlusions (data/depth_calibration.json;
                                      # prior 3); s_hi = 2 sqrt(sigma^2 + sigma_w_max^2) = 3.0 px
    centre_tol: float = 1.0 / 3.0     # centring: |s_p + s_m| <= centre_tol s_sep (D2 addition)
    symmetry: bool = False            # |q_p| / |q_m| in [1/symmetry_ratio, symmetry_ratio]
    symmetry_ratio: float = 4.0
    k_min: int = 11                   # k_c = max(k_min, ceil(2 sigma_w_c) + 1)
    # D-4: jump
    m: int = 8
    w_t: int = 4
    m_min: int = 4
    max_excluded: float = 0.2
    planarity_factor: float = 3.0
    b_tri_margin: float = 1.25
    null_rms_quantile: float = 0.95
    gap_mode: str = "at_x0"           # at_x0 | min_span
    doublet_coef: float = 1.03        # Delta rho_doublet = coef |q|_lobe rho_sigma(x0) s_sep^2 (thin fallback)
    # D-5: gates
    tau_low: float = 0.02
    tau_high: float = 0.04
    min_component_px: int = 5
    z: float = 3.29                   # two-sided p_fa = 1e-3
    null_stride: int = 4
    lambda_clamp: tuple[float, float] = (0.5, 1.0e4)
    # D-6: magnitude
    smooth_n: int = 5                 # running median of rho_hat+- along contours
    sky_far_frac: float = 0.5         # far window at least this share sky -> Delta Z = inf
    canonical_focal: float = 300.0    # Z = f_n D / canonical_focal
    # D-7: weight
    dZ0: float = 0.10                 # m, knee of w_c = max(w_min, dZ0 / (dZ + dZ0))
    w_min: float = 0.05
    band_min: int = 2


# ---------------------------------------------------------------- D-4 / D-5 pieces (shared with the calibration)

def fit_kwargs(p: DepthParams, ctx: GeometryContext) -> dict:
    """two_sided_fit keywords for depth windows: samples on Omega only and no dense-normal fallback (the depth channel
    never builds the normals), the gap at x0."""
    return dict(m=p.m, w_t=p.w_t, m_min=p.m_min, max_excluded=p.max_excluded, fallback=False, gap_mode=p.gap_mode,
                usable=ctx.omega)


def null_grid(ctx: GeometryContext, p: DepthParams, f: DepthField) -> tuple[np.ndarray, np.ndarray]:
    """The null sample: a stride-p.null_stride grid over Omega (|grad Dd| > 0; the gradient floor is not required),
    x0 = the pixel centre, each pixel's own n. Returns (x0 (N, 2), n (N, 2))."""
    H, W = ctx.shape
    s0 = p.null_stride // 2
    vv, uu = np.mgrid[s0:H:p.null_stride, s0:W:p.null_stride]
    sel = ctx.omega[vv, uu] & (f.grad_norm[vv, uu] > 0)
    return np.stack([uu[sel], vv[sel]], -1).astype(np.float64), f.n[vv[sel], uu[sel]]


@dataclass
class DepthNull:
    eps_hat: float          # median rms/alpha of the null windows (no contamination check)
    q_null: float           # their null_rms_quantile
    tri: tuple
    res: TwoSidedResult     # the null pass with the contamination check, at k = k_min
    z2: np.ndarray          # (Delta rho* / SE)^2 over the valid null windows
    lambda_raw: float       # median(z2) / median(chi^2_1)
    lambda_hat: float       # clamped

    def rms_max(self, p: DepthParams):
        def f(k: np.ndarray, m: int) -> np.ndarray:
            return np.fmax(np.fmax(p.planarity_factor * self.eps_hat, self.q_null),
                           p.b_tri_margin * b_tri(k, m, self.tri))
        return f


def depth_null(ctx: GeometryContext, p: DepthParams, f: DepthField, k: int | None = None) -> DepthNull:
    """Contamination threshold (flag 3: max(planarity_factor eps_hat, Q_{null_rms_quantile}, b_tri_margin b_tri)) and
    lambda_hat_Delta_rho = median((Delta rho*/SE)^2 over the null) / 0.455, clamped to lambda_clamp."""
    k = p.k_min if k is None else k
    xn, en = null_grid(ctx, p, f)
    kw = fit_kwargs(p, ctx)
    cal = window_rms(two_sided_fit(ctx, xn, en, k, rms_max=None, **kw))
    eps_hat = float(np.median(cal)) if cal.size else float("nan")
    q_null = float(np.quantile(cal, p.null_rms_quantile)) if cal.size else float("nan")
    tri = (ctx.det.tri_sigma_s, ctx.det.tri_sigma_r, ctx.det.tri_sigma_g)
    nul = DepthNull(eps_hat=eps_hat, q_null=q_null, tri=tri, res=None, z2=np.empty(0), lambda_raw=float("nan"),
                    lambda_hat=1.0)
    res = two_sided_fit(ctx, xn, en, k, rms_max=nul.rms_max(p), **kw)
    ok = res.p.planar & res.m.planar & (res.d_rho_se > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z2 = (res.d_rho[ok] / res.d_rho_se[ok]) ** 2
    nul.res, nul.z2 = res, z2
    if z2.size:
        nul.lambda_raw = float(np.median(z2) / CHI2_1_MEDIAN)
        nul.lambda_hat = float(np.clip(nul.lambda_raw, *p.lambda_clamp))
    return nul


def significance(d_rho: np.ndarray, se: np.ndarray, lambda_hat: float) -> np.ndarray:
    """|Delta rho*| / (SE sqrt(lambda_hat))."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(d_rho) / (se * math.sqrt(lambda_hat))


def measure_jump(ctx: GeometryContext, p: DepthParams, x0: np.ndarray, n: np.ndarray, k: np.ndarray | int,
                 null: DepthNull) -> TwoSidedResult:
    """Two-sided fit at x0 along n (side + = +n, the near side) at skip k, m = p.m, with the contamination check
    (D-4 first attempt; the m = 4 / constant-fit / doublet retries are D3). d_rho is Delta rho* in p.gap_mode,
    rho_m_gap the far side rho_hat_-(x0)."""
    return two_sided_fit(ctx, x0, n, k, rms_max=null.rms_max(p), **fit_kwargs(p, ctx))


def relative_jump(res: TwoSidedResult) -> np.ndarray:
    """r = Delta rho* / rho_hat_-(x0) = Delta Z / Z_near (D-5)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return res.d_rho / res.rho_m_gap
