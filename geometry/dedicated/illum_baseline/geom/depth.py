"""Depth-edge channel (TASK_DEPTH_CHANNEL; docs/depth_channel.md): depth discontinuities found Canny-style in the
disparity, their metric jump Delta Z, and the weight w_depth.

The field (kernels, detrended orientation, q, gradient floor) and the candidates (D-3) are geom/candidates.py:
depth_field, depth_candidates. Here: the parameters, the pieces shared with the calibration (scripts/
calibrate_depth.py: null grid, contamination threshold, two-sided fit at x0, lambda_hat), and DepthChannel (D-4 ...
D-7): the jump with its retries (m = 8, m = 4, a constant fit, the doublet estimate), the gates (sign, significance,
relative jump with hysteresis), contour-smoothed metric Delta Z (sky, lower bound, focal bracket), the weight,
rasterization and strength.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..detector import disk
from .candidates import DepthField, depth_candidates, depth_field
from .channels import ChannelResult
from .context import GeometryContext
from .twosided import SideResult, TwoSidedResult, b_tri, one_sided_constant, two_sided_fit, window_rms

CALIBRATION_PATH = Path(__file__).with_name("data") / "depth_calibration.json"
CHI2_1_MEDIAN = 0.45493642311957283  # median of chi^2_1
NULL_MODELS = ("multiplicative", "additive")


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
    sigma_w_max: float = 1.02         # px, P99 of sigma_w_c at unambiguous real occlusions (data/depth_calibration.json;
                                      # P95 0.77, prior 3); s_hi = 2 sqrt(sigma^2 + sigma_w_max^2) = 3.23 px
    centre_tol: float = 0.4           # centring: |s_p + s_m| <= centre_tol s_sep (D2 addition)
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
    min_component_px: int = 5         # distinct pixels per component
    # TASK_GEOM_V2 V3: short thin components (pieces of thin doublet fits below the component minimum)
    short_thin: str = "off"           # off | strong | linked
    short_thin_min_px: int = 2        # strong: a waived piece has >= this many distinct pixels
    short_thin_gap_px: float = 4.0    # linked: endpoint gap between two pieces
    short_thin_angle_deg: float = 20.0  # linked: tangent difference
    short_thin_lateral_px: float = 1.5  # linked: the other piece's endpoint within this of the tangent line
    link_radius: float = 3.0          # px, candidates closer than this belong to one contour component (D3 decision)
    z: float = 3.29                   # two-sided p_fa = 1e-3
    null_stride: int = 4
    lambda_clamp: tuple[float, float] = (0.5, 1.0e4)
    null_model: str = "additive"      # gate null (same key as the fold channel; additive default since D5): multiplicative (lambda_hat) | additive (lambda_n SE^2 + sigma_geo_rho^2);
                                      # both significance columns are always recorded
    lambda_n: float = 104.6           # noise-only inflation of (Delta rho*/SE)^2 at k = 11 (data/depth_calibration.json)
    # D-6: magnitude
    smooth_n: int = 5                 # running median of rho_hat+- along contours
    sky_far_frac: float = 0.5         # far window at least this share sky -> Delta Z = inf
    canonical_focal: float = 300.0    # Z = f_n D / canonical_focal
    # D-7: weight
    dZ0: float = 0.10                 # m, knee of w_c = max(w_min, dZ0 / (dZ + dZ0))
    w_min: float = 0.05
    band_min: int = 2
    # TASK_GEOM_INTEGRATION G2: depth_and_mask also paints w_min on the sky-mask boundary ring (radius sky_ring_px)
    # within sky_ring_near_px of an accepted sky-edge candidate (a sky region with no accepted sky edge stays unpainted)
    sky_band: str = "depth"           # depth | depth_and_mask
    sky_ring_px: int = 2
    sky_ring_near_px: float = 4.0

    @classmethod
    def from_dict(cls, d: dict | None) -> DepthParams:
        d = dict(d or {})
        if "lambda_clamp" in d:
            d["lambda_clamp"] = tuple(d["lambda_clamp"])
        return cls(**d)

    @property
    def s_hi(self) -> float:
        return 2 * math.sqrt(self.sigma**2 + self.sigma_w_max**2)


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
    x: np.ndarray = None    # (N, 2) the null grid
    sigma2_add: float = 0.0  # additive null: sigma_geo_rho^2 with median(d^2 / (lambda_n SE^2 + sigma^2)) = 0.455

    def rms_max(self, p: DepthParams):
        def f(k: np.ndarray, m: int) -> np.ndarray:
            return np.fmax(np.fmax(p.planarity_factor * self.eps_hat, self.q_null),
                           p.b_tri_margin * b_tri(k, m, self.tri))
        return f


def depth_null(ctx: GeometryContext, p: DepthParams, f: DepthField, k: int | None = None) -> DepthNull:
    """Contamination threshold (flag 3: max(planarity_factor eps_hat, Q_{null_rms_quantile}, b_tri_margin b_tri)),
    lambda_hat_Delta_rho = median((Delta rho*/SE)^2 over the null) / 0.455, clamped to lambda_clamp, and the additive
    null's sigma_geo_rho^2 (recorded; the gate uses it only with null_model "additive")."""
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
    nul.res, nul.z2, nul.x = res, z2, xn
    if z2.size:
        nul.lambda_raw = float(np.median(z2) / CHI2_1_MEDIAN)
        nul.lambda_hat = float(np.clip(nul.lambda_raw, *p.lambda_clamp))
        nul.sigma2_add = _solve_sig2_rho(res.d_rho[ok], res.d_rho_se[ok], p.lambda_n)
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


# ---------------------------------------------------------------- D-4: the jump at x0 with its retries

def _side_at(side: SideResult, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A side's plane value and its SE at x0 (NaN where the side has no planar fit)."""
    d = x0 - side.c
    a = np.concatenate([np.ones((len(x0), 1)), d], -1)
    val = side.alpha + np.sum(side.beta * d, -1)
    var = np.einsum("ni,nij,nj->n", a, side.cov, a)
    with np.errstate(invalid="ignore"):
        return np.where(side.planar, val, np.nan), np.where(side.planar, np.sqrt(var), np.nan)


def _strip_share(mask: np.ndarray, x0: np.ndarray, n: np.ndarray, s_from: np.ndarray, s_to: np.ndarray, w_t: int,
                 sign: int) -> np.ndarray:
    """Share of the in-image samples (nearest pixel, s = s_from, s_from + 1, ... <= s_to, |t| <= w_t) of the strip
    x0 + sign s n + t n_perp that lie in mask; NaN if no sample is inside the image."""
    N = len(x0)
    out = np.full(N, np.nan)
    if N == 0:
        return out
    n_pos = np.maximum(np.floor(s_to - s_from + 1e-9).astype(int) + 1, 0)
    L = max(int(n_pos.max()), 1)
    s = s_from[:, None] + np.arange(L)[None]
    t = np.arange(-w_t, w_t + 1, dtype=np.float64)
    et = np.stack([-n[:, 1], n[:, 0]], -1)
    pts = x0[:, None, None, :] + sign * s[:, :, None, None] * n[:, None, None, :] + t[None, None, :, None] * et[:, None, None, :]
    H, W = mask.shape
    pu, pv = np.rint(pts[..., 0]).astype(np.intp), np.rint(pts[..., 1]).astype(np.intp)
    inside = (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H) & (np.arange(L)[None, :, None] < n_pos[:, None, None])
    hit = mask[np.clip(pv, 0, H - 1), np.clip(pu, 0, W - 1)] & inside
    with np.errstate(invalid="ignore", divide="ignore"):
        out = hit.sum((1, 2)) / inside.sum((1, 2))
    return out


def _obstacles(x0: np.ndarray, n: np.ndarray, obst: np.ndarray, reach: np.ndarray, s_a: float, w_t: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Distance along +n and -n to the nearest obstacle point (another D-3 candidate) inside the strip |t| <= w_t + 0.5,
    beyond s_a and within reach (inf if none): an edge between x0 and a window means the window is not on x0's
    surface."""
    N = len(x0)
    d_p, d_m = np.full(N, np.inf), np.full(N, np.inf)
    if N == 0 or len(obst) == 0:
        return d_p, d_m
    R = float(np.max(reach)) + w_t + 1.0
    sm = cKDTree(x0).sparse_distance_matrix(cKDTree(obst), R, output_type="coo_matrix")
    i, j = sm.row, sm.col
    d = obst[j] - x0[i]
    s = np.sum(d * n[i], -1)
    t = d[:, 1] * n[i, 0] - d[:, 0] * n[i, 1]
    ok = (np.abs(t) <= w_t + 0.5) & (np.abs(s) > s_a) & (np.abs(s) <= reach[i])
    pos, neg = ok & (s > 0), ok & (s < 0)
    np.minimum.at(d_p, i[pos], s[pos])
    np.minimum.at(d_m, i[neg], -s[neg])
    return d_p, d_m


def _solve_sig2_rho(d: np.ndarray, se: np.ndarray, lam_n: float) -> float:
    """sigma_geo_rho^2 >= 0 with median(d^2 / (lam_n se^2 + sigma^2)) = median(chi^2_1) (bisection on log10)."""
    med = lambda a: float(np.median(d**2 / (lam_n * se**2 + a)))  # noqa: E731
    if not d.size or med(0.0) <= CHI2_1_MEDIAN:
        return 0.0
    lo, hi = -40.0, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if med(10.0**mid) > CHI2_1_MEDIAN else (lo, mid)
    return 10.0 ** (0.5 * (lo + hi))


SIDE_MODES = ("", "plane8", "plane4", "const")
SKY_BAND_KEYS = ("sky_band", "sky_ring_px", "sky_ring_near_px")
SHORT_THIN_KEYS = ("short_thin", "short_thin_min_px", "short_thin_gap_px", "short_thin_angle_deg",
                   "short_thin_lateral_px")
SHORT_THIN = ("off", "strong", "linked")


def params_record(p: dict) -> dict:
    """The depth parameters as recorded in the outputs: TASK_GEOM_INTEGRATION keys only when their option is on, so
    runs with the options off keep the records of the runs made before they existed."""
    d = dict(p)
    if d.get("sky_band", "depth") == "depth":
        for k in SKY_BAND_KEYS:
            d.pop(k, None)
    if d.get("short_thin", "off") == "off":
        for k in SHORT_THIN_KEYS:
            d.pop(k, None)
    return d


def sky_boundary_ring(S: np.ndarray, radius: int) -> np.ndarray:
    """The sky-side boundary pixels of S dilated by radius (3x3 steps), the image border excluded (the definition of
    the 070 sky-band ROI, scripts/depth_rois.py)."""
    b = S & ~ndimage.binary_erosion(S, structure=np.ones((3, 3), bool), border_value=1)
    b[:1, :] = b[-1:, :] = False
    b[:, :1] = b[:, -1:] = False
    return ndimage.binary_dilation(b, structure=np.ones((3, 3), bool), iterations=radius) if radius > 0 else b


def se_eff(p: DepthParams, null: DepthNull, se: np.ndarray, pair: bool = False) -> np.ndarray:
    """The SE inflated as the active null inflates it: sqrt(lambda_hat SE^2) (multiplicative) or
    sqrt(lambda_n SE^2 + sigma_geo_rho^2) for Delta rho* (additive; sigma_geo_rho^2 / 2 per side)."""
    if p.null_model == "additive":
        return np.sqrt(p.lambda_n * se**2 + null.sigma2_add * (1.0 if pair else 0.5))
    return np.sqrt(null.lambda_hat) * se


def jump(ctx: GeometryContext, p: DepthParams, x0: np.ndarray, n: np.ndarray, k_c: np.ndarray, q_p: np.ndarray,
         q_m: np.ndarray, h_p: np.ndarray, h_m: np.ndarray, s_sep: np.ndarray, rho_s0: np.ndarray, s_q: float,
         null: DepthNull, obst: np.ndarray) -> dict[str, np.ndarray]:
    """D-4 at the candidates (x0, n toward the near side, skip k_c) and the per-candidate quantities of D-5/D-6.

    Per side (+ = +n, near; - = far): the plane at k_c with m = 8 (the contamination check retries m = 4), else
    m = 4 at k_c, else a constant fit (beta = 0); a side whose window lies beyond the next edge (another D-3 candidate
    in its strip, closer than k_c + m_used) is not used, and its constant fit covers the clear span
    [s_a, min(d_next - s_a, k_c + m)], s_a = ceil(s_hi / 2) + 1 (clear of both doublets). If a side still has no value,
    the doublet estimate Delta rho_doublet = coef |q|_lobe rho_sigma(x0) s_sep^2 = coef |h|_lobe s_sep^2 is the jump (flag thin, mode
    "doublet"; significance from the lobes: min(|q_p|, |q_m|) / s_q). Sky shares are measured on the spans used.
    """
    N = len(x0)
    kw = fit_kwargs(p, ctx)
    rmax = null.rms_max(p)
    s_a = math.ceil(p.s_hi / 2) + 1
    reach = (k_c + p.m).astype(np.float64)
    d_next = dict(zip((1, -1), _obstacles(x0, n, obst, reach, s_a, p.w_t)))
    val = {s: np.full(N, np.nan) for s in (1, -1)}
    se = {s: np.full(N, np.nan) for s in (1, -1)}
    beta = {s: np.zeros((N, 2)) for s in (1, -1)}  # plane slopes (0 for constant fits and doublet sides)
    mode = {s: np.zeros(N, dtype=np.int8) for s in (1, -1)}  # index into SIDE_MODES
    span = {s: (k_c.astype(np.float64), k_c + float(p.m)) for s in (1, -1)}
    r8 = two_sided_fit(ctx, x0, n, k_c, rms_max=rmax, **kw)
    for sgn, side in ((1, r8.p), (-1, r8.m)):
        v, e = _side_at(side, x0)
        use = side.planar & (d_next[sgn] > k_c + side.m_used)
        val[sgn][use], se[sgn][use], beta[sgn][use] = v[use], e[use], side.beta[use]
        mode[sgn][use] = np.where(side.m_used[use] == p.m, 1, 2)
        span[sgn][1][use] = (k_c + side.m_used)[use]
    need = (mode[1] == 0) | (mode[-1] == 0)
    if need.any():
        idx = np.nonzero(need)[0]
        r4 = two_sided_fit(ctx, x0[idx], n[idx], k_c[idx], rms_max=rmax, **{**kw, "m": p.m_min})
        for sgn, side in ((1, r4.p), (-1, r4.m)):
            v, e = _side_at(side, x0[idx])
            use = side.planar & (d_next[sgn][idx] > k_c[idx] + side.m_used) & (mode[sgn][idx] == 0)
            g = idx[use]
            val[sgn][g], se[sgn][g], mode[sgn][g], beta[sgn][g] = v[use], e[use], 2, side.beta[use]
            span[sgn][1][g] = (k_c[idx] + side.m_used)[use]
    for sgn in (1, -1):
        unset = mode[sgn] == 0
        obstructed = d_next[sgn] <= reach
        s_from = np.where(obstructed, float(s_a), k_c.astype(np.float64))
        s_to = np.where(obstructed, np.minimum(d_next[sgn] - s_a, reach), k_c + float(p.m_min))
        span[sgn][0][unset], span[sgn][1][unset] = s_from[unset], s_to[unset]
        if unset.any():
            idx = np.nonzero(unset)[0]
            a, e, _, ok = one_sided_constant(ctx, x0[idx], n[idx], s_from[idx], s_to[idx], p.w_t, sgn, ctx.omega,
                                             p.max_excluded)
            g = idx[ok]
            val[sgn][g], se[sgn][g], mode[sgn][g] = a[ok], e[ok], 3
    sky_share = {s: _strip_share(ctx.S, x0, n, span[s][0], span[s][1], p.w_t, s) for s in (1, -1)}

    doublet_rho = p.doublet_coef * 0.5 * (h_p + np.abs(h_m)) * s_sep**2  # |q|_lobe rho_sigma(x0) = |h|_lobe
    thin = (mode[1] == 0) | (mode[-1] == 0)
    rho_p, rho_m = val[1].copy(), val[-1].copy()
    only_p, only_m = thin & (mode[1] > 0), thin & (mode[-1] > 0)
    none_ = thin & (mode[1] == 0) & (mode[-1] == 0)
    rho_m[only_p] = rho_p[only_p] - doublet_rho[only_p]
    rho_p[only_m] = rho_m[only_m] + doublet_rho[only_m]
    rho_m[none_] = rho_s0[none_] - doublet_rho[none_] / 2
    rho_p[none_] = rho_s0[none_] + doublet_rho[none_] / 2
    d_rho = np.where(thin, doublet_rho, rho_p - rho_m)
    d_rho_se = np.where(thin, np.nan, np.hypot(se[1], se[-1]))
    lam = null.lambda_hat
    with np.errstate(invalid="ignore", divide="ignore"):
        signif_fit = np.abs(d_rho) / (d_rho_se * math.sqrt(lam))
        signif_lobe = np.minimum(q_p, np.abs(q_m)) / s_q
        signif = np.where(thin, signif_lobe, signif_fit)
        T_add = d_rho**2 / (p.lambda_n * d_rho_se**2 + null.sigma2_add)
        signif_add = np.where(thin, signif_lobe, np.sqrt(T_add))
        # far side consistent with rho = 0 (rho_hat_- - z SE_eff,- <= 0, SE inflated as the active null does): r is the
        # lower bound Delta rho* / (rho_hat_- + z SE_eff,-) (D3), flagged r_lower_bound
        zse_m = p.z * se_eff(p, null, se[-1])
        r_lb = (rho_m - zse_m <= 0) & ~thin
        den_lb = rho_m + zse_m
        r = np.where(r_lb, np.where(den_lb > 0, d_rho / den_lb, np.inf), d_rho / rho_m)
        r_gate = p.z * se_eff(p, null, d_rho_se, pair=True) / np.where(r_lb, den_lb, rho_m)
        doublet_ratio = np.where(thin, np.nan, doublet_rho / d_rho)
    fit_mode = np.full(N, "doublet", dtype=object)
    both = ~thin
    fit_mode[both & (mode[1] == 1) & (mode[-1] == 1)] = "m8"
    fit_mode[both & (mode[1] <= 2) & (mode[-1] <= 2) & ((mode[1] == 2) | (mode[-1] == 2))] = "m4"
    fit_mode[both & ((mode[1] == 3) | (mode[-1] == 3))] = "const"
    sky_edge = sky_share[-1] >= p.sky_far_frac
    sky_near = sky_share[1] >= 0.5
    for a in (r, signif, signif_add):
        a[sky_edge] = np.inf
    return {"fit_mode": fit_mode, "mode_p": np.asarray(SIDE_MODES, dtype=object)[mode[1]],
            "mode_m": np.asarray(SIDE_MODES, dtype=object)[mode[-1]], "thin": thin,
            "d_next_p": d_next[1], "d_next_m": d_next[-1], "rho_p": rho_p, "rho_m": rho_m, "se_p": se[1],
            "se_m": se[-1], "d_rho": d_rho, "d_rho_se": d_rho_se, "d_rho_min_span": r8.d_rho_min_span,
            "planes_cross": r8.planes_cross, "planar8_p": r8.p.planar, "planar8_m": r8.m.planar,
            "m_used_p": r8.p.m_used, "m_used_m": r8.m.m_used, "rms_rel_p": r8.p.rms_rel, "rms_rel_m": r8.m.rms_rel,
            "excl_p": r8.p.excl, "excl_m": r8.m.excl, "doublet": doublet_rho, "doublet_ratio": doublet_ratio,
            "signif": signif, "signif_add": signif_add, "r": r, "r_gate": r_gate, "sky_share_near": sky_share[1],
            "sky_share_far": sky_share[-1], "sky_edge": sky_edge, "sky_near": sky_near,
            "r_lower_bound": r_lb, "beta_p": beta[1], "beta_m": beta[-1]}


# ---------------------------------------------------------------- D-6: contours

def short_thin_accept(p, x: np.ndarray, n: np.ndarray, lab: np.ndarray, sizes: np.ndarray, thin: np.ndarray,
                      r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TASK_GEOM_V2 V3 on the surviving candidates (x, n, their D-5 component labels lab, component sizes in distinct
    pixels, thin flags, r). Pieces = components below min_component_px whose every member is thin.
    strong: a piece with >= short_thin_min_px pixels whose every member has r > tau_high is accepted.
    linked: pieces are linked end to end (union-find) when an endpoint gap is <= short_thin_gap_px, their tangents differ
      by <= short_thin_angle_deg, their mean normals point to the same side (same polarity) and each piece's nearest
      endpoint lies within short_thin_lateral_px of the other's tangent line; a linked chain is accepted if its pixels
      (summed over its pieces) reach min_component_px and its maximum r exceeds tau_high.
    Returns (accepted, link id (-1: not in a linked chain of >= 2 pieces)) per candidate."""
    acc = np.zeros(len(x), dtype=bool)
    link = np.full(len(x), -1, dtype=int)
    pieces = [k for k in np.unique(lab) if sizes[k] < p.min_component_px and thin[lab == k].all()]
    if not pieces:
        return acc, link
    if p.short_thin == "strong":
        for k in pieces:
            m = lab == k
            if sizes[k] >= p.short_thin_min_px and (r[m] > p.tau_high).all():
                acc[m] = True
        return acc, link
    # linked: per piece its mean normal, tangent, endpoints (extreme members along the tangent)
    info = []
    for k in pieces:
        m = np.nonzero(lab == k)[0]
        nn = n[m] * np.sign(n[m] @ n[m[0]])[:, None]
        nbar = nn.mean(0)
        nbar /= max(np.linalg.norm(nbar), 1e-12)
        tau = np.array([-nbar[1], nbar[0]])
        proj = x[m] @ tau
        info.append({"k": k, "m": m, "n": nbar, "tau": tau, "ends": x[m[[np.argmin(proj), np.argmax(proj)]]],
                     "px": int(sizes[k]), "rmax": float(np.nanmax(np.where(np.isfinite(r[m]), r[m], np.inf)))})
    parent = list(range(len(info)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cos_t = math.cos(math.radians(p.short_thin_angle_deg))
    ends = np.concatenate([d["ends"] for d in info])
    owner = np.repeat(np.arange(len(info)), 2)
    tree = cKDTree(ends)
    for a_, b_ in tree.query_pairs(p.short_thin_gap_px, output_type="ndarray"):
        i, j = owner[a_], owner[b_]
        if i == j:
            continue
        A, B = info[i], info[j]
        if abs(A["tau"] @ B["tau"]) < cos_t or A["n"] @ B["n"] <= 0:
            continue
        lat_ab = abs((ends[b_] - ends[a_]) @ A["n"])     # B's endpoint from A's tangent line (through A's endpoint)
        lat_ba = abs((ends[a_] - ends[b_]) @ B["n"])
        if lat_ab <= p.short_thin_lateral_px and lat_ba <= p.short_thin_lateral_px:
            parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(len(info))])
    for rt in np.unique(roots):
        grp = np.nonzero(roots == rt)[0]
        if len(grp) < 2:
            continue
        px = sum(info[g]["px"] for g in grp)
        rmax = max(info[g]["rmax"] for g in grp)
        for g in grp:
            link[info[g]["m"]] = rt
            if px >= p.min_component_px and rmax > p.tau_high:
                acc[info[g]["m"]] = True
    return acc, link


def chains(x: np.ndarray, n: np.ndarray, radius: float = 1.5, cos_min: float = 0.5) -> list[np.ndarray]:
    """Order candidates into contours: each links to its nearest neighbour ahead and behind along the tangent
    (n rotated by +90 deg) within radius, with n_i . n_j >= cos_min, and a link is kept only when mutual. A junction
    (a third contour meeting at an angle) fails the orientation or the mutual test, so chains stop there."""
    N = len(x)
    if N == 0:
        return []
    nxt = np.full(N, -1)
    pairs = cKDTree(x).query_pairs(radius, output_type="ndarray")
    if len(pairs):
        i = np.concatenate([pairs[:, 0], pairs[:, 1]])
        j = np.concatenate([pairs[:, 1], pairs[:, 0]])
        ok = np.sum(n[i] * n[j], -1) >= cos_min
        i, j = i[ok], j[ok]
        d = x[j] - x[i]
        along = d[:, 0] * -n[i, 1] + d[:, 1] * n[i, 0]
        dist = np.hypot(d[:, 0], d[:, 1])
        ahead, behind = np.full(N, -1), np.full(N, -1)
        order = np.lexsort((dist, i))
        for sel_mask, arr in ((along > 0, ahead), (along < 0, behind)):
            sel = order[sel_mask[order]]
            ii = i[sel]
            first = np.unique(ii, return_index=True)[1]
            arr[ii[first]] = j[sel][first]
        cand = ahead >= 0
        nxt[cand] = np.where(behind[ahead[cand]] == np.nonzero(cand)[0], ahead[cand], -1)
    has_prev = np.zeros(N, dtype=bool)
    has_prev[nxt[nxt >= 0]] = True
    seen = np.zeros(N, dtype=bool)
    out = []
    for start in list(np.nonzero(~has_prev)[0]) + list(range(N)):
        if seen[start]:
            continue
        c = []
        k = start
        while k >= 0 and not seen[k]:
            seen[k] = True
            c.append(k)
            k = nxt[k]
        out.append(np.asarray(c))
    return out


def running_median(chs: list[np.ndarray], values: np.ndarray, n: int, x: np.ndarray | None = None,
                   beta: np.ndarray | None = None) -> np.ndarray:
    """Running median over n candidates along each chain (the window shrunk symmetrically at chain ends); NaN values
    ignored. With positions x (N, 2) and plane slopes beta (N, 2), each neighbour j contributes its plane evaluated at
    the candidate's own x0: values_j + beta_j . (x_k - x_j) (exact on planes)."""
    out = values.copy()
    h = n // 2
    for c in chs:
        L = len(c)
        if L < 3 or h == 0:
            continue
        v = values[c]
        for k in range(L):
            hk = min(h, k, L - 1 - k)
            if hk:
                win = c[k - hk:k + hk + 1]
                w = v[k - hk:k + hk + 1]
                if beta is not None:
                    w = w + np.sum(beta[win] * (x[c[k]] - x[win]), -1)
                if np.isfinite(w).any():
                    out[c[k]] = np.nanmedian(w)
    return out


# ---------------------------------------------------------------- the channel

DEPTH_REASONS = ("polarity", "height", "sharpness", "centring", "symmetry", "sky_near", "sign", "signif", "rel",
                 "component", "accepted")


class DepthChannel:
    name = "depth"

    def run(self, ctx: GeometryContext, params: dict | DepthParams | None = None) -> ChannelResult:
        p = params if isinstance(params, DepthParams) else DepthParams.from_dict(params)
        if p.null_model not in NULL_MODELS:
            raise ValueError(f"unknown null_model {p.null_model!r}; expected one of {NULL_MODELS}")
        if p.sky_band not in ("depth", "depth_and_mask"):
            raise ValueError(f"unknown sky_band {p.sky_band!r}; expected depth or depth_and_mask")
        if p.gap_mode != "at_x0":
            raise ValueError("the depth magnitude uses gap_mode 'at_x0' (min_span is recorded only)")
        H, W = ctx.shape
        t_start = time.perf_counter()
        ctx.rho_f  # shared context field (timed separately: another channel may already have built it)
        t_ctx = time.perf_counter() - t_start

        f = depth_field(ctx, p)
        c = depth_candidates(ctx, p, field=f)
        nul = depth_null(ctx, p, f)
        N = len(c.x0)
        reason = c.reason.copy()
        P = np.nonzero(c.passed)[0]
        # obstacles: D-3 candidates that are depth edges by their own doublet estimate (r_doublet > tau_low); noise
        # doublets sit orders of magnitude below (synthetic: <= 1e-3 at noise 5e-4)
        with np.errstate(invalid="ignore", divide="ignore"):
            r_doublet = p.doublet_coef * 0.5 * (c.h_p + np.abs(c.h_m)) * c.s_sep**2 / c.rho_s0
        obst = c.x0[P][r_doublet[P] > p.tau_low]
        J = jump(ctx, p, c.x0[P], c.n[P], c.k_c[P], c.q_p[P], c.q_m[P], c.h_p[P], c.h_m[P], c.s_sep[P], c.rho_s0[P],
                 f.s_q, nul, obst)
        signif = J["signif_add"] if p.null_model == "additive" else J["signif"]
        sky_edge, sky_near = J["sky_edge"], J["sky_near"]
        with np.errstate(invalid="ignore"):
            sign_ok = sky_edge | (J["d_rho"] > 0)
            sig_ok = sky_edge | (signif > p.z)
            rel_ok = sky_edge | (J["r"] > p.tau_low)
        rp = np.full(len(P), "", dtype=object)
        rp[~rel_ok] = "rel"
        rp[~sig_ok] = "signif"
        rp[~sign_ok] = "sign"
        rp[sky_near] = "sky_near"
        surv = rp == ""

        # D-5 hysteresis on contour components of the surviving candidates: candidates within link_radius are linked
        # (merged crossings lie up to ~2.2 px apart along oblique edges, so 8-connectivity of rounded pixels would
        # split them); a component's size is its number of distinct rounded pixels
        ij = np.clip(c.ij[P], 0, [W - 1, H - 1])
        S_idx = np.nonzero(surv)[0]
        comp_px, comp_rmax = np.zeros(len(P), dtype=int), np.full(len(P), np.nan)
        comp_id = np.full(len(P), -1, dtype=int)
        if S_idx.size:
            pairs = cKDTree(c.x0[P][S_idx]).query_pairs(p.link_radius, output_type="ndarray")
            ns = S_idx.size
            adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(ns, ns))
            nlab, lab = connected_components(adj, directed=False)
            pix = np.unique(np.stack([lab, ij[S_idx, 1] * W + ij[S_idx, 0]], -1), axis=0)
            sizes = np.bincount(pix[:, 0], minlength=nlab)
            rmax = np.full(nlab, -np.inf)
            np.maximum.at(rmax, lab, np.nan_to_num(J["r"][S_idx], nan=-np.inf))
            comp_px[S_idx], comp_rmax[S_idx], comp_id[S_idx] = sizes[lab], rmax[lab], lab
        accepted = surv & (comp_px >= p.min_component_px) & (comp_rmax > p.tau_high)
        # TASK_GEOM_V2 V3 short thin components (off: nothing here runs)
        if p.short_thin not in SHORT_THIN:
            raise ValueError(f"unknown short_thin {p.short_thin!r}; expected one of {SHORT_THIN}")
        short_acc, short_link = np.zeros(len(P), dtype=bool), np.full(len(P), -1, dtype=int)
        if p.short_thin != "off" and S_idx.size:
            acc_s, link_s = short_thin_accept(p, c.x0[P][S_idx], c.n[P][S_idx], lab, sizes, J["thin"][S_idx],
                                              J["r"][S_idx])
            short_acc[S_idx], short_link[S_idx] = acc_s & ~accepted[S_idx], link_s
            accepted = accepted | short_acc
        rp[surv & ~accepted] = "component"
        rp[accepted] = "accepted"
        reason[P] = rp

        # D-6 magnitude: running medians of rho_hat+ and rho_hat- along contours, metric Delta Z
        A = np.nonzero(accepted)[0]
        rho_p_s, rho_m_s = J["rho_p"].copy(), J["rho_m"].copy()
        if A.size:
            chs = chains(c.x0[P][A], c.n[P][A])
            for key, bkey, arr in (("rho_p", "beta_p", rho_p_s), ("rho_m", "beta_m", rho_m_s)):
                vals = J[key][A].copy()
                if key == "rho_m":
                    vals[sky_edge[A]] = np.nan  # no far plane at sky edges
                arr[A] = running_median(chs, vals, p.smooth_n, c.x0[P][A], J[bkey][A])
        scale = 0.5 * (ctx.K.f_x + ctx.K.f_y) / p.canonical_focal
        # lower bound with the SEs inflated as the active null inflates them (D3 decision; D6: per null)
        zm, zp = p.z * se_eff(p, nul, J["se_m"]), p.z * se_eff(p, nul, J["se_p"])
        with np.errstate(invalid="ignore", divide="ignore"):
            dZ = scale * (1.0 / rho_m_s - 1.0 / rho_p_s)
            lower = ~sky_edge & (rho_m_s - zm <= 0)
            far_np = ~sky_edge & ~np.isfinite(J["se_m"]) & (rho_m_s <= 0)
            near_lb = rho_p_s - zp
            dZ_lb = scale * (1.0 / (rho_m_s + zm) - np.where(near_lb > 0, 1.0 / near_lb, np.inf))
        dZ = np.where(lower, np.where(np.isfinite(dZ_lb) & (dZ_lb > 0), dZ_lb, np.inf), dZ)
        dZ = np.where(far_np | sky_edge, np.inf, dZ)
        lower_bound = lower | far_np
        rel_f = ctx.K.sigma_f_x / ctx.K.f_x
        dZ_lo, dZ_hi = dZ * (1 - rel_f), dZ * (1 + rel_f)

        # D-7 weight, rasterization, strength
        with np.errstate(invalid="ignore", divide="ignore"):
            w_c = np.maximum(p.w_min, p.dZ0 / (dZ + p.dZ0))
        w_c = np.where(np.isinf(dZ), p.w_min, w_c)
        sw = c.sigma_wc[P]
        band = np.maximum(p.band_min, np.ceil(sw).astype(int) + 1)
        w_depth = np.ones((H, W))
        strength = np.zeros((H, W))
        dZ_map = np.full((H, W), -np.inf)
        u, v = ij[:, 0], ij[:, 1]
        for b in np.unique(band[accepted]):
            a = accepted & (band == b)
            wi = np.ones((H, W))
            np.minimum.at(wi, (v[a], u[a]), w_c[a])
            si = np.zeros((H, W))
            np.maximum.at(si, (v[a], u[a]), J["r"][a])
            zi = np.full((H, W), -np.inf)
            np.maximum.at(zi, (v[a], u[a]), dZ[a])
            fp = disk(int(b))
            w_depth = np.minimum(w_depth, ndimage.grey_erosion(wi, footprint=fp, mode="constant", cval=1.0))
            strength = np.maximum(strength, ndimage.grey_dilation(si, footprint=fp, mode="constant", cval=0.0))
            dZ_map = np.maximum(dZ_map, ndimage.grey_dilation(zi, footprint=fp, mode="constant", cval=-np.inf))
        n_ring = 0
        if p.sky_band == "depth_and_mask":
            sk = accepted & sky_edge
            if sk.any():
                seed = np.ones((H, W), dtype=bool)
                seed[v[sk], u[sk]] = False
                near = ndimage.distance_transform_edt(seed) <= p.sky_ring_near_px
                ring = sky_boundary_ring(ctx.S, p.sky_ring_px) & near
                n_ring = int((ring & (w_depth > p.w_min)).sum())
                w_depth[ring] = p.w_min
                strength[ring] = np.inf
                dZ_map[ring] = np.inf
        dZ_map = np.where(np.isneginf(dZ_map), np.nan, dZ_map).astype(np.float32)
        edge_mask = np.zeros((H, W), dtype=bool)
        edge_mask[v[accepted], u[accepted]] = True
        with np.errstate(invalid="ignore", divide="ignore"):
            q_over_s = np.clip(f.q / f.s_q, -6.5e4, 6.5e4).astype(np.float16)

        # table: every D-3 candidate, D-4 ... D-7 columns where they apply
        def full(a, fill=np.nan):
            a = np.asarray(a)
            dt = np.float64 if (a.dtype.kind in "iu" and isinstance(fill, float)) else a.dtype
            out = np.full(N, fill, dtype=dt)
            out[P] = a
            return out

        with np.errstate(invalid="ignore", divide="ignore"):
            centring = np.abs(c.s_p + c.s_m) / c.s_sep
        df = pd.DataFrame({
            "u": c.ij[:, 0], "v": c.ij[:, 1], "u0": c.x0[:, 0], "v0": c.x0[:, 1], "n_u": c.n[:, 0], "n_v": c.n[:, 1],
            "g_n": c.g_n, "rho_s0": c.rho_s0, "r_doublet": r_doublet, "q_p": c.q_p, "q_m": c.q_m, "s_p": c.s_p, "s_m": c.s_m,
            "s_sep": c.s_sep, "centring": centring, "sigma_wc": c.sigma_wc, "k_c": c.k_c,
            **{k: full(J[k], "" if J[k].dtype == object else (False if J[k].dtype == bool else np.nan))
               for k in J if k not in ("sky_edge", "sky_near", "thin", "planes_cross", "planar8_p", "planar8_m",
                                       "beta_p", "beta_m", "r_lower_bound")},
            **{k: full(J[k], False) for k in ("sky_edge", "sky_near", "thin", "planes_cross", "planar8_p",
                                               "planar8_m", "r_lower_bound")},
            "component_id": full(comp_id, -1), "component_px": full(comp_px, 0), "comp_r_max": full(comp_rmax),
            "rho_p_s": full(rho_p_s), "rho_m_s": full(rho_m_s), "dZ": full(dZ), "dZ_lo": full(dZ_lo),
            "dZ_hi": full(dZ_hi), "lower_bound": full(lower_bound, False), "w_c": full(w_c),
            "band": full(band, 0), "accepted": full(accepted, False), "reason": reason,
        })
        df["reason"] = df["reason"].astype(str)
        if p.short_thin != "off":  # V3 records (tables without the option are unchanged)
            df["short_thin"] = full(short_acc, False)
            df["short_thin_link"] = full(short_link, -1)

        # gate diagnostics (candidates reaching the gates with a fit: not sky, not thin, sign ok)
        gd = ~sky_edge & ~J["thin"] & ~sky_near & (J["d_rho"] > 0) & np.isfinite(J["r_gate"])
        rg = J["r_gate"][gd]
        counts = {**c.counts, "n_sky_near": int(sky_near.sum()), "n_sign_ok": int((~sky_near & sign_ok).sum()),
                  "n_signif_ok": int((~sky_near & sign_ok & sig_ok).sum()),
                  "n_rel_ok": int(surv.sum()), "n_accepted": int(accepted.sum())}
        modes = {f"n_fit_{m}": int((J["fit_mode"] == m).sum()) for m in ("m8", "m4", "const", "doublet")}
        scalars = {
            "s_q": f.s_q, "sigma_g": f.sigma_g, "g_floor": f.g_floor, "valid_frac": float(f.valid.mean()),
            "lambda_hat": nul.lambda_hat, "lambda_hat_raw": nul.lambda_raw, "eps_hat": nul.eps_hat,
            "null_rms_q": nul.q_null, "n_null": int(len(nul.res.d_rho)), "n_null_valid": int(nul.z2.size),
            "sigma_geo_rho": math.sqrt(nul.sigma2_add), "null_model": p.null_model,
            "sigma_w_max": p.sigma_w_max, "s_lo": c.s_lo, "s_hi": c.s_hi, **counts, **modes,
            "n_signif_ok_add": int((~sky_near & sign_ok & (sky_edge | (J["signif_add"] > p.z))).sum()),
            "n_thin_accepted": int((accepted & J["thin"]).sum()), "n_sky_edge_accepted": int((accepted & sky_edge).sum()),
            **({"n_short_thin_accepted": int(short_acc.sum())} if p.short_thin != "off" else {}),
            "n_lower_bound_accepted": int((accepted & lower_bound).sum()),
            "r_gate_P10": float(np.percentile(rg, 10)) if rg.size else float("nan"),
            "r_gate_P50": float(np.percentile(rg, 50)) if rg.size else float("nan"),
            "r_gate_P90": float(np.percentile(rg, 90)) if rg.size else float("nan"),
            "share_signif_binding": float((rg > p.tau_low).mean()) if rg.size else float("nan"),
            "f_n": 0.5 * (ctx.K.f_x + ctx.K.f_y), "sigma_f_rel": float(rel_f), "dZ0": p.dZ0,
            "frac_w_lt_0.5": float((w_depth < 0.5).mean()), "frac_w_lt_0.9": float((w_depth < 0.9).mean()),
            **({"n_sky_ring_px": n_ring} if p.sky_band != "depth" else {}),
            "params": params_record(dataclasses.asdict(p)),
        }
        scalars["context_s"] = t_ctx
        scalars["runtime_s"] = time.perf_counter() - t_start - t_ctx
        nr = nul.res
        null_df = pd.DataFrame({"u": nul.x[:, 0], "v": nul.x[:, 1], "d_rho": nr.d_rho, "d_rho_se": nr.d_rho_se,
                                "planar_p": nr.p.planar, "planar_m": nr.m.planar})
        return ChannelResult(name=self.name, w_net=w_depth, strength=strength, candidates=df,
                             arrays={"dZ": dZ_map, "q_over_s": q_over_s, "edge_mask": edge_mask}, scalars=scalars,
                             w_min=p.w_min, tables={"null": null_df})
