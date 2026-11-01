"""Line candidates for the geometric-edge channels.

F2.1 fold candidates: structure tensor of the dense normals, T_s = sum_c G_sigma_T * (grad n_c grad n_c^T), with
Gaussian derivatives at sigma_d on valid normals only (pixels whose derivative support holds an invalid normal are
excluded) and normalized convolution at sigma_T. Largest eigenvalue lambda_1 and its unit eigenvector e1 (across the
fold); rotation rate r_n = sqrt(lambda_1) (rad/px). Non-maximum suppression along e1 (bilinear samples at x +- e1),
r_n > cand_rate_factor * median(r_n), subpixel x0 from a parabola along e1.

Sign convention: e1 = (cos phi, sin phi) with phi in (-90, 90] deg, i.e. e1_u > 0, or e1 = (0, 1). Side "+" is +e1.

F2.2 rounding width: the 10%-90% distance of the normalized normal-angle profile along e1.

Depth candidates (TASK_DEPTH_CHANNEL D-K, D-1, D-2; the candidates themselves, D-3, follow in D2): depth_kernels,
depth_field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..detector import _correlate_valid, dilate, gaussian_gradient, kernels
from .context import GeometryContext
from .sampling import bilinear


@dataclass(frozen=True)
class Candidates:
    ij: np.ndarray       # (N, 2) int (u, v): the NMS pixel
    x0: np.ndarray       # (N, 2) subpixel location (u, v)
    e1: np.ndarray       # (N, 2) unit, across the line; side "+" is +e1
    rate: np.ndarray     # (N,) r_n at the pixel (rad/px) for folds
    rate_map: np.ndarray  # (H, W), NaN where not valid
    e1_map: np.ndarray   # (H, W, 2)
    valid_map: np.ndarray  # (H, W) bool: where the rate is defined
    threshold: float
    median_rate: float


def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian (detector kernels, truncate 4), zero outside the image."""
    _, g0, _ = kernels(sigma, 4.0)
    out = ndimage.correlate1d(a, g0, axis=1, mode="constant", cval=0.0)
    return ndimage.correlate1d(out, g0, axis=0, mode="constant", cval=0.0)


def normal_structure_tensor(ctx: GeometryContext, sigma_d: float, sigma_T: float
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r_n (H, W), e1 (H, W, 2), valid (H, W)) from the structure tensor of n_hat."""
    nv = ctx.normal_valid
    R = int(math.ceil(4.0 * sigma_d))
    vd = ndimage.minimum_filter(nv, size=2 * R + 1, mode="constant", cval=False)
    Juu, Juv, Jvv = (np.zeros(ctx.shape) for _ in range(3))
    for c in range(3):
        nc = np.where(nv, ctx.n_hat[..., c], 0.0)
        du, dv = gaussian_gradient(nc, sigma_d)
        Juu += du * du
        Juv += du * dv
        Jvv += dv * dv
    m = vd.astype(np.float64)
    den = _smooth(m, sigma_T)
    with np.errstate(invalid="ignore", divide="ignore"):
        a, b, d = (_smooth(J * m, sigma_T) / den for J in (Juu, Juv, Jvv))
    lam1 = 0.5 * (a + d) + np.sqrt(0.25 * (a - d) ** 2 + b * b)
    phi = 0.5 * np.arctan2(2 * b, a - d)  # (-pi/2, pi/2]
    e1 = np.stack([np.cos(phi), np.sin(phi)], -1)
    e1[np.isclose(phi, -np.pi / 2)] = (0.0, 1.0)
    r_n = np.sqrt(np.maximum(lam1, 0.0))
    r_n[~vd] = np.nan
    e1[~vd] = np.nan
    return r_n, e1, vd


def fold_candidates(ctx: GeometryContext, sigma_d: float, sigma_T: float, rate_factor: float) -> Candidates:
    """F2.1."""
    r_n, e1, valid = normal_structure_tensor(ctx, sigma_d, sigma_T)
    med = float(np.median(r_n[valid])) if valid.any() else float("nan")
    thr = rate_factor * med
    r0 = np.where(valid, r_n, 0.0)
    v, u = np.nonzero(valid & (r0 > thr))
    e = e1[v, u]
    all_px = np.ones(ctx.shape, dtype=bool)
    rp, _ = bilinear(r0, all_px, u + e[:, 0], v + e[:, 1])
    rm, _ = bilinear(r0, all_px, u - e[:, 0], v - e[:, 1])
    rp, rm = np.nan_to_num(rp, nan=0.0), np.nan_to_num(rm, nan=0.0)
    rc = r0[v, u]
    keep = (rc > rp) & (rc >= rm)
    u, v, e, rc, rp, rm = u[keep], v[keep], e[keep], rc[keep], rp[keep], rm[keep]
    curv = rp - 2 * rc + rm
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(curv < 0, (rm - rp) / (2 * curv), 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    x0 = np.stack([u + delta * e[:, 0], v + delta * e[:, 1]], -1)
    return Candidates(ij=np.stack([u, v], -1), x0=x0, e1=e, rate=rc, rate_map=r_n, e1_map=e1, valid_map=valid,
                      threshold=thr, median_rate=med)


def rounding_width(ctx: GeometryContext, x0: np.ndarray, e1: np.ndarray, half_len: int,
                   w_lo: float = 2.0, w_hi: float = 12.0, theta_min: float = 1e-6
                   ) -> tuple[np.ndarray, np.ndarray]:
    """F2.2: (width (N,), ok (N,)). theta(s) = angle(n(s), n(-L)) for s = -L..L, normalized by theta(L); the
    10%/90% crossings of its running maximum, linearly interpolated; width clamped to [w_lo, w_hi]. Profiles with
    an invalid sample or theta(L) < theta_min get w_hi and ok = False."""
    s = np.arange(-half_len, half_len + 1, dtype=np.float64)
    pts = x0[:, None, :] + s[None, :, None] * e1[:, None, :]
    n, okp = bilinear(ctx.n_hat, ctx.normal_valid, pts[..., 0], pts[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        ref = n[:, :1, :]
        th = np.arctan2(np.linalg.norm(np.cross(n, ref), axis=-1), np.sum(n * ref, -1))
        ok = okp.all(1) & (th[:, -1] >= theta_min)
        prof = np.maximum.accumulate(np.where(ok[:, None], th / th[:, -1:], 0.0), axis=1)

    def crossing(level: float) -> np.ndarray:
        i = np.argmax(prof >= level, axis=1)  # first index at or above the level (i >= 1 since prof[0] = 0)
        i = np.maximum(i, 1)
        p0, p1 = prof[np.arange(len(i)), i - 1], prof[np.arange(len(i)), i]
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(p1 > p0, (level - p0) / (p1 - p0), 0.0)
        return s[i - 1] + f

    width = np.clip(crossing(0.9) - crossing(0.1), w_lo, w_hi)
    width = np.where(ok, width, w_hi)
    return width, ok


# ---------------------------------------------------------------- depth candidates (TASK_DEPTH_CHANNEL)

MAD_TO_STD = 0.6745


def depth_kernels(sigma: float, truncate: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(x, g0, g1, g2) on x in [-R, R], R = ceil(truncate sigma) (D-K).

    g0, g2: the detector's moment-constrained Gaussian and second-derivative kernels (sum g0 = 1, sum g2 = 0,
    sum x^2 g2 = 2). g1: -(x / sigma^2) g0, normalized so that the implemented filter (a correlation, as in
    detector._correlate_valid) returns exactly 1 on f(u) = u, i.e. sum x g1 = 1 (which flips the sign of -(x/s^2) g0);
    sum g1 = 0 by antisymmetry.
    """
    x, g0, g2 = kernels(sigma, truncate)
    g1 = -(x / sigma**2) * g0
    g1 = g1 / (x * g1).sum()
    return x, g0, g1, g2


def _sep(a: np.ndarray, ku: np.ndarray, kv: np.ndarray) -> np.ndarray:
    """Separable correlation: ku along u (axis 1), kv along v (axis 0), odd-reflect padding (exact on affine
    functions), same shape."""
    ru, rv = ku.size // 2, kv.size // 2
    p = np.pad(a, ((rv, rv), (ru, ru)), mode="reflect", reflect_type="odd")
    return _correlate_valid(_correlate_valid(p, ku, axis=1), kv, axis=0)


@dataclass(frozen=True)
class DepthField:
    rho_s: np.ndarray       # (H, W) G_sigma * rho_src
    H_uu: np.ndarray        # Hessian of rho_s
    H_vv: np.ndarray
    H_uv: np.ndarray
    grad: np.ndarray        # (H, W, 2) the orientation gradient: grad rho_s - grad rho_{sigma_L} (detrended) or grad rho_s
    grad_norm: np.ndarray   # (H, W)
    n: np.ndarray           # (H, W, 2) grad / |grad| (toward larger rho, the near side); (1, 0) where |grad| = 0
    q: np.ndarray           # (H, W) n^T H n / max(rho_s, rho_floor), signed; 0 outside Omega
    valid: np.ndarray       # (H, W) Omega and |grad| >= k_g median(|grad| over Omega)
    s_q: float              # median(|q| over valid, away from the sky and the border) / 0.6745
    sigma_L: float
    grad_floor: float


def depth_field(ctx: GeometryContext, p, detrend: bool = True) -> DepthField:
    """D-K, D-1, D-2 on the context's disparity (p.disparity: "rho_f", the default and the fold channel's field, or
    "rho" for ablations). p: DepthParams (sigma, truncate, sigma_w_prior, sigma_L_factor, k_g, disparity)."""
    src = {"rho_f": ctx.rho_f, "rho": ctx.rho}[p.disparity]
    src = np.asarray(src, dtype=np.float64)
    omega = ctx.omega
    _, g0, g1, g2 = depth_kernels(p.sigma, p.truncate)
    rho_s = _sep(src, g0, g0)
    H_uu, H_vv, H_uv = _sep(src, g2, g0), _sep(src, g0, g2), _sep(src, g1, g1)
    gu, gv = _sep(src, g1, g0), _sep(src, g0, g1)
    sigma_tot = math.hypot(p.sigma, p.sigma_w_prior)
    sigma_L = p.sigma_L_factor * sigma_tot
    if detrend:
        _, h0, h1, _ = depth_kernels(sigma_L, p.truncate)
        gu, gv = gu - _sep(src, h1, h0), gv - _sep(src, h0, h1)
    gnorm = np.hypot(gu, gv)
    with np.errstate(invalid="ignore", divide="ignore"):
        nu = np.where(gnorm > 0, gu / gnorm, 1.0)
        nv = np.where(gnorm > 0, gv / gnorm, 0.0)
    q = (nu * nu * H_uu + 2 * nu * nv * H_uv + nv * nv * H_vv) / np.maximum(rho_s, ctx.rho_floor)
    q = np.where(omega, q, 0.0)
    floor = p.k_g * float(np.median(gnorm[omega])) if omega.any() else 0.0
    valid = omega & (gnorm >= floor) & (gnorm > 0)
    R = int(math.ceil(p.truncate * p.sigma))
    clean = valid & ~dilate(~omega, R)
    clean[:R, :] = clean[-R:, :] = False
    clean[:, :R] = clean[:, -R:] = False
    s_q = float(np.median(np.abs(q[clean])) / MAD_TO_STD) if clean.any() else float("nan")
    return DepthField(rho_s=rho_s, H_uu=H_uu, H_vv=H_vv, H_uv=H_uv, grad=np.stack([gu, gv], -1), grad_norm=gnorm,
                      n=np.stack([nu, nv], -1), q=q, valid=valid, s_q=s_q, sigma_L=sigma_L, grad_floor=floor)


def depth_candidates(ctx: GeometryContext, params) -> Candidates:
    """D-3 (zero crossings, polarity, lobes, two-sided height, sharpness): TASK_DEPTH_CHANNEL D2."""
    raise NotImplementedError("depth candidates (D-3) are milestone D2")
