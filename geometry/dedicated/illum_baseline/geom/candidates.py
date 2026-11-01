"""Line candidates for the geometric-edge channels.

F2.1 fold candidates: structure tensor of the dense normals, T_s = sum_c G_sigma_T * (grad n_c grad n_c^T), with
Gaussian derivatives at sigma_d on valid normals only (pixels whose derivative support holds an invalid normal are
excluded) and normalized convolution at sigma_T. Largest eigenvalue lambda_1 and its unit eigenvector e1 (across the
fold); rotation rate r_n = sqrt(lambda_1) (rad/px). Non-maximum suppression along e1 (bilinear samples at x +- e1),
r_n > cand_rate_factor * median(r_n), subpixel x0 from a parabola along e1.

Sign convention: e1 = (cos phi, sin phi) with phi in (-90, 90] deg, i.e. e1_u > 0, or e1 = (0, 1). Side "+" is +e1.

F2.2 rounding width: the 10%-90% distance of the normalized normal-angle profile along e1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..detector import gaussian_gradient, kernels
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


def depth_candidates(ctx: GeometryContext, params: dict) -> Candidates:
    """Future depth channel: Canny-like candidates on the detrended disparity gradient (docs/geometry_channels.md
    section 5). Not implemented."""
    raise NotImplementedError("depth candidates are a later milestone")
