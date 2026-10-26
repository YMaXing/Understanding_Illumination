"""Stage B detector (spec M3, B1-B7): pure numpy/scipy, float64, network resolution, upright frame.

w = max(w_min, eps / (q_pool + eps)), where q = |LoG_sigma(rho)| / max(rho_sigma, rho_floor) and rho = 1/D.
Disparity is affine in pixel coordinates on any plane, and the kernels below annihilate affine
functions exactly (sum g2 = 0, odd moments 0, odd-reflect padding), so q = 0 on planes of any slant.

No torch or network imports here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

MAD_TO_STD = 0.6745  # median(|X|) / std for a Gaussian


@dataclass(frozen=True)
class Params:
    sky_threshold: float = 0.5
    open_close_radius: int = 1
    min_area_px: int = 50
    sigma: float = 1.25
    sigma_w_prior: float = 1.0
    truncate: float = 4.0
    rho_floor_frac: float = 0.5
    rho_floor_percentile: float = 1.0
    k_tau: float = 3.0
    k_eps: float = 3.0
    w_min: float = 0.05
    s_hat_min: float | None = 1.0e-6
    s_hat_max: float | None = None

    @property
    def R(self) -> int:
        """Kernel half-support."""
        return int(math.ceil(self.truncate * self.sigma))

    @property
    def r(self) -> int:
        """Pooling / sky-band radius: ceil(sqrt(sigma^2 + sigma_w_prior^2)) + 1."""
        return int(math.ceil(math.hypot(self.sigma, self.sigma_w_prior))) + 1


# ---------------------------------------------------------------- kernels and filtering

def kernels(sigma: float, truncate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, g0, g2) on x in [-R, R]: sum g0 = 1, sum g2 = 0, sum x^2 g2 = 2, odd moments 0."""
    R = int(math.ceil(truncate * sigma))
    x = np.arange(-R, R + 1, dtype=np.float64)
    g0 = np.exp(-(x**2) / (2 * sigma**2))
    g0 /= g0.sum()
    g2 = (x**2 / sigma**4 - 1 / sigma**2) * g0
    g2 -= g2.sum() * g0
    g2 /= (x**2 * g2).sum() / 2
    return x, g0, g2


def _correlate_valid(a: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """Valid-mode 1-D correlation along axis (kernels are symmetric, so this is also convolution)."""
    n = a.shape[axis] - (k.size - 1)
    out = np.zeros(a.shape[:axis] + (n,) + a.shape[axis + 1:], dtype=np.float64)
    for i, ki in enumerate(k):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(i, i + n)
        out += ki * a[tuple(sl)]
    return out


def smooth_and_laplacian(rho: np.ndarray, sigma: float, truncate: float) -> tuple[np.ndarray, np.ndarray]:
    """rho_sigma = (g0 x g0) * rho and L = (g2_x g0_y + g0_x g2_y) * rho, odd-reflect padded, same shape."""
    _, g0, g2 = kernels(sigma, truncate)
    R = g0.size // 2
    p = np.pad(np.asarray(rho, dtype=np.float64), R, mode="reflect", reflect_type="odd")
    a0 = _correlate_valid(p, g0, axis=1)  # g0 along x
    a2 = _correlate_valid(p, g2, axis=1)  # g2 along x
    rho_sigma = _correlate_valid(a0, g0, axis=0)
    L = _correlate_valid(a2, g0, axis=0) + _correlate_valid(a0, g2, axis=0)
    return rho_sigma, L


# ---------------------------------------------------------------- morphology

def disk(r: int) -> np.ndarray:
    """Boolean footprint x^2 + y^2 <= r^2 (r=1: 5 px, r=3: 29 px)."""
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x**2 + y**2 <= r**2


def _binary_op(op, mask: np.ndarray, r: int) -> np.ndarray:
    """Binary morphology with edge-replicated borders, so regions touching the border are not eroded there.

    The pad is 2r because opening/closing chain two ops; scipy's border_value=0 then only reaches the
    padding, never the cropped image.
    """
    if r <= 0:
        return mask.copy()
    m = 2 * r
    p = np.pad(mask, m, mode="edge")
    return op(p, structure=disk(r))[m:-m, m:-m]


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    return _binary_op(ndimage.binary_erosion, mask, r)


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    return _binary_op(ndimage.binary_dilation, mask, r)


def sky_mask(sky_prob: np.ndarray | None, shape: tuple[int, int], p: Params) -> np.ndarray:
    """B1: threshold, open then close with disk(open_close_radius), drop small 8-connected components."""
    if sky_prob is None:
        return np.zeros(shape, dtype=bool)
    s = np.asarray(sky_prob, dtype=np.float64) >= p.sky_threshold
    s = _binary_op(ndimage.binary_opening, s, p.open_close_radius)
    s = _binary_op(ndimage.binary_closing, s, p.open_close_radius)
    lab, n = ndimage.label(s, structure=np.ones((3, 3), dtype=bool))
    if n:
        sizes = np.bincount(lab.ravel())
        small = sizes < p.min_area_px
        small[0] = False
        s[small[lab]] = False
    return s


# ---------------------------------------------------------------- main entry

def compute_weights(D: np.ndarray, sky_prob: np.ndarray | None, params: Params) -> dict[str, Any]:
    """Canonical depth D (H, W) and sky probability -> weights and intermediates (all float64 / bool)."""
    p = params
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError(f"D must be 2-D, got {D.shape}")
    R, r = p.R, p.r
    H, W = D.shape

    # B1
    S = sky_mask(sky_prob, D.shape, p)
    sky_frac = float(S.mean())
    ground = ~S
    if not ground.any():
        raise ValueError("the whole image is sky; nothing to weight")

    # B2
    bad = ground & ~(np.isfinite(D) & (D > 0))
    D_fix = D.copy()
    if bad.any():
        good = ground & ~bad
        if not good.any():
            raise ValueError("no valid depth outside the sky")
        D_fix[bad] = 1e-6 * np.median(D[good])
    rho = np.zeros_like(D_fix)
    rho[ground] = 1.0 / D_fix[ground]

    # B3
    rho_sigma, L = smooth_and_laplacian(rho, p.sigma, p.truncate)

    # B4
    rho_floor = p.rho_floor_frac * float(np.percentile(rho[ground], p.rho_floor_percentile))
    L_over_rho = L / np.maximum(rho_sigma, rho_floor)
    sky_interior = erode(S, r)
    L_over_rho[sky_interior] = 0.0
    q = np.abs(L_over_rho)

    # B5
    valid = ~dilate(S, r)
    valid[:R, :] = valid[-R:, :] = False
    valid[:, :R] = valid[:, -R:] = False
    n_valid = int(valid.sum())
    if n_valid:
        s_hat_raw = float(np.median(q[valid]) / MAD_TO_STD)
    elif p.s_hat_min is not None:
        s_hat_raw = float("nan")
    else:
        raise ValueError("no pixels available for the noise estimate and s_hat_min is null")
    s_hat = s_hat_raw if n_valid else p.s_hat_min
    if p.s_hat_min is not None:
        s_hat = max(s_hat, p.s_hat_min)
    if p.s_hat_max is not None:
        s_hat = min(s_hat, p.s_hat_max)
    s_hat_clamped = not (n_valid and s_hat == s_hat_raw)

    # B6: threshold first, then pool.
    tau = p.k_tau * s_hat
    eps = p.k_eps * s_hat
    q_eff = np.maximum(q - tau, 0.0)
    q_pool = ndimage.grey_dilation(q_eff, footprint=disk(r), mode="constant", cval=0.0)

    # B7
    if eps > 0:
        w = np.maximum(p.w_min, eps / (q_pool + eps))
    else:
        w = np.where(q_pool > 0, p.w_min, 1.0)
    w[q_pool == 0] = 1.0

    return {
        "w": w,
        "q": q,
        "L_over_rho": L_over_rho,
        "q_eff": q_eff,
        "q_pool": q_pool,
        "rho": rho,
        "rho_sigma": rho_sigma,
        "L": L,
        "sky_mask": S,
        "noise_valid": valid,
        "scalars": {
            "s_hat": s_hat,
            "s_hat_raw": s_hat_raw,
            "s_hat_clamped": bool(s_hat_clamped),
            "tau": tau,
            "eps": eps,
            "rho_floor": rho_floor,
            "r": r,
            "R": R,
            "sky_frac": sky_frac,
            "n_bad_depth": int(bad.sum()),
            "n_noise_valid": n_valid,
            "frac_w_lt_0.5": float((w < 0.5).mean()),
            "frac_w_lt_0.9": float((w < 0.9).mean()),
            "frac_w_lt_1": float((w < 1).mean()),
            "shape_HxW": [H, W],
        },
    }
