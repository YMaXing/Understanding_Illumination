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
    a_min: float | None = None
    lambda_ref: float = 15.0
    eps_floor: bool = False  # with a_min: eps = max(k_eps s_hat, (k_eps / k_tau) tau_min)
    prefilter: str = "none"  # none | trilateral (tangent-plane bilateral on rho before the Laplacian)
    tri_sigma_s: float = 3.0   # px, spatial
    tri_sigma_r: float = 0.01  # relative deviation from the local tangent plane
    tri_sigma_g: float = 3.0   # px, Gaussian scale of the tangent-plane gradient
    tri_shat: str = "unfiltered"  # s_hat from the unfiltered (unfiltered) or filtered response
    scale_band: bool = False   # q from L_sigma - L_{band_factor sigma}: cancels constant curvature (scale-ratio test)
    band_factor: float = 2.0
    scale_band_mode: str = "replace"  # replace: q = |band|; veto: q = min(|L|, g |band|), g = 1 / (1 - 1/band_factor)

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


# ---------------------------------------------------------------- threshold floor (TASK_070_STRIPES T4)

def threshold_floor(a_min: float, lambda_ref: float, sigma: float) -> float:
    """tau_min = a_min k0^2 exp(-sigma^2 k0^2 / 2), k0 = 2 pi / lambda_ref: the peak q of a relative disparity
    ripple of amplitude a_min at wavelength lambda_ref after the Gaussian-smoothed Laplacian."""
    k0 = 2 * math.pi / lambda_ref
    return a_min * k0**2 * math.exp(-(sigma**2) * k0**2 / 2)


# ---------------------------------------------------------------- trilateral prefilter (stripe alternative)

def gaussian_gradient(rho: np.ndarray, sigma: float, truncate: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """(d rho/du, d rho/dv) with a Gaussian-weighted derivative kernel k1 = x g0 / sum(x^2 g0), exact on
    affine functions (sum k1 = 0, sum x k1 = 1), odd-reflect padded."""
    x, g0, _ = kernels(sigma, truncate)
    k1 = x * g0 / (x**2 * g0).sum()
    R = x.size // 2
    p = np.pad(rho, R, mode="reflect", reflect_type="odd")
    gu = _correlate_valid(_correlate_valid(p, k1, axis=1), g0, axis=0)
    gv = _correlate_valid(_correlate_valid(p, g0, axis=1), k1, axis=0)
    return gu, gv


def trilateral(rho: np.ndarray, ground: np.ndarray, rho_floor: float, sigma_s: float, sigma_r: float,
               sigma_g: float) -> np.ndarray:
    """Tangent-plane ("trilateral") bilateral filter of rho on non-sky pixels.

    rho_f(x) = rho(x) + sum_d w(x, d) e(x, d) / sum_d w(x, d), with the deviation from the local tangent plane
    e(x, d) = rho(x+d) - rho(x) - grad rho(x) . d and weights
    w = exp(-|d|^2 / 2 sigma_s^2) exp(-(e / max(rho(x), rho_floor))^2 / 2 sigma_r^2) [x+d not sky].
    Planes are reproduced exactly (e = 0, even with one-sided weights); the range term is relative, so the
    filter commutes with a global depth scale; deviations well above sigma_r (steps, sharp folds) get no weight.
    """
    H, W = rho.shape
    gu, gv = gaussian_gradient(rho, sigma_g)
    Rs = int(math.ceil(3 * sigma_s))
    P = np.pad(rho, Rs, mode="reflect", reflect_type="odd")
    G = np.pad(ground, Rs, mode="edge")
    inv = 1.0 / (np.maximum(rho, rho_floor) * sigma_r)
    num = np.zeros_like(rho)
    wsum = np.zeros_like(rho)
    for dy in range(-Rs, Rs + 1):
        for dx in range(-Rs, Rs + 1):
            if dx * dx + dy * dy > Rs * Rs:
                continue
            ws = math.exp(-(dx * dx + dy * dy) / (2 * sigma_s**2))
            nb = P[Rs + dy:Rs + dy + H, Rs + dx:Rs + dx + W]
            e = nb - rho - (gu * dx + gv * dy)
            w = ws * np.exp(-0.5 * (e * inv) ** 2) * G[Rs + dy:Rs + dy + H, Rs + dx:Rs + dx + W]
            num += w * e
            wsum += w
    out = rho.copy()
    out[ground] = rho[ground] + num[ground] / wsum[ground]
    return out


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

    # B3 (optionally on the trilateral-filtered disparity)
    rho_floor = p.rho_floor_frac * float(np.percentile(rho[ground], p.rho_floor_percentile))
    sky_interior = erode(S, r)

    band_replace = p.scale_band and p.scale_band_mode == "replace"
    band_veto = p.scale_band and p.scale_band_mode == "veto"
    if p.scale_band and not (band_replace or band_veto):
        raise ValueError(f"unknown scale_band_mode {p.scale_band_mode!r}")

    def response(rh: np.ndarray, band: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rs, L_ = smooth_and_laplacian(rh, p.sigma, p.truncate)
        if band:
            # Scale-ratio test: a constant-curvature (quadratic) surface has the same Laplacian at every scale and
            # cancels exactly; a fold keeps 1 - 1/f of its response (f = band_factor), a step about 1 - 1/f^2.
            L_ = L_ - smooth_and_laplacian(rh, p.band_factor * p.sigma, p.truncate)[1]
        lor = L_ / np.maximum(rs, rho_floor)
        lor[sky_interior] = 0.0
        return rs, L_, lor

    if p.prefilter == "trilateral":
        rho_f = trilateral(rho, ground, rho_floor, p.tri_sigma_s, p.tri_sigma_r, p.tri_sigma_g)
    elif p.prefilter == "none":
        rho_f = rho
    else:
        raise ValueError(f"unknown prefilter {p.prefilter!r}")

    # B4
    rho_sigma, L, L_over_rho = response(rho_f, band=band_replace)
    q = np.abs(L_over_rho)
    q_plain = q  # response without the veto: also the noise source in veto mode
    if band_veto:
        # Veto: keep q where the response sharpens with scale (steps, folds, fine texture), remove it where it
        # does not (smooth curvature). The gain restores an ideal fold's band response to its full value.
        band_gain = 1.0 / (1.0 - 1.0 / p.band_factor)
        q = np.minimum(q, band_gain * np.abs(response(rho_f, band=True)[2]))
        L_over_rho = np.sign(L_over_rho) * q
    if p.prefilter == "none" or p.tri_shat == "filtered":
        q_noise = q_plain if band_veto else q
    else:
        q_noise = np.abs(response(rho, band=band_replace)[2])

    # B5
    valid = ~dilate(S, r)
    valid[:R, :] = valid[-R:, :] = False
    valid[:, :R] = valid[:, -R:] = False
    n_valid = int(valid.sum())
    if n_valid:
        s_hat_raw = float(np.median(q_noise[valid]) / MAD_TO_STD)
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
    tau_noise = p.k_tau * s_hat
    tau_min = threshold_floor(p.a_min, p.lambda_ref, p.sigma) if p.a_min is not None else None
    tau = tau_noise if tau_min is None else max(tau_noise, tau_min)
    eps_noise = p.k_eps * s_hat
    eps_min = (p.k_eps / p.k_tau) * tau_min if (p.eps_floor and tau_min is not None) else None
    eps = eps_noise if eps_min is None else max(eps_noise, eps_min)
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
        **({} if p.prefilter == "none" else {"rho_filtered": rho_f}),
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
            **({} if tau_min is None else {
                "tau_noise": tau_noise, "tau_min": tau_min, "tau_min_over_tau_noise": tau_min / tau_noise,
                "a_min": p.a_min, "lambda_ref": p.lambda_ref}),
            **({} if eps_min is None else {"eps_noise": eps_noise, "eps_min": eps_min, "eps_floor": True}),
            **({} if not p.scale_band else {"scale_band": True, "band_factor": p.band_factor,
                                            "scale_band_mode": p.scale_band_mode,
                                            **({"band_gain": band_gain} if band_veto else {})}),
            **({} if p.prefilter == "none" else {
                "prefilter": p.prefilter, "tri_shat": p.tri_shat,
                "s_hat_filtered_response": float(np.median(q[valid]) / MAD_TO_STD) if n_valid else None}),
        },
    }
