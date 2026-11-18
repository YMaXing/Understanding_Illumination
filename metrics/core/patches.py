"""Patch statistics (DESIGN.md §2 as amended by C3).

A patch is centred on an integer pixel (x = column, y = row). Shape (config patch.shape):
  square  all (dx, dy) with |dx|, |dy| <= r, i.e. (2r+1)^2 pixels (C3 default; the old tool's patch)
  disc    all (dx, dy) with dx^2 + dy^2 <= r^2
Pixels outside the image count as invalid. P(ℓX) = per-channel median over the valid pixels (np.median, as the old
exporter). A patch is invalid if fewer than min_valid_frac of its pixels are valid.
SE (per channel) = se_median_factor * mad_to_sigma * MAD / sqrt(n), MAD = median |x - median|, n = valid count.
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def offsets(shape, r):
    if shape not in ("square", "disc"):
        raise ValueError(f"patch shape {shape!r}")
    r = int(r)
    d = np.arange(-r, r + 1)
    dy, dx = np.meshgrid(d, d, indexing="ij")
    keep = np.ones(dx.shape, bool) if shape == "square" else (dx * dx + dy * dy <= r * r)
    return dy[keep].ravel(), dx[keep].ravel()


def patch_pixels(valid_shape, x, y, r, shape):
    """(rows, cols) of the in-bounds patch pixels and the nominal pixel count (in- and out-of-bounds)."""
    dy, dx = offsets(shape, r)
    rr, cc = y + dy, x + dx
    H, W = valid_shape
    inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    return rr[inb], cc[inb], len(dy)


def patch_stats(X, valid, x, y, r, P, I=None, noise=None):
    """Statistics of map X (H, W, C) over the patch at (x, y), radius / half-size r. P = config['patch'].
    se = MAD-based SE (§2 formula; on ℓI this is se_tex's per-end term). With raw I (DN) and the sensor-noise model
    noise = [(g, r2)] per channel, also se_noise = factor * sqrt(mean_i (g I_i + r2) / I_i^2 / n) (Delta Q1)."""
    rr, cc, n_total = patch_pixels(valid.shape, int(x), int(y), int(r), P["shape"])
    v = valid[rr, cc]
    vals = X[rr[v], cc[v]]
    n = int(v.sum())
    C = X.shape[-1]
    if n:
        med = np.median(vals, axis=0)
        mad = np.median(np.abs(vals - med), axis=0)
        se = P["se_median_factor"] * P["mad_to_sigma"] * mad / np.sqrt(n)
    else:
        med = mad = se = np.full(C, np.nan)
    out = dict(median=med, mad=mad, se=se, n_valid=n, n_total=n_total, n_inbounds=len(rr), frac_valid=n / n_total,
               ok=bool(n / n_total >= P["min_valid_frac"]))
    if I is not None and noise is not None:
        if n:
            Iv = np.maximum(I[rr[v], cc[v]].astype(np.float64), 1.0)
            g = np.array([p[0] for p in noise])
            r2 = np.array([p[1] for p in noise])
            var_log = (g * Iv + r2) / Iv ** 2
            out["se_noise"] = P["se_median_factor"] * np.sqrt(var_log.mean(axis=0) / n)
        else:
            out["se_noise"] = np.full(C, np.nan)
    return out


def combined_se(sa, sb, key):
    """RMS over channels of sqrt(SE_A^2 + SE_B^2): the per-channel SD of Δ, channel-normed. NaN if a patch is empty."""
    return float(np.sqrt(np.mean(sa[key] ** 2 + sb[key] ** 2)))


def low_snr(sa, sb, tau, divisor, key="se_noise"):
    """Noise guard on ℓI patches: combined SE > tau / divisor (NaN -> True). key "se_noise" (Delta Q1) or "se"."""
    q = combined_se(sa, sb, key)
    return bool(not (q <= tau / divisor)), q


def brightness(v):
    """b(v) = mean over channels (§2)."""
    return float(np.mean(v))
