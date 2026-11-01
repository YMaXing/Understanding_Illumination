"""Bilinear sampling at subpixel positions (exact for affine functions)."""

from __future__ import annotations

import numpy as np


def bilinear(arr: np.ndarray, valid: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample arr (H, W[, C]) at (u, v) (any shape). Returns (values, ok).

    ok: all four corners inside the image and valid; values are NaN where not ok.
    """
    H, W = valid.shape
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u0f, v0f = np.floor(u), np.floor(v)
    inside = (u0f >= 0) & (v0f >= 0) & (u0f <= W - 2) & (v0f <= H - 2)
    u0 = np.clip(u0f, 0, W - 2).astype(np.intp)
    v0 = np.clip(v0f, 0, H - 2).astype(np.intp)
    fu, fv = u - u0f, v - v0f
    ok = inside & valid[v0, u0] & valid[v0, u0 + 1] & valid[v0 + 1, u0] & valid[v0 + 1, u0 + 1]
    w00, w01, w10, w11 = (1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv
    if arr.ndim == 3:
        w00, w01, w10, w11 = (w[..., None] for w in (w00, w01, w10, w11))
    with np.errstate(invalid="ignore"):
        out = (w00 * arr[v0, u0] + w01 * arr[v0, u0 + 1] + w10 * arr[v0 + 1, u0] + w11 * arr[v0 + 1, u0 + 1])
    out[~ok] = np.nan
    return out, ok
