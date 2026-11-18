"""Path profile and excursion (DESIGN.md §2).

Sampling: n = ceil(T) + 1 points evenly spaced from A (t = 0) to B (t = T), so the step is <= 1 px (T = |B - A|).
At each t, 2w + 1 band samples at perpendicular offsets -w..w px (w = band_half_width_px), each bilinear-interpolated;
a band sample is valid only if every neighbour with non-zero weight is a valid pixel. Per channel, the median across
the valid band samples; then Gaussian smoothing along t (sigma_t px) as a normalized convolution over the t that
have a valid band median (no padding at the ends).

One degree of freedom per pixel (§1): p_L is defined as p_I - p_R (both use the same samples and weights), so the
identity holds along the profile exactly, as ΔℓL = ΔℓI - ΔℓR does at the endpoints.

Excursion H_X = max_t of the channel-norm distance from p_X(t) to the segment [P_A(ℓX), P_B(ℓX)] in C-space.
"""

import math
import warnings

import numpy as np


def channel_norm(v, axis=-1):
    return np.sqrt(np.mean(np.square(v), axis=axis))


def _bilinear(X, valid, xs, ys):
    H, W = valid.shape
    inb = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)
    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, max(W - 2, 0))
    y0 = np.clip(np.floor(ys).astype(int), 0, max(H - 2, 0))
    fx, fy = xs - x0, ys - y0
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    out = 0.0
    ok = inb.copy()
    for yy, xx, w in ((y0, x0, (1 - fy) * (1 - fx)), (y0, x1, (1 - fy) * fx), (y1, x0, fy * (1 - fx)), (y1, x1, fy * fx)):
        ok &= valid[yy, xx] | (w == 0)
        out = out + w[..., None] * X[yy, xx]
    return out, ok


def sample_band(maps, valid, A, B, w):
    """Band medians along A->B for each map in `maps` (name -> (H, W, C)). Returns t (n,), {name: (n, C)}, n_ok (n,)."""
    ax, ay, bx, by = float(A["x"]), float(A["y"]), float(B["x"]), float(B["y"])
    T = math.hypot(bx - ax, by - ay)
    n = int(math.ceil(T)) + 1 if T > 0 else 1
    t = np.linspace(0.0, T, n)
    ux, uy = ((bx - ax) / T, (by - ay) / T) if T > 0 else (1.0, 0.0)
    nx, ny = -uy, ux
    s = np.arange(-w, w + 1, dtype=float)
    xs = ax + t[:, None] * ux + s[None, :] * nx
    ys = ay + t[:, None] * uy + s[None, :] * ny
    out = {}
    ok = None
    for name, X in maps.items():
        vals, ok = _bilinear(X, valid, xs, ys)            # (n, 2w+1, C); ok depends only on valid
        v = np.where(ok[..., None], vals, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)        # all-NaN band -> NaN median
            out[name] = np.nanmedian(v, axis=1)
    return t, out, ok.sum(axis=1)


def smooth(t, p, sigma):
    """Normalized Gaussian convolution along t over finite samples; NaN where no finite sample has weight."""
    fin = np.all(np.isfinite(p), axis=-1)
    if sigma <= 0:
        return p.copy()
    W = np.exp(-0.5 * ((t[:, None] - t[None, :]) / sigma) ** 2) * fin[None, :]
    den = W.sum(axis=1)
    num = W @ np.where(fin[:, None], p, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den[:, None] > 0, num / den[:, None], np.nan)


def profiles(lI, lR, valid, A, B, P, sigma_t=None):
    """p_I, p_R, p_L (= p_I - p_R) along A->B. P = config['profile']; sigma_t overrides P['sigma_t_px'].
    Returns dict(t, I, R, L, n_ok, sigma_t)."""
    s = P["sigma_t_px"] if sigma_t is None else sigma_t
    t, bm, n_ok = sample_band({"I": lI, "R": lR}, valid, A, B, P["band_half_width_px"])
    pI = smooth(t, bm["I"], s)
    pR = smooth(t, bm["R"], s)
    return dict(t=t, I=pI, R=pR, L=pI - pR, n_ok=n_ok, sigma_t=s)


def excursion(p, a, b):
    """max_t channel-norm distance from p(t) (n, C) to the segment [a, b]; NaN if no finite p(t)."""
    d = b - a
    dd = float(d @ d)
    if dd > 0:
        s = np.clip(((p - a) @ d) / dd, 0.0, 1.0)
        q = a + s[:, None] * d
    else:
        q = a[None, :]
    dist = channel_norm(p - q)
    return float(np.nanmax(dist)) if np.isfinite(dist).any() else float("nan")


def _pav_increasing(y):
    """Pool-adjacent-violators: least-squares non-decreasing fit of the 1-D array y."""
    vals, wts, lens = [], [], []
    for v in y:
        vals.append(float(v))
        wts.append(1.0)
        lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            n2 = lens[-2] + lens[-1]
            del vals[-1], wts[-1], lens[-1]
            vals[-1], wts[-1], lens[-1] = v2, w, n2
    return np.repeat(vals, lens)


def isotonic(y):
    """Per channel, the better (smaller SSE) of the non-decreasing and non-increasing least-squares fits."""
    up = _pav_increasing(y)
    dn = -_pav_increasing(-y)
    return up if np.sum((y - up) ** 2) <= np.sum((y - dn) ** 2) else dn


def iso_excursion(p):
    """H_ref = max_t channel-norm of p(t) - iso(t) (Checkpoint 2, D1 alternative): the input's wiggle about its own
    per-channel monotone fit (removes the step, keeps the texture). NaN if no finite sample."""
    fin = np.all(np.isfinite(p), axis=-1)
    if not fin.any():
        return float("nan")
    q = p[fin]
    fit = np.stack([isotonic(q[:, c]) for c in range(q.shape[1])], axis=-1)
    return float(np.max(channel_norm(q - fit)))
