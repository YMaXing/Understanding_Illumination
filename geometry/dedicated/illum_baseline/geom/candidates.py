"""Line candidates for the geometric-edge channels.

F2.1 fold candidates: structure tensor of the dense normals, T_s = sum_c G_sigma_T * (grad n_c grad n_c^T), with
Gaussian derivatives at sigma_d on valid normals only (pixels whose derivative support holds an invalid normal are
excluded) and normalized convolution at sigma_T. Largest eigenvalue lambda_1 and its unit eigenvector e1 (across the
fold); rotation rate r_n = sqrt(lambda_1) (rad/px). Non-maximum suppression along e1 (bilinear samples at x +- e1),
r_n > cand_rate_factor * median(r_n), subpixel x0 from a parabola along e1.

Sign convention: e1 = (cos phi, sin phi) with phi in (-90, 90] deg, i.e. e1_u > 0, or e1 = (0, 1). Side "+" is +e1.

F2.2 rounding width: the 10%-90% distance of the normalized normal-angle profile along e1.

Depth candidates (TASK_DEPTH_CHANNEL D-K, D-1, D-2, D-3): depth_kernels, depth_field (q, the detrended orientation n
and the gradient floor), depth_candidates (zero crossings of q, polarity, lobes, two-sided height, sharpness).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..detector import _correlate_valid, dilate, gaussian_gradient, kernels
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

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


RAYLEIGH_MEDIAN = math.sqrt(2.0 * math.log(2.0))  # median of |g| for g ~ N(0, sigma^2 I_2), in units of sigma


@dataclass(frozen=True)
class DepthField:
    rho_s: np.ndarray       # (H, W) G_sigma * rho_src
    H_uu: np.ndarray        # Hessian of rho_s
    H_vv: np.ndarray
    H_uv: np.ndarray
    grad: np.ndarray        # (H, W, 2) the orientation gradient: grad rho_s - grad rho_{sigma_L} (detrended) or grad rho_s
    grad_norm: np.ndarray   # (H, W)
    n: np.ndarray           # (H, W, 2) grad / |grad| (toward larger rho, the near side); (1, 0) where |grad| = 0
    q: np.ndarray           # (H, W) n^T H n / max(rho_s, rho_floor), signed (not zeroed on the sky: rho = 0 there)
    h: np.ndarray           # (H, W) n^T H n (unnormalized): the lobe positions (D3)
    den: np.ndarray         # (H, W) max(rho_s, rho_floor)
    g_n: np.ndarray         # (H, W) |grad| / max(rho_s, rho_floor), the normalized gradient
    valid: np.ndarray       # (H, W) Omega and g_n > k_g sigma_g
    s_q: float              # median(|q| over Omega, away from the sky and the border) / 0.6745
    sigma_g: float          # median(g_n over Omega) / sqrt(2 ln 2) (Rayleigh)
    g_floor: float          # k_g sigma_g
    sigma_L: float
    clean: np.ndarray       # (H, W) Omega at least R px from the sky and the border (the s_q sample)


def depth_field(ctx: GeometryContext, p, detrend: bool = True) -> DepthField:
    """D-K, D-1, D-2 on the context's disparity (p.disparity: "rho_f", the default and the fold channel's field, or
    "rho" for ablations). p: DepthParams (sigma, truncate, sigma_w_prior, sigma_L_factor, k_g, disparity).

    Gradient floor (D2): g_n = |grad Dd| / max(rho_s, rho_floor), sigma_g = median(g_n over Omega) / 1.1774 (the
    Rayleigh median), valid where g_n > k_g sigma_g (k_g = 3: about 1.1% of pure-noise pixels pass).
    q is computed everywhere: rho = 0 on the sky (the far limit), so a sky boundary is a step like any other; the
    D-6 sky rules decide what it means.
    """
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
    den = np.maximum(rho_s, ctx.rho_floor)
    h = nu * nu * H_uu + 2 * nu * nv * H_uv + nv * nv * H_vv
    q = h / den
    g_n = gnorm / den
    sigma_g = float(np.median(g_n[omega]) / RAYLEIGH_MEDIAN) if omega.any() else 0.0
    g_floor = p.k_g * sigma_g
    valid = omega & (g_n > g_floor)
    R = int(math.ceil(p.truncate * p.sigma))
    clean = omega & ~dilate(~omega, R)
    clean[:R, :] = clean[-R:, :] = False
    clean[:, :R] = clean[:, -R:] = False
    s_q = float(np.median(np.abs(q[clean])) / MAD_TO_STD) if clean.any() else float("nan")
    return DepthField(rho_s=rho_s, H_uu=H_uu, H_vv=H_vv, H_uv=H_uv, grad=np.stack([gu, gv], -1), grad_norm=gnorm,
                      n=np.stack([nu, nv], -1), q=q, h=h, den=den, g_n=g_n, valid=valid, s_q=s_q, sigma_g=sigma_g, g_floor=g_floor,
                      sigma_L=sigma_L, clean=clean)


DEPTH_CAND_STAGES = ("polarity", "height", "sharpness", "centring", "symmetry")  # D-3 reject reasons, in order


@dataclass(frozen=True)
class DepthCandidates:
    """Every zero crossing of q above the gradient floor (merged within merge_radius), with its D-3 outcome."""
    x0: np.ndarray        # (N, 2) subpixel (u, v)
    n: np.ndarray         # (N, 2) unit, toward the near side (larger rho); side "+" is +n
    ij: np.ndarray        # (N, 2) int (u, v): x0 rounded
    g_n: np.ndarray       # (N,) normalized gradient at x0
    rho_s0: np.ndarray    # (N,) rho_sigma(x0)
    q_p: np.ndarray       # (N,) q at the positive-lobe peak (>= 0; 0 if no lobe), at s_p < 0
    q_m: np.ndarray       # (N,) q at the negative-lobe peak (<= 0), at s_m > 0
    h_p: np.ndarray       # (N,) n^T H n at the peaks (the doublet magnitude: Delta rho = 1.03 |h|_lobe s_sep^2)
    h_m: np.ndarray
    s_p: np.ndarray
    s_m: np.ndarray
    s_sep: np.ndarray     # (N,) s_m - s_p
    sigma_wc: np.ndarray  # (N,) sqrt(max(0, (s_sep/2)^2 - sigma^2))
    k_c: np.ndarray       # (N,) int, max(k_min, ceil(2 sigma_wc) + 1)
    reason: np.ndarray    # (N,) object: "" (passed D-3) or the first failed stage (DEPTH_CAND_STAGES)
    counts: dict          # n_cross, n_floor, n_cand, then the survivors of each stage (n_polarity ... n_passed)
    s_lo: float
    s_hi: float
    field: DepthField

    @property
    def passed(self) -> np.ndarray:
        return self.reason == ""


def _crossings(q: np.ndarray, g_n: np.ndarray, omega: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Sign changes of q between pixel pairs along axis (1: (u, v)-(u+1, v); 0: (u, v)-(u, v+1)) with at least one
    pixel in Omega: (x0 (N, 2) (u, v), g_n linearly interpolated at x0 (N,))."""
    a = [slice(None), slice(None)]
    b = [slice(None), slice(None)]
    a[axis], b[axis] = slice(None, -1), slice(1, None)
    qi, qj = q[tuple(a)], q[tuple(b)]
    hit = (qi * qj < 0) & (omega[tuple(a)] | omega[tuple(b)])
    vv, uu = np.nonzero(hit)
    t = qi[vv, uu] / (qi[vv, uu] - qj[vv, uu])
    g = (1 - t) * g_n[tuple(a)][vv, uu] + t * g_n[tuple(b)][vv, uu]
    x0 = np.stack([uu + t, vv.astype(np.float64)], -1) if axis == 1 else np.stack([uu.astype(np.float64), vv + t], -1)
    return x0, g


def _merge(x0: np.ndarray, g: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge points within radius of each other (single linkage): the mean position, the max g."""
    if len(x0) < 2:
        return x0, g
    pairs = cKDTree(x0).query_pairs(radius, output_type="ndarray")
    if not len(pairs):
        return x0, g
    N = len(x0)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(N, N))
    nc, lab = connected_components(adj, directed=False)
    cnt = np.bincount(lab, minlength=nc)
    xm = np.stack([np.bincount(lab, x0[:, i], minlength=nc) / cnt for i in (0, 1)], -1)
    gm = np.full(nc, -np.inf)
    np.maximum.at(gm, lab, g)
    return xm, gm


def _lobe(qs: np.ndarray, ok: np.ndarray, sign: int, step: float, rel_peak: float = 0.5
          ) -> tuple[np.ndarray, np.ndarray]:
    """Peak of the lobe of sign `sign` contiguous with x0 along one direction. qs, ok: (N, L) samples at s = (j+1)
    step, j = 0..L-1, moving away from x0. The peak is the nearest local maximum of the run that reaches rel_peak of
    the run's maximum (D3: inside a thin structure the near lobes of both edges form one run; the nearest peak is the
    candidate's own, and small noise wiggles on the flank are skipped). Returns (peak |q| (0 if no lobe), distance
    |s| of the peak, parabola-refined)."""
    N, L = qs.shape
    y = np.where(ok, sign * qs, -np.inf)
    run = np.cumprod(y > 0, axis=1).astype(bool)
    yr = np.where(run, y, -np.inf)
    ymax = yr.max(axis=1)
    nxt = np.concatenate([yr[:, 1:], np.full((N, 1), -np.inf)], 1)
    prv = np.concatenate([np.full((N, 1), -np.inf), yr[:, :-1]], 1)
    local = run & (yr >= nxt) & (yr >= prv) & (yr >= rel_peak * ymax[:, None])
    j = np.where(local.any(1), np.argmax(local, axis=1), np.argmax(yr, axis=1))
    has = run[:, 0]
    r = np.arange(N)
    y0 = yr[r, j]
    jm, jp = np.maximum(j - 1, 0), np.minimum(j + 1, L - 1)
    ym, yp = y[r, jm], y[r, jp]
    inner = (j > 0) & (j < L - 1) & np.isfinite(ym) & np.isfinite(yp)
    with np.errstate(invalid="ignore", divide="ignore"):
        curv = ym - 2 * y0 + yp
        d = np.clip(np.where(inner & (curv < 0), 0.5 * (ym - yp) / curv, 0.0), -0.5, 0.5)
        peak = np.where(inner, y0 - 0.25 * (ym - yp) * d, y0)
    peak = np.where(has, peak, 0.0)
    dist = np.where(has, (j + d + 1) * step, np.nan)
    return peak, dist


def depth_candidates(ctx: GeometryContext, p, field: DepthField | None = None, merge_radius: float = 0.5
                     ) -> DepthCandidates:
    """D-3: zero crossings of q (subpixel, merged within merge_radius px) at or above the gradient floor; polarity
    q(x0 - n/2) > 0 > q(x0 + n/2); lobes along x0 + s n, s in [-L, L] (step p.lobe_step), each the peak of the run of
    one sign contiguous with x0, located on h = n^T H n and measured in q; two-sided height
    min(|q_p|, |q_m|) > k_h s_q; sharpness s_lo <= s_sep <= s_hi;
    centring |s_p + s_m| <= centre_tol s_sep; optional symmetry. p: DepthParams."""
    f = depth_field(ctx, p) if field is None else field
    H, W = f.q.shape
    xs, gs = zip(*(_crossings(f.q, f.g_n, ctx.omega, ax) for ax in (1, 0)))
    x0, g = np.concatenate(xs), np.concatenate(gs)
    n_cross = len(x0)
    keep = g > f.g_floor
    x0, g = _merge(x0[keep], g[keep], merge_radius)
    n_floor = int(keep.sum())
    inside = (x0[:, 0] >= 1) & (x0[:, 0] <= W - 2) & (x0[:, 1] >= 1) & (x0[:, 1] <= H - 2)
    x0, g = x0[inside], g[inside]
    N = len(x0)
    all_px = np.ones((H, W), dtype=bool)
    nn, _ = bilinear(f.n, all_px, x0[:, 0], x0[:, 1])
    nn = nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-300)
    rho0, _ = bilinear(f.rho_s, all_px, x0[:, 0], x0[:, 1])
    reason = np.full(N, "", dtype=object)

    qa, oka = bilinear(f.q, all_px, x0[:, 0] - nn[:, 0] / 2, x0[:, 1] - nn[:, 1] / 2)
    qb, okb = bilinear(f.q, all_px, x0[:, 0] + nn[:, 0] / 2, x0[:, 1] + nn[:, 1] / 2)
    pol = oka & okb & (qa > 0) & (qb < 0)
    reason[~pol] = "polarity"

    L = int(round(p.lobe_half_len / p.lobe_step))
    s = np.arange(1, L + 1) * p.lobe_step
    pts_b = x0[:, None, :] - s[None, :, None] * nn[:, None, :]
    pts_f = x0[:, None, :] + s[None, :, None] * nn[:, None, :]
    # lobe positions on the unnormalized h = n^T H n (D3): q = h / rho_s with rho_s varying across a doublet drags the
    # far lobe outward for large jumps (peak at -2.1 sigma_tot px for r = 5, -3.3 for r = 100); h peaks at
    # -+ sigma_tot for any r. Heights are q at those peaks (scale-free, against s_q).
    hbk, okbk = bilinear(f.h, all_px, pts_b[..., 0], pts_b[..., 1])
    hfw, okfw = bilinear(f.h, all_px, pts_f[..., 0], pts_f[..., 1])
    h_p, d_p = _lobe(hbk, okbk, +1, p.lobe_step)
    h_m, d_m = _lobe(hfw, okfw, -1, p.lobe_step)
    s_p, s_m, h_m = -d_p, d_m, -h_m
    s_sep = s_m - s_p
    dp_, _ = bilinear(f.den, all_px, x0[:, 0] + np.nan_to_num(s_p) * nn[:, 0], x0[:, 1] + np.nan_to_num(s_p) * nn[:, 1])
    dm_, _ = bilinear(f.den, all_px, x0[:, 0] + np.nan_to_num(s_m) * nn[:, 0], x0[:, 1] + np.nan_to_num(s_m) * nn[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        q_p = np.where(np.isfinite(dp_), h_p / dp_, 0.0)
        q_m = np.where(np.isfinite(dm_), h_m / dm_, 0.0)
    height = np.minimum(q_p, -q_m) > p.k_h * f.s_q
    reason[(reason == "") & ~height] = "height"

    s_lo = p.s_lo_factor * 2 * p.sigma
    s_hi = 2 * math.sqrt(p.sigma**2 + p.sigma_w_max**2)
    with np.errstate(invalid="ignore"):
        sharp = (s_sep >= s_lo) & (s_sep <= s_hi)
    reason[(reason == "") & ~sharp] = "sharpness"
    # centring (D2 addition): x0 in the middle of its doublet, |s_p + s_m| <= centre_tol s_sep (a smeared step has its
    # peaks at -+ sigma_tot). Rejects crossings on the decaying flank of a large doublet, where the detrended n has
    # flipped and the borrowed lobe lies ~5 px away (ghost lines next to steps on rho_f; |s_p + s_m| / s_sep ~ 0.54).
    with np.errstate(invalid="ignore"):
        centred = np.abs(s_p + s_m) <= p.centre_tol * s_sep
    reason[(reason == "") & ~centred] = "centring"
    if p.symmetry:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = q_p / -q_m
        sym = (ratio >= 1 / p.symmetry_ratio) & (ratio <= p.symmetry_ratio)
        reason[(reason == "") & ~sym] = "symmetry"

    with np.errstate(invalid="ignore"):
        sigma_wc = np.sqrt(np.maximum(0.0, (s_sep / 2) ** 2 - p.sigma**2))
    k_c = np.maximum(p.k_min, np.ceil(2 * np.nan_to_num(sigma_wc)).astype(int) + 1)
    counts = {"n_cross": n_cross, "n_floor": n_floor, "n_cand": N, "n_polarity": int(pol.sum()),
              "n_height": int((pol & height).sum()), "n_sharpness": int((pol & height & sharp).sum()),
              "n_centring": int((pol & height & sharp & centred).sum()), "n_passed": int((reason == "").sum())}
    return DepthCandidates(x0=x0, n=nn, ij=np.rint(x0).astype(int), g_n=g, rho_s0=rho0, q_p=q_p, q_m=q_m, h_p=h_p,
                           h_m=h_m, s_p=s_p,
                           s_m=s_m, s_sep=s_sep, sigma_wc=sigma_wc, k_c=k_c, reason=reason, counts=counts, s_lo=s_lo,
                           s_hi=s_hi, field=f)
