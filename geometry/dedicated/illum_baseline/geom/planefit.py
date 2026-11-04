"""Plane fits on the filtered disparity rho_f.

N1 (dense_fit): at every pixel x, the edge-aware weighted LS plane over offsets |d| <= R_n = ceil(3 sigma_n),

    (alpha, beta) = argmin sum_d W(x, d) [rho_f(x + d) - alpha - beta.d]^2,

with W the trilateral weights of detector.tangent_plane_weights evaluated on rho_f (spatial sigma_n, relative range
sigma_r on the deviation from the tangent plane of a gradient g, sky excluded).

The gradient g (g_init = "side_window", the default): a Gaussian-derivative gradient sees a depth step up to its
support (12 px), so the range weights then reject the pixel's own side too and the fit becomes invalid in a band
along every step (sum W < min_weight_sum within ~3 px of a 1.5x step). Instead g starts from the best of 8 half-disk
("side window") LS fits of radius R_n, one every 45 deg, the one with the smallest relative residual; the weighted
fit runs with that g, then once more (refit) with g = beta of the first pass. g_init = "gaussian" is the plain
sigma_g Gaussian-derivative gradient (F1 behaviour). On a plane every choice gives the exact plane.

Implementation: the regressand is centred on the tangent plane, e(x, d) = rho_f(x + d) - rho_f(x) - g(x).d. Since
rho_f(x) + g(x).d lies in the column space of the design matrix A = [1, d], the WLS solution is
(alpha, beta) = (rho_f(x), g(x)) + theta_e with theta_e the fit of e: the same estimator, but sum W r^2 is accumulated
in one pass from moments of e (of the size of the residuals) instead of rho_f (no cancellation).
Covariance (sandwich, the weights are not inverse variances):
    Cov = s^2 M^-1 Q M^-1, M = A^T W A, Q = A^T W^2 A, s^2 = (sum W r^2 / sum W) N_eff / (N_eff - 3),
    N_eff = (sum W)^2 / sum W^2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..detector import dilate, gaussian_gradient, tangent_plane_weights

N_SIDES = 8  # half-disk side windows, one every 45 deg


@dataclass(frozen=True)
class DenseFitParams:
    sigma_n: float = 3.0         # px, spatial scale of the fit window
    sigma_r: float = 0.01        # relative deviation from the tangent plane
    sigma_g: float = 3.0         # px, scale of the tangent-plane gradient
    cond_max: float = 1.0e6      # condition number of M = A^T W A (d in px)
    min_weight_sum: float = 5.0  # minimum sum W
    g_init: str = "side_window"  # side_window | gaussian: tangent-plane gradient of the first pass
    refit: int = 1               # further passes with g = beta of the previous pass

    @property
    def R_n(self) -> int:
        return int(math.ceil(3 * self.sigma_n))


@dataclass(frozen=True)
class DenseFit:
    alpha: np.ndarray   # (H, W) fitted disparity at x; NaN where invalid
    beta: np.ndarray    # (H, W, 2) (d rho/du, d rho/dv) per px
    cov: np.ndarray     # (H, W, 3, 3) Cov(alpha, beta_u, beta_v)
    m0: np.ndarray      # (H, W) sum W
    n_eff: np.ndarray   # (H, W)
    s2: np.ndarray      # (H, W)
    cond: np.ndarray    # (H, W) condition number of M; NaN where not evaluated
    valid: np.ndarray   # (H, W) bool


def _sym3(a0, a1, a2, a11, a12, a22) -> np.ndarray:
    """(n, 3, 3) from the moments of [1, d_u, d_v]: [[a0, a1, a2], [a1, a11, a12], [a2, a12, a22]]."""
    return np.stack([np.stack([a0, a1, a2], -1), np.stack([a1, a11, a12], -1), np.stack([a2, a12, a22], -1)], -2)


def side_window_gradient(rho_f: np.ndarray, ground: np.ndarray, p: DenseFitParams
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(g_u, g_v, ok): slope of the best of N_SIDES half-disk LS planes per pixel.

    Half-disk j: offsets |d| <= R_n with d . omega_j >= 0, omega_j at 45 j deg; weights exp(-|d|^2 / 2 sigma_n^2)
    [x+d not sky]; regressand rho_f(x+d) - rho_f(x) on (1, d). Picked: the smallest relative residual
    sum w r^2 / (sum w rho_f(x)^2) among half-disks with sum w >= min_weight_sum. ok = False where none qualifies.
    """
    H, W = rho_f.shape
    R = p.R_n
    P = np.pad(rho_f, R, mode="reflect", reflect_type="odd")
    G = np.pad(ground, R, mode="edge").astype(np.float64)
    ang = np.arange(N_SIDES) * (2 * np.pi / N_SIDES)
    om = np.stack([np.cos(ang), np.sin(ang)], -1)
    acc = np.zeros((N_SIDES, 10, H, W))  # S0 Su Sv Suu Suv Svv B0 Bu Bv Yy
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if dx * dx + dy * dy > R * R:
                continue
            sides = np.nonzero(om @ np.array([dx, dy], float) >= -1e-9)[0]
            ws = math.exp(-(dx * dx + dy * dy) / (2 * p.sigma_n**2))
            w = ws * G[R + dy:R + dy + H, R + dx:R + dx + W]
            y = P[R + dy:R + dy + H, R + dx:R + dx + W] - rho_f
            wy = w * y
            terms = (w, dx * w, dy * w, (dx * dx) * w, (dx * dy) * w, (dy * dy) * w, wy, dx * wy, dy * wy, wy * y)
            for j in sides:
                for i, tm in enumerate(terms):
                    acc[j, i] += tm
    best = np.full((H, W), np.inf)
    gu, gv = np.full((H, W), np.nan), np.full((H, W), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(ground, rho_f**2, np.nan)
    for j in range(N_SIDES):
        a = acc[j]
        M = _sym3(a[0], a[1], a[2], a[3], a[4], a[5]).reshape(-1, 3, 3)
        b = np.stack([a[6], a[7], a[8]], -1).reshape(-1, 3)
        det = np.linalg.det(M)
        ok = (a[0].ravel() >= p.min_weight_sum) & (det > 1e-9 * np.maximum(a[0].ravel(), 1e-300) ** 3)
        Ms = np.where(ok[:, None, None], M, np.eye(3))
        th = np.linalg.solve(Ms, b[..., None])[..., 0]
        rss = np.maximum(a[9].ravel() - np.einsum("ni,ni->n", th, b), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            res = np.where(ok, rss / (a[0].ravel() * scale.ravel()), np.inf).reshape(H, W)
        better = res < best
        best[better] = res[better]
        gu[better] = th[:, 1].reshape(H, W)[better]
        gv[better] = th[:, 2].reshape(H, W)[better]
    return gu, gv, np.isfinite(best)


def _weighted_fit(rho_f: np.ndarray, ground: np.ndarray, rho_floor: float, p: DenseFitParams,
                  gu: np.ndarray, gv: np.ndarray) -> DenseFit:
    """One pass of N1 with the tangent-plane gradient (gu, gv)."""
    H, W = rho_f.shape
    R = p.R_n
    z = lambda: np.zeros((H, W))  # noqa: E731
    S0, Su, Sv, Suu, Suv, Svv = z(), z(), z(), z(), z(), z()  # M = A^T W A
    T0, Tu, Tv, Tuu, Tuv, Tvv = z(), z(), z(), z(), z(), z()  # Q = A^T W^2 A
    B0, Bu, Bv, Ee = z(), z(), z(), z()                       # A^T W e, sum W e^2
    for dx, dy, _, e, w in tangent_plane_weights(rho_f, ground, rho_floor, p.sigma_n, p.sigma_r, p.sigma_g,
                                                 grad=(gu, gv)):
        w2 = w * w
        we = w * e
        S0 += w
        T0 += w2
        B0 += we
        Ee += we * e
        if dx:
            Su += dx * w
            Suu += (dx * dx) * w
            Tu += dx * w2
            Tuu += (dx * dx) * w2
            Bu += dx * we
        if dy:
            Sv += dy * w
            Svv += (dy * dy) * w
            Tv += dy * w2
            Tvv += (dy * dy) * w2
            Bv += dy * we
        if dx and dy:
            Suv += (dx * dy) * w
            Tuv += (dx * dy) * w2

    pre = ground & ~dilate(~ground, R) & (S0 >= p.min_weight_sum)
    pre[:R, :] = pre[-R:, :] = False
    pre[:, :R] = pre[:, -R:] = False
    idx = np.nonzero(pre)
    M = _sym3(*(a[idx] for a in (S0, Su, Sv, Suu, Suv, Svv)))
    cond_v = np.linalg.cond(M) if M.shape[0] else np.zeros(0)
    ok = np.isfinite(cond_v) & (cond_v <= p.cond_max)
    cond = np.full((H, W), np.nan)
    cond[idx] = cond_v
    sel = tuple(i[ok] for i in idx)
    M = M[ok]
    Q = _sym3(*(a[sel] for a in (T0, Tu, Tv, Tuu, Tuv, Tvv)))
    b = np.stack([B0[sel], Bu[sel], Bv[sel]], -1)

    theta = np.linalg.solve(M, b[..., None])[..., 0]
    Minv = np.linalg.inv(M)
    m0 = S0[sel]
    n_eff_v = m0**2 / T0[sel]
    rss = np.maximum(Ee[sel] - np.einsum("ni,ni->n", theta, b), 0.0)  # sum W r^2 = sum W e^2 - theta . b
    s2_v = rss / m0 * n_eff_v / (n_eff_v - 3.0)
    cov_v = s2_v[:, None, None] * (Minv @ Q @ Minv)

    valid = np.zeros((H, W), dtype=bool)
    valid[sel] = True
    alpha = np.full((H, W), np.nan)
    beta = np.full((H, W, 2), np.nan)
    cov = np.full((H, W, 3, 3), np.nan)
    n_eff = np.full((H, W), np.nan)
    s2 = np.full((H, W), np.nan)
    alpha[sel] = rho_f[sel] + theta[:, 0]
    beta[sel] = np.stack([gu[sel], gv[sel]], -1) + theta[:, 1:]
    cov[sel] = cov_v
    n_eff[sel] = n_eff_v
    s2[sel] = s2_v
    return DenseFit(alpha=alpha, beta=beta, cov=cov, m0=S0, n_eff=n_eff, s2=s2, cond=cond, valid=valid)


def dense_fit(rho_f: np.ndarray, ground: np.ndarray, rho_floor: float, p: DenseFitParams) -> DenseFit:
    """N1 on every pixel. Invalid: sky, within R_n of the sky or the border, sum W < min_weight_sum, cond > cond_max."""
    rho_f = np.asarray(rho_f, dtype=np.float64)
    gu0, gv0 = gaussian_gradient(rho_f, p.sigma_g)
    if p.g_init == "gaussian":
        gu, gv = gu0, gv0
    elif p.g_init == "side_window":
        gu, gv, ok = side_window_gradient(rho_f, ground, p)
        gu, gv = np.where(ok, gu, gu0), np.where(ok, gv, gv0)
    else:
        raise ValueError(f"unknown g_init {p.g_init!r}")
    fit = _weighted_fit(rho_f, ground, rho_floor, p, gu, gv)
    for _ in range(p.refit):
        gu = np.where(fit.valid, fit.beta[..., 0], gu)
        gv = np.where(fit.valid, fit.beta[..., 1], gv)
        fit = _weighted_fit(rho_f, ground, rho_floor, p, gu, gv)
    return fit


@dataclass(frozen=True)
class WindowFit:
    alpha: np.ndarray    # (N,) fitted value at the centroid
    beta: np.ndarray     # (N, 2)
    c: np.ndarray        # (N, 2) sample centroid (u, v)
    cov: np.ndarray      # (N, 3, 3) s^2 (A^T A)^-1 (block diagonal: the design is centred)
    rms_rel: np.ndarray  # (N,) rms(residual) / alpha
    n: np.ndarray        # (N,) samples used
    ok: np.ndarray       # (N,) bool: enough samples and a non-degenerate design


def window_fit(values: np.ndarray, xy: np.ndarray, mask: np.ndarray) -> WindowFit:
    """Ordinary LS plane per row: values (N, S) at positions xy (N, S, 2), samples where mask (N, S).

    rho = alpha + beta.(x - c), c the sample centroid; Cov = s^2 (A^T A)^-1, s^2 = RSS / (n - 3).
    """
    m = mask.astype(np.float64)
    n = m.sum(1)
    ok = n >= 4
    nn = np.where(ok, n, 1.0)
    y = np.where(mask, values, 0.0)
    c = (m[..., None] * xy).sum(1) / nn[:, None]
    d = (xy - c[:, None, :]) * m[..., None]
    alpha = y.sum(1) / nn
    Suu, Suv, Svv = (d[..., 0] ** 2).sum(1), (d[..., 0] * d[..., 1]).sum(1), (d[..., 1] ** 2).sum(1)
    det = Suu * Svv - Suv**2
    ok &= det > 1e-9 * np.maximum(Suu * Svv, 1e-300)
    dd = np.where(ok, det, 1.0)
    ry = (y - alpha[:, None]) * m
    bu, bv = (d[..., 0] * ry).sum(1), (d[..., 1] * ry).sum(1)
    beta = np.stack([(Svv * bu - Suv * bv) / dd, (Suu * bv - Suv * bu) / dd], -1)
    r = (ry - (d * beta[:, None, :]).sum(-1)) * m
    rss = (r**2).sum(1)
    s2 = rss / np.maximum(n - 3, 1.0)
    cov = np.zeros((len(n), 3, 3))
    cov[:, 0, 0] = s2 / nn
    cov[:, 1, 1] = s2 * Svv / dd
    cov[:, 2, 2] = s2 * Suu / dd
    cov[:, 1, 2] = cov[:, 2, 1] = -s2 * Suv / dd
    with np.errstate(invalid="ignore", divide="ignore"):
        rms_rel = np.sqrt(rss / nn) / alpha
    nan = ~ok
    alpha[nan], beta[nan], c[nan], cov[nan], rms_rel[nan] = np.nan, np.nan, np.nan, np.nan, np.nan
    return WindowFit(alpha=alpha, beta=beta, c=c, cov=cov, rms_rel=rms_rel, n=n.astype(int), ok=ok)
