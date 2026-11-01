"""Two-sided plane fits across a candidate line (F2.3-F2.5), shared by the fold and the future depth channel.

For a candidate at x0 with unit normal-to-the-line direction e1 (side "+" is +e1) the one-sided 2D windows are

    W±(k) = { x0 + s e1 + t e_t :  ±s in [k, k + m],  |t| <= w_t },   e_t = e1 rotated by +90 deg,

sampled on the integer (s, t) grid with bilinear interpolation of rho_f (exact for affine functions). A sample is
excluded if any of its four bilinear corners is sky, has an invalid dense normal, or lies outside the image; a window
with more than max_excluded of its samples excluded is invalid. Each valid window gets an OLS plane (planefit.
window_fit), the contamination check (rms/alpha > rms_max(k, m): retry with m/2, invalid once m < m_min; the caller
supplies rms_max, e.g. max(3 eps_hat, b_tri(k, m), Q95 of null rms)), and N2 normals at its centroid. Invalid windows
fall back to the dense normal at the pixel nearest the nominal window centre (flag `fallback`) when fallback=True.

Delta rho* is the smallest gap between the two fitted planes along the candidate's normal line, s in [-k, k]
(closed form for affine functions); planes_cross marks a sign change of the gap inside that span (a fold-like
intersection: Delta rho* = 0).

This module makes no decisions: thresholds, gates, vetoes and weights belong to the channels.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

import numpy as np

from .context import GeometryContext
from .normals import normals
from .planefit import window_fit
from .sampling import bilinear


@dataclass
class SideResult:
    alpha: np.ndarray         # (N,) plane value at c (fallback: dense alpha at the pixel)
    beta: np.ndarray          # (N, 2)
    c: np.ndarray             # (N, 2) centroid (fallback: the pixel)
    cov: np.ndarray           # (N, 3, 3) Cov(alpha, beta); NaN for fallback
    rms_rel: np.ndarray       # (N,) rms(residual)/alpha of the final planar fit; NaN if none
    n: np.ndarray             # (N,) samples used
    excl: np.ndarray          # (N,) excluded fraction of the last window tried
    m_used: np.ndarray        # (N,) m of the accepted planar window, 0 if none
    planar: np.ndarray        # (N,) bool: planar window fit accepted
    fallback: np.ndarray      # (N,) bool: dense normal used instead
    nhat: np.ndarray          # (N, 3)
    Sigma_n: np.ndarray       # (N, 3, 3)
    sigma_theta: np.ndarray   # (N,)

    @property
    def usable(self) -> np.ndarray:
        return self.planar | self.fallback


@dataclass
class TwoSidedResult:
    p: SideResult             # side +e1
    m: SideResult             # side -e1
    d_rho: np.ndarray         # (N,) Delta rho* = signed gap rho_+ - rho_- at s_gap; NaN unless both sides planar
    d_rho_se: np.ndarray      # (N,) its standard error
    s_gap: np.ndarray         # (N,) where along x0 + s e1 (s in [-k, k]) the gap is smallest
    planes_cross: np.ndarray  # (N,) bool: the gap changes sign inside [-k, k] (Delta rho* = 0)
    rho_p_gap: np.ndarray     # (N,) each side's plane at x0 + s_gap e1
    rho_m_gap: np.ndarray
    dZ_rel: np.ndarray        # (N,) |Delta rho*| / min(rho_+, rho_-) at s_gap = Delta Z / Z_near
    d_rho_x0: np.ndarray      # (N,) the gap at x0 itself (record)
    dtheta: np.ndarray        # (N,) rad, atan2(|n+ x n-|, n+ . n-)
    T: np.ndarray             # (N,) delta^T (Sigma_+ + Sigma_-)^-1 delta in the tangent basis at n_bar
    valid: np.ndarray         # (N,) both sides usable

    def side_fields(self) -> dict[str, np.ndarray]:
        """Flat per-candidate columns: <field>_p / <field>_m for scalar side fields."""
        out = {}
        for tag, s in (("p", self.p), ("m", self.m)):
            for f in fields(s):
                a = getattr(s, f.name)
                if a.ndim == 1:
                    out[f"{f.name}_{tag}"] = a
        return out


RmsMax = Callable[[np.ndarray, int], np.ndarray]  # (k per row, m) -> maximum rms(residual)/alpha per row


def _side(ctx: GeometryContext, x0: np.ndarray, e1: np.ndarray, k: np.ndarray, sign: int, m: int, w_t: int,
          rms_max: RmsMax | None, m_min: int, max_excluded: float, fallback: bool) -> SideResult:
    N = len(x0)
    et = np.stack([-e1[:, 1], e1[:, 0]], -1)
    usable_px = ctx.omega & ctx.normal_valid
    alpha, rms_rel, excl = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    beta, c, cov = np.full((N, 2), np.nan), np.full((N, 2), np.nan), np.full((N, 3, 3), np.nan)
    n_used, m_used = np.zeros(N, dtype=int), np.zeros(N, dtype=int)
    planar = np.zeros(N, dtype=bool)
    pending = np.ones(N, dtype=bool)
    t = np.arange(-w_t, w_t + 1, dtype=np.float64)
    m_cur = m
    while pending.any() and m_cur >= m_min:
        idx = np.nonzero(pending)[0]
        s = sign * (k[idx, None] + np.arange(m_cur + 1, dtype=np.float64))  # (n, m+1)
        pts = (x0[idx, None, None, :] + s[:, :, None, None] * e1[idx, None, None, :]
               + t[None, None, :, None] * et[idx, None, None, :]).reshape(len(idx), -1, 2)
        val, ok = bilinear(ctx.rho_f, usable_px, pts[..., 0], pts[..., 1])
        ex = 1.0 - ok.mean(1)
        excl[idx] = ex
        fit = window_fit(val, pts, ok)
        good = fit.ok & (ex <= max_excluded)
        retry = np.zeros(len(idx), dtype=bool)
        if rms_max is not None:
            fails = good & ~(fit.rms_rel <= rms_max(k[idx], m_cur))
            retry, good = fails, good & ~fails
        g = idx[good]
        alpha[g], beta[g], c[g], cov[g] = fit.alpha[good], fit.beta[good], fit.c[good], fit.cov[good]
        rms_rel[g], n_used[g], m_used[g] = fit.rms_rel[good], fit.n[good], m_cur
        planar[g] = True
        pending[:] = False
        pending[idx[retry]] = True
        m_cur //= 2

    nv = normals(alpha, beta, cov[:, 1:, 1:], c[:, 0], c[:, 1], ctx.K)
    nhat, Sigma_n, sigma_theta = nv.n_hat, nv.Sigma_n, nv.sigma_theta
    fb = np.zeros(N, dtype=bool)
    if fallback:
        need = ~planar
        centre = x0 + sign * (k + m / 2)[:, None] * e1
        pu, pv = np.rint(centre[:, 0]).astype(np.intp), np.rint(centre[:, 1]).astype(np.intp)
        H, W = ctx.shape
        inside = (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H)
        puc, pvc = np.clip(pu, 0, W - 1), np.clip(pv, 0, H - 1)
        fb = need & inside & ctx.normal_valid[pvc, puc]
        alpha[fb] = ctx.dense.alpha[pvc[fb], puc[fb]]
        beta[fb] = ctx.dense.beta[pvc[fb], puc[fb]]
        c[fb] = np.stack([pu[fb], pv[fb]], -1)
        nhat[fb] = ctx.n_hat[pvc[fb], puc[fb]]
        Sigma_n[fb] = ctx.Sigma_n[pvc[fb], puc[fb]]
        sigma_theta[fb] = ctx.sigma_theta[pvc[fb], puc[fb]]
    return SideResult(alpha=alpha, beta=beta, c=c, cov=cov, rms_rel=rms_rel, n=n_used, excl=excl, m_used=m_used,
                      planar=planar, fallback=fb, nhat=nhat, Sigma_n=Sigma_n, sigma_theta=sigma_theta)


def tangent_basis(n: np.ndarray) -> np.ndarray:
    """(N, 3, 2) orthonormal basis of the plane perpendicular to unit vectors n (N, 3)."""
    a = np.zeros_like(n)
    a[np.arange(len(n)), np.argmin(np.abs(n), axis=1)] = 1.0
    b1 = a - np.sum(a * n, -1, keepdims=True) * n
    b1 /= np.linalg.norm(b1, axis=-1, keepdims=True)
    b2 = np.cross(n, b1)
    return np.stack([b1, b2], -1)


def fold_angle(n_p: np.ndarray, n_m: np.ndarray) -> np.ndarray:
    """Delta theta = atan2(|n+ x n-|, n+ . n-) in [0, pi]."""
    return np.arctan2(np.linalg.norm(np.cross(n_p, n_m), axis=-1), np.sum(n_p * n_m, -1))


def null_projection(n_p: np.ndarray, n_m: np.ndarray, S_p: np.ndarray, S_m: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(delta (N, 2), S (N, 2, 2)): delta = B^T (n- - n+) and S = B^T (Sigma_+ + Sigma_-) B, B the tangent basis at
    n_bar = (n+ + n-)/|.|."""
    nbar = n_p + n_m
    nbar = nbar / np.linalg.norm(nbar, axis=-1, keepdims=True)
    B = tangent_basis(nbar)
    delta = np.einsum("nia,ni->na", B, n_m - n_p)
    S = np.einsum("nia,nij,njb->nab", B, S_p + S_m, B)
    return delta, S


def quad_form(delta: np.ndarray, S: np.ndarray, scale: float = 1.0, add: float = 0.0) -> np.ndarray:
    """delta^T (scale S + add I)^-1 delta for 2x2 S; inf if singular with delta != 0, 0 if delta = 0 too."""
    a, b, d = scale * S[:, 0, 0] + add, scale * S[:, 0, 1], scale * S[:, 1, 1] + add
    det = a * d - b * b
    q = d * delta[:, 0] ** 2 - 2 * b * delta[:, 0] * delta[:, 1] + a * delta[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(det > 0, q / det, np.where(np.abs(delta).sum(-1) > 0, np.inf, 0.0))


def chi2_statistic(n_p: np.ndarray, n_m: np.ndarray, S_p: np.ndarray, S_m: np.ndarray) -> np.ndarray:
    """T = delta^T (Sigma_+ + Sigma_-)^-1 delta in the tangent basis at n_bar: chi^2_2 under the null when the
    covariances are right. inf if the projected covariance is singular and delta != 0 (noise-free data)."""
    return quad_form(*null_projection(n_p, n_m, S_p, S_m))


def two_sided_fit(ctx: GeometryContext, x0: np.ndarray, e1: np.ndarray, k: np.ndarray | int, m: int, w_t: int, *,
                  rms_max: RmsMax | None = None, m_min: int = 4, max_excluded: float = 0.2,
                  fallback: bool = True, k_m: np.ndarray | int | None = None) -> TwoSidedResult:
    """Vectorized over N candidates: x0 (N, 2) (u, v), e1 (N, 2) unit, k (N,) or scalar int.

    rms_max None skips the contamination check (calibration passes, the veto's k2 windows).
    k_m: a separate skip for the - side (the + side uses k); the plane-gap span is then [-k_m, k]. None: k on both.
    """
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1, 2)
    e1 = np.asarray(e1, dtype=np.float64).reshape(-1, 2)
    k = np.broadcast_to(np.asarray(k), (len(x0),)).astype(np.float64)
    km = k if k_m is None else np.broadcast_to(np.asarray(k_m), (len(x0),)).astype(np.float64)
    kw = dict(m=m, w_t=w_t, rms_max=rms_max, m_min=m_min, max_excluded=max_excluded, fallback=fallback)
    P = _side(ctx, x0, e1, k, +1, **kw)
    M = _side(ctx, x0, e1, km, -1, **kw)

    def plane_at(sd: SideResult, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.concatenate([np.ones((len(x), 1)), x - sd.c], -1)
        val = sd.alpha + np.sum(sd.beta * (x - sd.c), -1)
        var = np.einsum("ni,nij,nj->n", a, sd.cov, a)
        return np.where(sd.planar, val, np.nan), np.where(sd.planar, var, np.nan)

    # gap(s) = g0 + g1 s along x0 + s e1, affine: smallest |gap| on [-k, k] is 0 if the root lies inside
    rp0, _ = plane_at(P, x0)
    rm0, _ = plane_at(M, x0)
    g0 = rp0 - rm0
    g1 = np.sum((P.beta - M.beta) * e1, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        root = -g0 / g1
    if k_m is None:
        planes_cross = np.isfinite(root) & (np.abs(root) <= k)
        end = np.where(np.abs(g0 + g1 * k) <= np.abs(g0 - g1 * k), k, -k)
    else:
        planes_cross = np.isfinite(root) & (root <= k) & (root >= -km)
        end = np.where(np.abs(g0 + g1 * k) <= np.abs(g0 - g1 * km), k, -km)
    s_gap = np.where(planes_cross, root, end)
    s_gap = np.where(np.isfinite(g0), s_gap, np.nan)
    xg = x0 + np.nan_to_num(s_gap)[:, None] * e1
    rp, vp = plane_at(P, xg)
    rm, vm = plane_at(M, xg)
    d_rho = np.where(planes_cross, 0.0, rp - rm)
    d_rho = np.where(np.isfinite(rp - rm), d_rho, np.nan)
    with np.errstate(invalid="ignore"):
        d_rho_se = np.sqrt(vp + vm)
        dZ_rel = np.abs(d_rho) / np.minimum(rp, rm)
    valid = P.usable & M.usable
    dtheta = np.where(valid, fold_angle(P.nhat, M.nhat), np.nan)
    T = np.full(len(x0), np.nan)
    if valid.any():
        T[valid] = chi2_statistic(P.nhat[valid], M.nhat[valid], P.Sigma_n[valid], M.Sigma_n[valid])
    return TwoSidedResult(p=P, m=M, d_rho=d_rho, d_rho_se=d_rho_se, s_gap=s_gap, planes_cross=planes_cross,
                          rho_p_gap=rp, rho_m_gap=rm, dZ_rel=dZ_rel, d_rho_x0=g0, dtheta=dtheta, T=T, valid=valid)


def window_rms(res: TwoSidedResult) -> np.ndarray:
    """rms(residual)/alpha of every planar window (both sides) of a pass."""
    return np.concatenate([res.p.rms_rel[res.p.planar], res.m.rms_rel[res.m.planar]])


def planarity_scale(res: TwoSidedResult) -> float:
    """eps_hat: median rms(residual)/alpha over all planar windows (both sides) of a calibration pass."""
    r = window_rms(res)
    return float(np.median(r)) if r.size else float("nan")


# ---------------------------------------------------------------- trilateral contamination bound b_tri(k, m)

B_TRI_PATH = Path(__file__).with_name("data") / "b_tri.json"


@lru_cache(maxsize=None)
def _b_tri_table() -> dict:
    return json.loads(B_TRI_PATH.read_text())


def b_tri(k: np.ndarray, m: int, tri: tuple[float, float, float] = (3.0, 0.01, 3.0)) -> np.ndarray:
    """Upper envelope of rms(residual)/alpha that the trilateral filter's bias alone puts into a window at skip k
    with length m next to a feature (noise-free, measured by scripts/calibrate_fold.py on single folds and steps).
    tri = (sigma_s, sigma_r, sigma_g) of rho_f; the table holds only the default filter."""
    t = _b_tri_table()
    if tuple(t["tri"]) != tuple(tri):
        raise ValueError(f"b_tri is calibrated for trilateral {t['tri']}, not {tri}; rerun scripts/calibrate_fold.py")
    if str(m) not in t["b"]:
        raise ValueError(f"b_tri has no table for m = {m} (have {sorted(t['b'])})")
    row = np.asarray(t["b"][str(m)], dtype=np.float64)  # index k = 0 .. len-1; constant beyond
    kk = np.clip(np.asarray(k, dtype=np.int64), 0, len(row) - 1)
    return row[kk]
