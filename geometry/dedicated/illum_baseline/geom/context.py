"""GeometryContext: per-image fields shared by all geometric-edge channels, computed lazily once.

Channels only read it: every cached array is returned non-writeable.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from ..detector import (MAD_TO_STD, Disparity, Params, dilate, gaussian_gradient, prepare_disparity,
                        smooth_and_laplacian, trilateral)
from .camera import Intrinsics
from .normals import Normals, normals
from .planefit import DenseFit, DenseFitParams, dense_fit


def _ro(*arrays: np.ndarray) -> None:
    for a in arrays:
        a.flags.writeable = False


class GeometryContext:
    """Canonical depth D (H, W) and sky probability at network resolution, upright frame.

    det: detector Params; supplies the sky mask (B1), disparity (B2), rho_floor and the trilateral parameters
    (tri_sigma_s/r/g), which are used whatever det.prefilter says: normals always come from rho_f.
    fit: N1 parameters.
    """

    def __init__(self, D: np.ndarray, sky_prob: np.ndarray | None, K: Intrinsics, det: Params,
                 fit: DenseFitParams = DenseFitParams()):
        D = np.asarray(D)
        if D.shape != (K.H, K.W):
            raise ValueError(f"D has shape {D.shape}, intrinsics are for (H, W) = {(K.H, K.W)}")
        self.D, self.sky_prob, self.K, self.det, self.fit = D, sky_prob, K, det, fit
        self.shape = D.shape
        self.u, self.v = K.pixel_grid()
        _ro(self.u, self.v)

    # ------------------------------------------------------------ disparity
    @cached_property
    def disparity(self) -> Disparity:
        dp = prepare_disparity(self.D, self.sky_prob, self.det)
        _ro(dp.S, dp.ground, dp.bad, dp.rho)
        return dp

    @property
    def S(self) -> np.ndarray:
        return self.disparity.S

    @property
    def omega(self) -> np.ndarray:
        return self.disparity.ground

    @property
    def rho(self) -> np.ndarray:
        return self.disparity.rho

    @property
    def rho_floor(self) -> float:
        return self.disparity.rho_floor

    @cached_property
    def rho_f(self) -> np.ndarray:
        d = self.det
        out = trilateral(self.rho, self.omega, self.rho_floor, d.tri_sigma_s, d.tri_sigma_r, d.tri_sigma_g)
        _ro(out)
        return out

    @cached_property
    def g(self) -> np.ndarray:
        """(H, W, 2) Gaussian-derivative gradient of rho_f at fit.sigma_g (disparity per px)."""
        out = np.stack(gaussian_gradient(self.rho_f, self.fit.sigma_g), -1)
        _ro(out)
        return out

    # ------------------------------------------------------------ N1, N2
    @cached_property
    def dense(self) -> DenseFit:
        out = dense_fit(self.rho_f, self.omega, self.rho_floor, self.fit)
        _ro(out.alpha, out.beta, out.cov, out.m0, out.n_eff, out.s2, out.cond, out.valid)
        return out

    @cached_property
    def _normals(self) -> tuple[Normals, np.ndarray]:
        fit = self.dense
        H, W = self.shape
        sel = np.nonzero(fit.valid)
        nv = normals(fit.alpha[sel], fit.beta[sel], fit.cov[sel][:, 1:, 1:], self.u[sel], self.v[sel], self.K)
        ok = np.isfinite(nv.n_hat).all(-1) & np.isfinite(nv.sigma_theta)
        full = Normals(n_hat=np.full((H, W, 3), np.nan), Sigma_n=np.full((H, W, 3, 3), np.nan),
                       sigma_theta=np.full((H, W), np.nan), mu=np.full((H, W, 3), np.nan),
                       mu_norm=np.full((H, W), np.nan))
        keep = tuple(i[ok] for i in sel)
        for name in ("n_hat", "Sigma_n", "sigma_theta", "mu", "mu_norm"):
            getattr(full, name)[keep] = getattr(nv, name)[ok]
        valid = np.zeros((H, W), dtype=bool)
        valid[keep] = True
        _ro(full.n_hat, full.Sigma_n, full.sigma_theta, full.mu, full.mu_norm, valid)
        return full, valid

    @property
    def normals(self) -> Normals:
        return self._normals[0]

    @property
    def n_hat(self) -> np.ndarray:
        return self._normals[0].n_hat

    @property
    def Sigma_n(self) -> np.ndarray:
        return self._normals[0].Sigma_n

    @property
    def sigma_theta(self) -> np.ndarray:
        return self._normals[0].sigma_theta

    @property
    def normal_valid(self) -> np.ndarray:
        return self._normals[1]

    # ------------------------------------------------------------ record-only fields
    @cached_property
    def L_over_rho_f(self) -> np.ndarray:
        """Un-vetoed L_sigma(rho_f) / max(rho_sigma, rho_floor) (detector B3/B4 on rho_f): only its sign is used,
        to record convex (negative: disparity ridge) vs concave folds."""
        rs, L = smooth_and_laplacian(self.rho_f, self.det.sigma, self.det.truncate)
        out = L / np.maximum(rs, self.rho_floor)
        _ro(out)
        return out

    @cached_property
    def L_veto_over_rho_f(self) -> np.ndarray:
        """Record-only (TASK_GEOM_V2 V4): the scale-ratio-vetoed response on rho_f, signed as L: sign(L) min(|L|,
        g |L_sigma - L_{band_factor sigma}|) / max(rho_sigma, rho_floor), g = 1 / (1 - 1 / band_factor) (the tri_veto4
        veto; smooth curvature cancels in the band term). Used by the fold channel's rescue_undecidable: corroborate."""
        d = self.det
        rs, L = smooth_and_laplacian(self.rho_f, d.sigma, d.truncate)
        Lb = L - smooth_and_laplacian(self.rho_f, d.band_factor * d.sigma, d.truncate)[1]
        den = np.maximum(rs, self.rho_floor)
        g = 1.0 / (1.0 - 1.0 / d.band_factor)
        out = np.sign(L) * np.minimum(np.abs(L / den), g * np.abs(Lb / den))
        _ro(out)
        return out

    @cached_property
    def s_hat_L(self) -> float:
        """Record-only: the detector's B5 noise scale on the un-vetoed |L_over_rho_f| (median / 0.6745 over the pixels
        r px clear of the sky and R px clear of the border, clamped to [s_hat_min, s_hat_max]); used by the fold
        channel's crease rescue (TASK_GEOM_INTEGRATION G3). Equals the Laplacian channel's s_hat when that one is
        computed on the un-vetoed filtered response (tri_shat filtered)."""
        p = self.det
        valid = ~dilate(self.S, p.r)
        R = p.R
        valid[:R, :] = valid[-R:, :] = False
        valid[:, :R] = valid[:, -R:] = False
        s = float(np.median(np.abs(self.L_over_rho_f[valid])) / MAD_TO_STD) if valid.any() else p.s_hat_min
        if p.s_hat_min is not None:
            s = max(s, p.s_hat_min)
        if p.s_hat_max is not None:
            s = min(s, p.s_hat_max)
        return s
