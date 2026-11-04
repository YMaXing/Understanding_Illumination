"""N2: surface normals from a local disparity plane (alpha, beta) at pixel (u, v).

A plane n.X = d0 images to rho = m.r with m = n/d0 (r = K^-1 (u, v, 1)). Matching rho(x + d) ~ alpha + beta.d at x
gives m = alpha mu with the dimensionless plane vector

    mu = (f_x g_u, f_y g_v, 1 - (u - c_x) g_u - (v - c_y) g_v),   g = beta / alpha (relative gradient, per px),

and the unit normal pointing toward the camera n_hat = -mu / |mu| (n_hat . r < 0, since mu . r = 1). mu is the same
at every pixel of one plane and does not change under a global depth scale (g is relative).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Intrinsics


@dataclass(frozen=True)
class Normals:
    n_hat: np.ndarray        # (..., 3) unit, toward the camera
    Sigma_n: np.ndarray      # (..., 3, 3) covariance of n_hat (tangent to the unit sphere)
    sigma_theta: np.ndarray  # (...) rad, sqrt(trace Sigma_n)
    mu: np.ndarray           # (..., 3) plane vector
    mu_norm: np.ndarray      # (...) |mu| = 1 / (|r| cos alpha_inc)


def plane_vector(alpha: np.ndarray, beta: np.ndarray, u: np.ndarray, v: np.ndarray, K: Intrinsics) -> np.ndarray:
    """mu (..., 3) from alpha (...), beta (..., 2) = (d rho/du, d rho/dv) and pixel coordinates u, v (...)."""
    gu, gv = beta[..., 0] / alpha, beta[..., 1] / alpha
    return np.stack([K.f_x * gu, K.f_y * gv, 1.0 - (u - K.c_x) * gu - (v - K.c_y) * gv], axis=-1)


def plane_distance(alpha: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """d0 = 1 / (alpha |mu|), in the depth units that rho = 1/D implies."""
    return 1.0 / (alpha * np.linalg.norm(mu, axis=-1))


def normals(alpha: np.ndarray, beta: np.ndarray, cov_beta: np.ndarray, u: np.ndarray, v: np.ndarray,
            K: Intrinsics) -> Normals:
    """n_hat, its covariance and sigma_theta.

    delta mu = J delta g with J = [[f_x, 0], [0, f_y], [-(u - c_x), -(v - c_y)]]; Sigma_g = Cov(beta) / alpha^2
    (the alpha uncertainty is neglected); Sigma_n = P J Sigma_g J^T P / |mu|^2 with P = I - n n^T.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    u = np.broadcast_to(np.asarray(u, dtype=np.float64), alpha.shape)
    v = np.broadcast_to(np.asarray(v, dtype=np.float64), alpha.shape)
    mu = plane_vector(alpha, beta, u, v, K)
    mu_norm = np.linalg.norm(mu, axis=-1)
    n_hat = -mu / mu_norm[..., None]

    J = np.zeros(alpha.shape + (3, 2))
    J[..., 0, 0] = K.f_x
    J[..., 1, 1] = K.f_y
    J[..., 2, 0] = -(u - K.c_x)
    J[..., 2, 1] = -(v - K.c_y)
    Sigma_g = cov_beta / (alpha**2)[..., None, None]
    Sigma_mu = np.einsum("...ia,...ab,...jb->...ij", J, Sigma_g, J)
    P = np.eye(3) - n_hat[..., :, None] * n_hat[..., None, :]
    Sigma_n = np.einsum("...ia,...ab,...jb->...ij", P, Sigma_mu, P) / (mu_norm**2)[..., None, None]
    sigma_theta = np.sqrt(np.maximum(np.trace(Sigma_n, axis1=-2, axis2=-1), 0.0))
    return Normals(n_hat=n_hat, Sigma_n=Sigma_n, sigma_theta=sigma_theta, mu=mu, mu_norm=mu_norm)


def focal_bracket(K: Intrinsics) -> tuple[Intrinsics, Intrinsics]:
    """(K with f (1 - sigma_f/f), K with f (1 + sigma_f/f)): the focal diagnostics of F2.7, never used in a gate."""
    rel = K.sigma_f_x / K.f_x
    return K.scaled(1.0 - rel), K.scaled(1.0 + rel)
