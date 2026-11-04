"""TASK_DEPTH_CHANNEL D1: kernels (D-K), detrended orientation (D-1), normalized second directional derivative (D-2).

1. g1 moments; the Hessian exact on quadratics, and identically 0 on affine inputs, borders included.
2. Planes of random slant give q = 0 (noise-free).
3. Detrending theorem: a step against the background slope with |Delta rho| < sqrt(2 pi) sigma_tot |g_p| fails the
   polarity rule with n from grad rho_sigma and passes it with n from grad Dd.
"""

import dataclasses
import math

import numpy as np
import pytest
from scipy.special import ndtr

from illum_baseline.detector import Params
from illum_baseline.geom.camera import Intrinsics
from illum_baseline.geom.candidates import _sep, depth_field, depth_kernels
from illum_baseline.geom.context import GeometryContext
from illum_baseline.geom.depth import DepthParams
from illum_baseline.geom.sampling import bilinear
from test_detector import random_plane_D

TRI = Params(prefilter="trilateral")
DP = DepthParams()
W, H = 504, 378
K = Intrinsics.centered(400.0, 400.0, W, H)


@pytest.mark.parametrize("sigma", [1.25, 2.0, DP.sigma_L_factor * math.hypot(DP.sigma, DP.sigma_w_prior)])
def test_1_g1_moments(sigma):
    x, g0, g1, g2 = depth_kernels(sigma)
    assert abs(g1.sum()) < 1e-15
    assert (x * g1).sum() == pytest.approx(1.0, abs=1e-14)
    for k in (0, 2, 4):
        assert abs((x**k * g1).sum()) < 1e-12
    assert np.allclose(g1, -g1[::-1], atol=0, rtol=0)
    u = np.tile(np.arange(60.0), (40, 1))  # f(u) = u: the implemented filter returns exactly 1, borders included
    assert np.abs(_sep(u, g1, g0) - 1.0).max() < 1e-12
    assert np.abs(_sep(u.T.copy(), g0, g1) - 1.0).max() < 1e-12


def test_1_hessian_exact_on_quadratics_zero_on_affine():
    _, g0, g1, g2 = depth_kernels(DP.sigma)
    R = g0.size // 2
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rng = np.random.default_rng(2)
    for _ in range(4):
        a, b, c, d, e, f = rng.normal(size=6) * [1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1.0]
        quad = a * u**2 + b * u * v + c * v**2 + d * u + e * v + f
        Huu, Hvv, Huv = _sep(quad, g2, g0), _sep(quad, g0, g2), _sep(quad, g1, g1)
        inner = (slice(R, -R), slice(R, -R))  # odd-reflect padding is exact for affine, not quadratic, functions
        assert np.abs(Huu[inner] - 2 * a).max() < 1e-12
        assert np.abs(Hvv[inner] - 2 * c).max() < 1e-12
        assert np.abs(Huv[inner] - b).max() < 1e-12
        aff = d * u + e * v + f
        for kx, ky in ((g2, g0), (g0, g2), (g1, g1)):
            assert np.abs(_sep(aff, kx, ky)).max() < 1e-12  # whole image, borders included


@pytest.mark.parametrize("WH", [(504, 378), (378, 504)])
def test_2_planes_give_q_zero(WH):
    w, h = WH
    k = Intrinsics.centered(400.0, 400.0, w, h)
    rng = np.random.default_rng(5)
    for _ in range(5):
        D = random_plane_D(rng, w, h)
        ctx = GeometryContext(D, None, k, TRI)
        f = depth_field(ctx, DP)
        assert np.abs(f.q[ctx.omega]).max() <= 1e-9
        assert "_normals" not in ctx.__dict__ and "dense" not in ctx.__dict__  # the depth field never builds normals


def _step_scene(drho_factor, g_p=0.002, rho0=0.5, u_s=250.5):
    """rho(u) = rho0 + g_p (u - c_x) + drho Phi((u - u_s) / sigma_w): a step smeared by sigma_w = sigma_w_prior,
    descending (drho < 0) against a rising background slope, |drho| = drho_factor sqrt(2 pi) sigma_tot g_p."""
    sigma_tot = math.hypot(DP.sigma, DP.sigma_w_prior)
    drho = -drho_factor * math.sqrt(2 * math.pi) * sigma_tot * g_p
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rho = rho0 + g_p * (u - K.c_x) + drho * ndtr((u - u_s) / DP.sigma_w_prior)
    return 1.0 / rho, u_s


def _polarity(f, u_s, rows):
    """Per row: the zero crossing of q nearest the step and the polarity rule q(x0 - n/2) > 0 > q(x0 + n/2)."""
    ok = []
    for r in rows:
        q = f.q[r]
        c = np.nonzero(np.sign(q[:-1]) != np.sign(q[1:]))[0]
        i = c[np.argmin(np.abs(c + 0.5 - u_s))]
        t = q[i] / (q[i] - q[i + 1])
        x0 = np.array([i + t, float(r)])
        n, _ = bilinear(f.n, np.ones(f.q.shape, bool), np.array([x0[0]]), np.array([x0[1]]))
        n = n[0] / np.linalg.norm(n[0])
        qa, _ = bilinear(f.q, np.ones(f.q.shape, bool), np.array([x0[0] - n[0] / 2, x0[0] + n[0] / 2]),
                         np.array([x0[1] - n[1] / 2, x0[1] + n[1] / 2]))
        ok.append(qa[0] > 0 > qa[1])
    return np.array(ok)


def test_3_detrending_theorem(capsys):
    rows = range(60, H - 60, 10)
    D, u_s = _step_scene(0.8)  # below the bound: no gradient maximum in grad rho_sigma
    p = dataclasses.replace(DP, disparity="rho")  # the theorem on the disparity itself
    ctx = GeometryContext(D, None, K, TRI)
    plain = _polarity(depth_field(ctx, p, detrend=False), u_s, rows)
    detr = _polarity(depth_field(ctx, p, detrend=True), u_s, rows)
    detr_f = _polarity(depth_field(ctx, DP, detrend=True), u_s, rows)  # through the trilateral filter (report)
    D2, _ = _step_scene(1.5)  # above the bound: both orientations pass (control)
    ctx2 = GeometryContext(D2, None, K, TRI)
    plain2 = _polarity(depth_field(ctx2, p, detrend=False), u_s, rows)
    with capsys.disabled():
        print(f"\n[D1 test 3] |drho| = 0.8 x bound: polarity with n from grad rho_sigma {plain.mean():.2f}, from grad Dd "
              f"{detr.mean():.2f} (rho), {detr_f.mean():.2f} (rho_f); control 1.5 x bound, grad rho_sigma {plain2.mean():.2f}")
    assert not plain.any()
    assert detr.all()
    assert plain2.all()
