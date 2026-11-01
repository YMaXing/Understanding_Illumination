"""TASK_DEPTH_CHANNEL D2: candidates (D-3), the gradient floor, and the shared two-sided fit's gap_mode / usable.

4. Creases at 30-120 deg (fold scenes, away from steps): 0 accepted depth candidates.
5. A plane plus a 7-px grid ripple of 0.2 %, through the trilateral filter: 0 accepted.
Plus: the gradient floor costs no recall on synthetic steps with r >= 5 %; the centring stage removes the ghost lines
next to large steps on rho_f; gap_mode "at_x0" measures the jump at x0 exactly and the depth fit never builds normals.
("Accepted" here is D-3: the gates of D-5 follow in D3.)
"""

import dataclasses
import itertools
import math

import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial import cKDTree

from illum_baseline.detector import Params
from illum_baseline.geom.camera import Intrinsics
from illum_baseline.geom.candidates import depth_candidates
from illum_baseline.geom.context import GeometryContext
from illum_baseline.geom.depth import DepthParams, depth_null, fit_kwargs, measure_jump, relative_jump
from illum_baseline.geom.synthetic import F_PX, fold_scene, rays, step_scene
from illum_baseline.geom.twosided import two_sided_fit

TRI = Params(prefilter="trilateral")
DP = DepthParams()
W, H = 504, 378
K = Intrinsics.centered(F_PX, F_PX, W, H)
C = np.array([(W - 1) / 2, (H - 1) / 2])


def noisy(D, rel, seed):
    return 1.0 / ((1.0 / D) * (1.0 + rel * np.random.default_rng(seed).normal(size=D.shape)))


def line_normal(phi):
    """Unit normal of the scenes' image line through the principal point at angle phi."""
    return np.array([-math.sin(math.radians(phi)), math.cos(math.radians(phi))])


def test_4_creases_give_no_candidates(capsys):
    tot_noisy, band_noisy = 0, 0
    for dth, phi, convex in itertools.product((30, 60, 90, 120), (0, 45, 90, 135), (True, False)):
        D, _, _, sky = fold_scene(dth, phi, convex)
        nq = line_normal(phi)
        for noise in (0.0, 5e-4):
            Dn = np.where(sky, 1.0, noisy(np.where(sky, 1.0, D), noise, 3)) if noise else np.where(sky, 1.0, D)
            ctx = GeometryContext(Dn, sky.astype(float), K, TRI)
            c = depth_candidates(ctx, DP)
            ij = np.clip(c.ij, 0, [W - 1, H - 1])
            away = ~ndimage.binary_dilation(ctx.S, iterations=20)[ij[:, 1], ij[:, 0]]  # away from the sky steps
            near_crease = np.abs((c.x0 - C) @ nq) <= 6.0
            if noise == 0:
                assert not (c.passed & away).any(), (dth, phi, convex)
            else:  # noise doublets occur anywhere at the noise rate, but none at the crease
                assert not (c.passed & away & near_crease).any(), (dth, phi, convex)
                tot_noisy += int((c.passed & away).sum())
                band_noisy += int((c.passed & away & near_crease).sum())
    with capsys.disabled():
        print(f"\n[D2 test 4] creases 30-120 deg: noise-free 0 D-3 candidates; noise 5e-4: {tot_noisy} noise doublets "
              f"over 32 scenes, {band_noisy} within 6 px of a crease")


@pytest.mark.parametrize("case", ["rows", "oblique"])
@pytest.mark.parametrize("noise", [0.0, 5e-4])
def test_5_ripple_through_trilateral_gives_no_candidates(case, noise):
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    n = np.array([0.25, -0.35, 1.0])
    n /= np.linalg.norm(n)
    rho = rays(W, H) @ n / 3.0
    phase = 2 * np.pi * (v / 7.0 if case == "rows" else (u + v) / (7.0 * math.sqrt(2)))
    D = 1.0 / (rho * (1 + 2e-3 * np.sin(phase)))
    if noise:
        D = noisy(D, noise, 5)
    c = depth_candidates(GeometryContext(D, None, K, TRI), DP)
    assert c.counts["n_passed"] == 0, c.counts


def test_gradient_floor_costs_no_recall(capsys):
    """Edge coverage (share of the true edge, sampled every 0.5 px, within 1.5 px of a D-3 candidate) on steps with
    r = 5 % ... 200 %, both polarities, noise 5e-4: identical with the floor (k_g = 3) and without (k_g = 0)."""
    rec, off, valid = [], [], []
    for ratio, tilts, phi in itertools.product((1.05, 1.5, 3.0), ((20, -20), (0, 40)), (0, 210)):
        D, _, _ = step_scene(ratio, *tilts, phi)
        ctx = GeometryContext(noisy(D, 5e-4, 7), None, K, TRI)
        t = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
        P = C + np.arange(-400, 400, 0.5)[:, None] * t
        P = P[(P[:, 0] > 15) & (P[:, 0] < W - 16) & (P[:, 1] > 15) & (P[:, 1] < H - 16)]
        r = []
        for k_g in (DP.k_g, 0.0):
            c = depth_candidates(ctx, dataclasses.replace(DP, k_g=k_g))
            d, _ = cKDTree(c.x0[c.passed]).query(P)
            r.append((d <= 1.5).mean())
            off.append(int((c.passed & (np.abs((c.x0 - C) @ line_normal(phi)) > 3)).sum()))
        valid.append(c.field.valid.mean())
        rec.append(r)
    rec = np.array(rec)
    with capsys.disabled():
        print(f"\n[D2 floor] recall with floor {rec[:, 0].mean():.4f} (min {rec[:, 0].min():.4f}), without "
              f"{rec[:, 1].mean():.4f}; off-edge D-3 passes per scene {np.mean(off[0::2]):.1f} vs {np.mean(off[1::2]):.1f}")
    assert np.array_equal(rec[:, 0], rec[:, 1])
    assert rec[:, 0].min() >= 0.99


@pytest.mark.parametrize("ratio", [1.1, 1.5, 3.0])
def test_centring_removes_ghost_lines(ratio):
    """On rho_f the trilateral filter leaves small wiggles 5-7 px from a step; there the detrended n has flipped and a
    crossing borrows the step's own lobe (|s_p + s_m| / s_sep ~ 0.5). Noise-free: with the prior smear bound
    (s_hi = 6.5 px) they pass D-3 without the centring stage and none pass with it; with the calibrated bound
    neither the edge nor anything else is lost."""
    D, _, _ = step_scene(ratio, 20, -20, 30)
    ctx = GeometryContext(D, None, K, TRI)
    nq = line_normal(30)
    out = {}
    for name, p in (("prior_off", dataclasses.replace(DP, sigma_w_max=3.0, centre_tol=math.inf)),
                    ("prior_on", dataclasses.replace(DP, sigma_w_max=3.0)), ("default", DP)):
        c = depth_candidates(ctx, p)
        d = np.abs((c.x0 - C) @ nq)
        out[name] = (int((c.passed & (d < 1.5)).sum()), int((c.passed & (d >= 1.5) & (d <= 10)).sum()))
    assert out["prior_on"][1] == 0 and out["default"][1] == 0
    assert out["prior_on"][0] == out["prior_off"][0]  # centring removes no edge candidate
    if ratio < 3.0:
        assert out["prior_off"][1] > 100  # the ghosts exist without it


def test_gap_at_x0_exact_and_lazy():
    """Noise-free step on slanted planes: at the true edge the at_x0 jump equals the planes' own disparities on the
    ray through x0 (to the trilateral bias at k = 11), r = Z_far/Z_near - 1, s_gap = 0, d_rho = d_rho_x0; the
    min-span gap is recorded too; the default call (min_span, normals mask) is unchanged by the new fields; and the
    depth-style fit (usable = Omega, no fallback) builds no normals."""
    ratio, tilts, phi = 1.5, (20, -20), 30
    D, _, ns = step_scene(ratio, *tilts, phi)
    ctx = GeometryContext(D, None, K, TRI)
    nq = line_normal(phi)
    t = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
    x0 = C + np.arange(-120, 121, 20.0)[:, None] * t
    n = np.broadcast_to(-nq, x0.shape).copy()  # toward the near side (side < 0)
    res = two_sided_fit(ctx, x0, n, 11, rms_max=None, **fit_kwargs(DP, ctx))
    assert "_normals" not in ctx.__dict__ and "dense" not in ctx.__dict__
    assert res.p.planar.all() and res.m.planar.all()
    np.testing.assert_array_equal(res.s_gap, 0.0)
    np.testing.assert_array_equal(res.d_rho, res.d_rho_x0)
    np.testing.assert_array_equal(res.d_rho_se, res.d_rho_x0_se)
    # truth: each plane's disparity on the ray through x0 (synthetic.step_scene geometry)
    r = np.stack([(x0[:, 0] - C[0]) / F_PX, (x0[:, 1] - C[1]) / F_PX, np.ones(len(x0))], -1)
    Z1 = 3.0
    rho_near = (r @ ns[1]) / (ns[1] @ np.array([0, 0, Z1]))
    rho_far = (r @ ns[2]) / (ns[2] @ np.array([0, 0, ratio * Z1]))
    np.testing.assert_allclose(res.rho_p_gap, rho_near, rtol=2e-4)
    np.testing.assert_allclose(res.rho_m_gap, rho_far, rtol=2e-4)
    np.testing.assert_allclose(relative_jump(res), rho_near / rho_far - 1, rtol=2e-3)
    mins = two_sided_fit(ctx, x0, n, 11, rms_max=None, **{**fit_kwargs(DP, ctx), "gap_mode": "min_span"})
    np.testing.assert_array_equal(res.d_rho_min_span, mins.d_rho)
    np.testing.assert_array_equal(mins.d_rho_min_span, mins.d_rho)
    with pytest.raises(ValueError):
        two_sided_fit(ctx, x0, n, 11, 8, 4, gap_mode="nearest")
    # the fold call (defaults) is unaffected: same values as before the gap_mode / usable options
    ctx2 = GeometryContext(D, None, K, TRI)
    a = two_sided_fit(ctx2, x0, n, 11, 8, 4)
    b = two_sided_fit(ctx2, x0, n, 11, 8, 4, gap_mode="min_span", usable=None)
    for f in ("d_rho", "d_rho_se", "s_gap", "rho_p_gap", "rho_m_gap", "dZ_rel", "d_rho_x0", "dtheta", "T"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))


def test_jump_and_null_on_steps():
    """Candidates on a noisy step (r = 10 %): the jump at k_c is positive (the near side is +n) and
    lambda_hat_Delta_rho on the step scene is in the range of the synthetic-noise calibration (~100)."""
    D, _, _ = step_scene(1.1, 20, -20, 30)
    ctx = GeometryContext(noisy(D, 5e-4, 2), None, K, TRI)
    c = depth_candidates(ctx, DP)
    nul = depth_null(ctx, DP, c.field)
    on = c.passed & (np.abs((c.x0 - C) @ line_normal(30)) < 1.5)
    res = measure_jump(ctx, DP, c.x0[on], c.n[on], c.k_c[on], nul)
    ok = res.p.planar & res.m.planar
    assert ok.mean() > 0.9
    r = relative_jump(res)[ok]
    assert (r > 0).all()
    assert abs(np.median(r) - 0.1) < 0.01
    assert 50 < nul.lambda_hat < 200
    assert "_normals" not in ctx.__dict__
