"""Spec tests 4-11: kernels, planes, features, noise, scale invariance, threshold-before-pool,
upsample alignment, pair weights. Plus sky-mask behavior (B1)."""

import numpy as np
import pytest
from scipy import ndimage
from scipy.signal import convolve2d

from illum_baseline.detector import (MAD_TO_STD, Params, compute_weights, disk, kernels,
                                     sky_mask, smooth_and_laplacian)
from illum_baseline.outputs import deliverable_w, pair_weights, upsample_weights

F_PX = 400.0
CANON = 300.0
P = Params()


def rays(W, H, f=F_PX):
    """Pinhole rays (x/z, y/z, 1) with principal point ((W-1)/2, (H-1)/2); y points down."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    return np.stack([(u - (W - 1) / 2) / f, (v - (H - 1) / 2) / f, np.ones_like(u)], axis=-1)


def random_plane_D(rng, W, H):
    r = rays(W, H)
    while True:
        n = rng.normal(size=3)
        n[2] = abs(n[2]) + 0.5
        n /= np.linalg.norm(n)
        nr = r @ n
        if nr.min() > 0.05:
            d = rng.uniform(1.0, 5.0)
            return (d / nr) * CANON / F_PX  # canonical depth D = Z * 300 / f


def box_scene(W=504, H=378):
    """Floor (camera 1.5 m high), back wall at z = 8, and an axis-aligned box at z in [2, 4]."""
    r = rays(W, H)
    dx, dy = r[..., 0], r[..., 1]
    t = np.full((H, W), 8.0)  # wall (d_z = 1, so t = Z)
    sid = np.full((H, W), 2, dtype=np.int32)
    with np.errstate(divide="ignore"):
        t_floor = np.where(dy > 0, 1.5 / dy, np.inf)
    hit = t_floor < t
    t[hit], sid[hit] = t_floor[hit], 1
    lo, hi = np.array([0.3, 0.6, 2.0]), np.array([1.3, 1.5, 4.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = lo / r, hi / r
    tnear, tfar = np.minimum(t1, t2), np.maximum(t1, t2)
    tnear = np.where(np.isnan(tnear), -np.inf, tnear)
    tfar = np.where(np.isnan(tfar), np.inf, tfar)
    t_in, t_out = tnear.max(-1), tfar.min(-1)
    face = tnear.argmax(-1)  # 0: x face, 1: y (top) face, 2: z (front) face
    hit = (t_out >= t_in) & (t_in > 0) & (t_in < t)
    t[hit], sid[hit] = t_in[hit], 3 + face[hit]
    return t * CANON / F_PX, sid


def features_and_interior(sid, R):
    b = np.zeros(sid.shape, dtype=bool)
    b[:, 1:] |= sid[:, 1:] != sid[:, :-1]
    b[:, :-1] |= sid[:, 1:] != sid[:, :-1]
    b[1:, :] |= sid[1:, :] != sid[:-1, :]
    b[:-1, :] |= sid[1:, :] != sid[:-1, :]
    # Chessboard distance: the separable kernel's support is a (2R+1)^2 square.
    dist = ndimage.distance_transform_cdt(~b, metric="chessboard")
    return b, dist > R + 1


def noisy(D, rel, seed):
    """i.i.d. Gaussian noise on rho with std rel * rho."""
    rho = 1.0 / D
    rho = rho * (1 + rel * np.random.default_rng(seed).normal(size=D.shape))
    return 1.0 / rho


# 4. Kernels
@pytest.mark.parametrize("sigma", [0.8, 1.25, 2.0])
def test_kernel_moments(sigma):
    x, g0, g2 = kernels(sigma, 4.0)
    assert g0.sum() == pytest.approx(1.0, abs=1e-12)
    assert abs(g2.sum()) < 1e-12
    assert (x**2 * g2).sum() == pytest.approx(2.0, abs=1e-12)
    for k in (1, 3, 5):
        assert abs((x**k * g0).sum()) < 1e-12
        assert abs((x**k * g2).sum()) < 1e-12


# 5. Planes exact
@pytest.mark.parametrize("WH", [(504, 378), (378, 504)])
def test_planes_exact(WH):
    rng = np.random.default_rng(5)
    for _ in range(6):
        D = random_plane_D(rng, *WH)
        out = compute_weights(D, None, P)
        assert out["q"].max() < 1e-9
        assert np.all(out["w"] == 1.0)


# 6. Features respond
def test_features_respond():
    D, sid = box_scene()
    assert len(np.unique(sid)) == 5  # floor, wall, box x/top/front faces all visible
    out = compute_weights(D, None, P)
    feat, interior = features_and_interior(sid, P.R)
    q = out["q"]
    assert interior.sum() > 0.5 * q.size
    assert q[interior].max() < 1e-9
    assert np.median(q[feat]) > 1e-4
    assert (q[feat] > 1e-6).mean() >= 0.95


# 7. Noise estimator
def test_noise_estimator(capsys):
    D = noisy(random_plane_D(np.random.default_rng(7), 504, 378), 1e-3, seed=0)
    _, g0, g2 = kernels(P.sigma, P.truncate)
    k_norm = np.linalg.norm(np.outer(g0, g2) + np.outer(g2, g0))
    expected = 1e-3 * k_norm
    out = compute_weights(D, None, P)
    s_hat = out["scalars"]["s_hat"]
    assert not out["scalars"]["s_hat_clamped"]
    assert s_hat == pytest.approx(expected, rel=0.10)
    lines = [f"s_hat={s_hat:.4e}, expected 1e-3*||k_LoG||={expected:.4e} (ratio {s_hat / expected:.3f})"]
    for k in (3.0, 4.0, 5.0):
        w = compute_weights(D, None, Params(k_tau=k))["w"]
        lines.append(f"k_tau={k:.0f}: frac(w < 1) = {(w < 1).mean():.4f}")
    with capsys.disabled():
        print("\n[test 7] " + "; ".join(lines))


# 8. Scale invariance
@pytest.mark.parametrize("c", [0.37, 2.9])
def test_scale_invariance(c):
    D, _ = box_scene()
    D = noisy(D, 1e-3, seed=1)
    a = compute_weights(D, None, P)
    b = compute_weights(c * D, None, P)
    assert not a["scalars"]["s_hat_clamped"] and not b["scalars"]["s_hat_clamped"]
    np.testing.assert_allclose(b["w"], a["w"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(b["q"], a["q"], rtol=1e-9, atol=1e-15)


# 9. Threshold before pool
def reference_weights(D, p):
    """Direct implementation: 2-D convolution with outer-product kernels, loop-based disk max."""
    _, g0, g2 = kernels(p.sigma, p.truncate)
    R = g0.size // 2
    rho = 1.0 / D
    pad = np.pad(rho, R, mode="reflect", reflect_type="odd")
    rho_s = convolve2d(pad, np.outer(g0, g0), mode="valid")
    L = convolve2d(pad, np.outer(g0, g2) + np.outer(g2, g0), mode="valid")
    floor = p.rho_floor_frac * np.percentile(rho, p.rho_floor_percentile)
    q = np.abs(L) / np.maximum(rho_s, floor)
    valid = np.zeros_like(q, dtype=bool)
    valid[R:-R, R:-R] = True
    s = max(np.median(q[valid]) / MAD_TO_STD, p.s_hat_min)
    q_eff = np.maximum(q - p.k_tau * s, 0.0)
    pool = np.zeros_like(q_eff)
    H, W = q.shape
    for dy, dx in zip(*np.nonzero(disk(p.r))):
        dy, dx = dy - p.r, dx - p.r
        src = q_eff[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)]
        dst = pool[max(0, -dy):H + min(0, -dy), max(0, -dx):W + min(0, -dx)]
        np.maximum(dst, src, out=dst)
    eps = p.k_eps * s
    w = np.maximum(p.w_min, eps / (pool + eps))
    return q, pool, w


def test_matches_reference_implementation():
    D, _ = box_scene()
    D = noisy(D, 1e-3, seed=2)
    out = compute_weights(D, None, P)
    q, pool, w = reference_weights(D, P)
    np.testing.assert_allclose(out["q"], q, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(out["q_pool"], pool, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(out["w"], w, rtol=0, atol=1e-12)


def test_subthreshold_spike_leaves_w_one():
    D0 = random_plane_D(np.random.default_rng(9), 504, 378)
    rho = 1.0 / D0
    rho[180, 250] *= 1.01  # isolated spike on a noise-free plane
    D = 1.0 / rho
    q_max = compute_weights(D, None, P)["q"].max()
    assert q_max > 1e-6
    # Fix the noise scale via the clamp so that tau = 3 * s_hat sits above / below the spike.
    above = Params(s_hat_min=q_max / 1.5)  # tau = 2 q_max
    out = compute_weights(D, None, above)
    assert out["scalars"]["s_hat_clamped"]
    assert np.all(out["w"] == 1.0)
    below = Params(s_hat_min=q_max / 6.0)  # tau = q_max / 2
    out = compute_weights(D, None, below)
    affected = out["w"] < 1
    assert affected.any()
    assert np.array_equal(affected, ndimage.binary_dilation(out["q"] > out["scalars"]["tau"], structure=disk(P.r)))


# 10. Upsample alignment
@pytest.mark.parametrize("c", [40, 101, 250, 377, 460])
def test_upsample_alignment(c):
    W_up, H_up, W_n, H_n = 1023, 768, 504, 378
    s_x = W_n / W_up
    img = np.zeros((H_n, W_n))
    img[:, c] = 1.0
    up = upsample_weights(img, (W_up, H_up), w_min=0.0)
    assert up.shape == (H_up, W_up)
    prof = up[H_up // 2]
    centroid = (prof * np.arange(W_up)).sum() / prof.sum()
    assert centroid == pytest.approx((c + 0.5) / s_x - 0.5, abs=0.1)


def test_deliverable_w_frame_and_range():
    w = np.random.default_rng(0).uniform(0.05, 1, (504, 378))  # upright portrait at network res
    for tag, shape in [(1, (1023, 768)), (6, (768, 1023))]:
        out = deliverable_w(w, (768, 1023), tag, 0.05)
        assert out.shape == shape
        assert out.min() >= 0.05 and out.max() <= 1.0


# 11. Pair weights
def test_pair_weights():
    w = np.random.default_rng(0).uniform(size=(5, 7))
    wx, wy = pair_weights(w)
    assert wx.shape == (5, 6) and wy.shape == (4, 7)
    for i in range(5):
        for j in range(6):
            assert wx[i, j] == min(w[i, j], w[i, j + 1])
    for i in range(4):
        for j in range(7):
            assert wy[i, j] == min(w[i, j], w[i + 1, j])


# B1 sky mask
def test_sky_mask_keeps_holes_and_drops_small_components():
    prob = np.zeros((60, 80))
    prob[:30, :] = 0.9           # sky band touching the top border
    prob[10:16, 30:36] = 0.1     # 6x6 hole (a real object): must stay a hole
    prob[50:54, 5:9] = 0.9       # 16-px sky blob: below min_area_px, removed
    S = sky_mask(prob, prob.shape, P)
    assert S[0, :].all()
    # Closing with the 5-px cross fills only the 4 corner pixels of the square hole; the hole stays.
    assert S[:30].sum() == 30 * 80 - 36 + 4
    assert not S[11:15, 30:36].any() and not S[10:16, 31:35].any()
    assert not S[40:].any()


def test_sky_boundary_responds_interior_zero():
    D, _ = box_scene()
    prob = np.zeros(D.shape)
    prob[:60, :] = 1.0
    out = compute_weights(D, prob, P)
    S = out["sky_mask"]
    assert S[:60].all() and not S[60:].any()
    assert np.all(out["q"][: 60 - P.r] == 0)
    assert out["q"][60 - P.r:62].max() > 1e-3   # the sky boundary band keeps a large response
    assert np.isfinite(out["q"]).all()


def test_smooth_preserves_affine():
    y, x = np.mgrid[0:40, 0:50].astype(np.float64)
    rho = 0.3 + 0.01 * x - 0.004 * y
    rs, L = smooth_and_laplacian(rho, 1.25, 4.0)
    np.testing.assert_allclose(rs, rho, atol=1e-14)
    assert np.abs(L).max() < 1e-14


# T4 threshold floor
def test_threshold_floor_matches_ripple_response_and_off_by_default():
    from illum_baseline.detector import threshold_floor

    lam, a = 15.0, 0.005
    y, x = np.mgrid[0:120, 0:150].astype(np.float64)
    rho = 0.3 * (1 + a * np.sin(2 * np.pi * y / lam))
    out = compute_weights(1.0 / rho, None, Params(a_min=a, lambda_ref=lam))
    q_peak = out["q"][20:-20, 20:-20].max()
    tmin = threshold_floor(a, lam, P.sigma)
    assert out["scalars"]["tau_min"] == pytest.approx(tmin)
    assert q_peak == pytest.approx(tmin, rel=0.05)  # discrete kernels vs the continuous formula
    # The ripple sits at the floor, so q - tau <= 0 (up to the 5%) and w stays 1 where q <= tau_min.
    assert np.all(out["w"][out["q"] <= out["scalars"]["tau"]] == 1.0)
    # a_min = None leaves tau and the outputs untouched.
    D, _ = box_scene()
    a0, b0 = compute_weights(D, None, P), compute_weights(D, None, Params(a_min=None))
    assert "tau_min" not in a0["scalars"]
    np.testing.assert_array_equal(a0["w"], b0["w"])


def test_eps_floor_scales_with_tau_floor_and_is_off_by_default():
    D = noisy(box_scene()[0], 1e-3, seed=3)
    off = compute_weights(D, None, Params(a_min=0.005))
    on = compute_weights(D, None, Params(a_min=0.005, eps_floor=True))
    s_off, s_on = off["scalars"], on["scalars"]
    assert "eps_min" not in s_off and s_off["eps"] == pytest.approx(P.k_eps * s_off["s_hat"])
    assert s_on["eps"] == pytest.approx(max(P.k_eps * s_on["s_hat"], (P.k_eps / P.k_tau) * s_on["tau_min"]))
    np.testing.assert_array_equal(off["q_pool"], on["q_pool"])  # tau unchanged, only the falloff changes
    assert (on["w"] >= off["w"]).all()                          # a larger eps never lowers w


# Trilateral prefilter (stripe alternative)
TRI = Params(prefilter="trilateral")


def test_trilateral_planes_exact():
    rng = np.random.default_rng(11)
    for _ in range(3):
        D = random_plane_D(rng, 504, 378)
        assert compute_weights(D, None, TRI)["q"].max() < 1e-9


def test_trilateral_scale_invariance():
    D = noisy(box_scene()[0], 1e-3, seed=4)
    a, b = compute_weights(D, None, TRI), compute_weights(2.9 * D, None, TRI)
    np.testing.assert_allclose(b["w"], a["w"], rtol=0, atol=1e-9)


def test_trilateral_attenuates_7px_ripple_and_keeps_features():
    D0 = random_plane_D(np.random.default_rng(12), 504, 378)
    v = np.arange(378)[:, None]
    D = 1.0 / ((1.0 / D0) * (1 + 0.003 * np.sin(2 * np.pi * v / 7.0)))
    q0 = compute_weights(D, None, P)["q"][20:-20, 20:-20].max()
    q1 = compute_weights(D, None, TRI)["q"][20:-20, 20:-20].max()
    assert q1 < q0 / 10
    Db, sid = box_scene()
    out = compute_weights(Db, None, TRI)
    feat, _ = features_and_interior(sid, P.R)
    far = ndimage.distance_transform_cdt(~feat, metric="chessboard") > P.R + 9 + 13  # + filter + gradient support
    assert out["q"][far].max() < 1e-9
    assert np.median(out["q"][feat]) > 1e-4


# Scale-ratio test (band-pass Laplacian)
BAND = Params(scale_band=True)


def test_scale_band_planes_exact_and_quadratic_cancels():
    rng = np.random.default_rng(13)
    for _ in range(3):
        assert compute_weights(random_plane_D(rng, 504, 378), None, BAND)["q"].max() < 1e-9
    # A quadratic disparity (constant curvature, like a large cylinder or sphere) cancels in the interior.
    v, u = np.mgrid[0:200, 0:240].astype(np.float64)
    rho = 0.3 + 1e-4 * u + 2e-6 * (u - 120) ** 2 + 1e-6 * (v - 100) ** 2
    L1 = compute_weights(1.0 / rho, None, P)["L"]
    Lb = compute_weights(1.0 / rho, None, BAND)["L"]
    b = int(np.ceil(4.0 * 2 * P.sigma)) + 1
    assert np.abs(Lb[b:-b, b:-b]).max() < 1e-9 * np.abs(L1[b:-b, b:-b]).max()  # cancels to rounding
    assert np.abs(L1[b:-b, b:-b]).min() > 1e-6  # the plain Laplacian does respond everywhere


def test_scale_band_keeps_about_half_of_a_fold_and_is_scale_invariant():
    v, u = np.mgrid[0:120, 0:160].astype(np.float64)
    rho = 0.3 + 4e-3 * np.abs(u - 80.0)  # a fold (roof) along v
    q1 = compute_weights(1.0 / rho, None, P)["q"][60, 76:85].max()
    qb = compute_weights(1.0 / rho, None, BAND)["q"][60, 76:85].max()
    assert 0.4 < qb / q1 < 0.6
    D = noisy(box_scene()[0], 1e-3, seed=5)
    a, c = compute_weights(D, None, BAND), compute_weights(0.37 * D, None, BAND)
    np.testing.assert_allclose(c["w"], a["w"], rtol=0, atol=1e-9)


def test_scale_band_veto_removes_curvature_keeps_fold_and_texture():
    veto = Params(scale_band=True, scale_band_mode="veto", band_factor=4.0)
    v, u = np.mgrid[0:200, 0:240].astype(np.float64)
    b = int(np.ceil(4.0 * 4 * P.sigma)) + 1
    curved = 0.3 + 1e-4 * u + 2e-6 * (u - 120) ** 2
    assert compute_weights(1.0 / curved, None, veto)["q"][b:-b, b:-b].max() < 1e-9 * compute_weights(
        1.0 / curved, None, P)["q"][b:-b, b:-b].max()
    fold = 0.3 + 4e-3 * np.abs(u - 120.0)
    q1 = compute_weights(1.0 / fold, None, P)["q"][100, 116:125].max()
    qv = compute_weights(1.0 / fold, None, veto)["q"][100, 116:125].max()
    assert qv == pytest.approx(q1, rel=0.1)  # an ideal fold is restored by the gain
    D = noisy(random_plane_D(np.random.default_rng(14), 504, 378), 1e-3, seed=6)  # fine "texture"
    a, c = compute_weights(D, None, P), compute_weights(D, None, veto)
    assert np.median(c["q"]) / np.median(a["q"]) > 0.9  # high-frequency content passes the veto
    assert c["scalars"]["s_hat"] == a["scalars"]["s_hat"]  # noise from the non-band response
