"""TASK_FOLD_CHANNEL F1: context, dense edge-aware plane fit (N1) and normals (N2).

Refactor checks: detector.trilateral is bit-identical to its pre-refactor implementation, and the context's
disparity fields are bit-identical to compute_weights'.
Tests 1-6 (synthetic, noise-free unless stated), K of BASELINE_SPEC test 5: f = 400 px, centred principal point.
"""

import json
import math

import numpy as np
import pytest
from scipy import ndimage

from illum_baseline import PROJECT_ROOT
from illum_baseline.detector import Params, compute_weights, gaussian_gradient, trilateral
from illum_baseline.geom.camera import Intrinsics
from illum_baseline.geom.context import GeometryContext
from illum_baseline.geom.normals import normals, plane_distance
from illum_baseline.geom.planefit import DenseFitParams, dense_fit
from test_detector import CANON, F_PX, box_scene, noisy, random_plane_D, rays

TRI = Params(prefilter="trilateral", tri_shat="filtered", scale_band=True, scale_band_mode="veto", band_factor=4.0)
FIT = DenseFitParams()
K_L = Intrinsics.centered(F_PX, F_PX, 504, 378)
# Box scene surfaces -> unit normal toward the camera (floor, wall, box x face at x = 0.3, top y = 0.6, front z = 2)
BOX_NORMALS = {1: (0, -1, 0), 2: (0, 0, -1), 3: (-1, 0, 0), 4: (0, -1, 0), 5: (0, 0, -1)}


def _old_trilateral(rho, ground, rho_floor, sigma_s, sigma_r, sigma_g):
    """Frozen copy of detector.trilateral before the tangent_plane_weights refactor (commit cadd614)."""
    H, W = rho.shape
    gu, gv = gaussian_gradient(rho, sigma_g)
    Rs = int(math.ceil(3 * sigma_s))
    P = np.pad(rho, Rs, mode="reflect", reflect_type="odd")
    G = np.pad(ground, Rs, mode="edge")
    inv = 1.0 / (np.maximum(rho, rho_floor) * sigma_r)
    num = np.zeros_like(rho)
    wsum = np.zeros_like(rho)
    for dy in range(-Rs, Rs + 1):
        for dx in range(-Rs, Rs + 1):
            if dx * dx + dy * dy > Rs * Rs:
                continue
            ws = math.exp(-(dx * dx + dy * dy) / (2 * sigma_s**2))
            nb = P[Rs + dy:Rs + dy + H, Rs + dx:Rs + dx + W]
            e = nb - rho - (gu * dx + gv * dy)
            w = ws * np.exp(-0.5 * (e * inv) ** 2) * G[Rs + dy:Rs + dy + H, Rs + dx:Rs + dx + W]
            num += w * e
            wsum += w
    out = rho.copy()
    out[ground] = rho[ground] + num[ground] / wsum[ground]
    return out


def _sky_top(H, W, rows):
    sky = np.zeros((H, W))
    sky[:rows] = 1.0
    return sky


def _real(stem="taiwei_cui_079"):
    sa = PROJECT_ROOT / "baseline_output" / "stage_a"
    if not (sa / f"{stem}.npz").exists():
        pytest.skip(f"no Stage A cache for {stem}")
    c = np.load(sa / f"{stem}.npz")
    return c["D"], c["sky_prob"].astype(np.float64), json.loads((sa / f"{stem}.json").read_text())


def _angle(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


# ---------------------------------------------------------------- refactor is bit-for-bit

def test_trilateral_bitwise_equals_pre_refactor_synthetic():
    D, _ = box_scene()
    D = noisy(D, 2e-3, seed=3)
    sky = _sky_top(*D.shape, 40)
    dp = GeometryContext(D, sky, K_L, TRI).disparity
    assert dp.S.any()
    args = (dp.rho, dp.ground, dp.rho_floor, 3.0, 0.01, 3.0)
    assert np.array_equal(trilateral(*args), _old_trilateral(*args))


def test_trilateral_bitwise_equals_pre_refactor_real():
    D, sky, _ = _real("taiwei_cui_070")  # the image with sky
    dp = GeometryContext(D, sky, Intrinsics.centered(300, 300, D.shape[1], D.shape[0]), TRI).disparity
    args = (dp.rho, dp.ground, dp.rho_floor, 3.0, 0.01, 3.0)
    assert np.array_equal(trilateral(*args), _old_trilateral(*args))


def test_context_disparity_bitwise_equals_detector():
    D, sky, meta = _real("taiwei_cui_070")
    ctx = GeometryContext(D, sky, Intrinsics.from_stage_a(meta), TRI)
    out = compute_weights(D, sky, TRI)
    assert np.array_equal(ctx.S, out["sky_mask"])
    assert np.array_equal(ctx.rho, out["rho"])
    assert ctx.rho_floor == out["scalars"]["rho_floor"]
    assert np.array_equal(ctx.rho_f, out["rho_filtered"])


def test_context_is_read_only():
    D, _ = box_scene()
    ctx = GeometryContext(D, None, K_L, TRI)
    for a in (ctx.rho, ctx.rho_f, ctx.dense.alpha, ctx.n_hat, ctx.sigma_theta, ctx.normal_valid):
        with pytest.raises(ValueError):
            a[0, 0] = 0


def test_intrinsics_from_stage_a():
    _, _, meta = _real()
    K = Intrinsics.from_stage_a(meta)
    net, gc = meta["network"], meta["geocalib"]
    assert (K.W, K.H) == (net["W_n"], net["H_n"])
    assert (K.c_x, K.c_y) == ((K.W - 1) / 2, (K.H - 1) / 2)
    assert K.f_x == gc["fx"] * net["s_x"] and K.f_y == gc["fy"] * net["s_y"]
    assert (K.f_x + K.f_y) / 2 == pytest.approx(meta["focal"]["f_n"], rel=1e-12)


# ---------------------------------------------------------------- 1. planes exact

@pytest.mark.parametrize("WH", [(504, 378), (378, 504)])
def test_1_random_planes_exact(WH):
    W, H = WH
    K = Intrinsics.centered(F_PX, F_PX, W, H)
    rng = np.random.default_rng(21)
    r = rays(W, H)
    for _ in range(5):
        D = random_plane_D(rng, W, H)
        # recover the plane: rho = m.r with m = n / d0 (up to the canonical scale)
        rho = 1.0 / D
        m = np.linalg.lstsq(r.reshape(-1, 3), rho.ravel(), rcond=None)[0]
        n_true = -m / np.linalg.norm(m)
        ctx = GeometryContext(D, None, K, TRI)
        v = ctx.normal_valid
        assert v.mean() > 0.9
        assert np.abs(ctx.n_hat[v] - n_true).max() < 1e-12
        assert np.all(np.sum(ctx.n_hat[v] * r[v], -1) < 0)  # toward the camera


def test_1_box_scene_planes_exact_away_from_features():
    D, sid = box_scene()
    ctx = GeometryContext(D, None, K_L, TRI)
    b = np.zeros(sid.shape, dtype=bool)
    b[:, 1:] |= sid[:, 1:] != sid[:, :-1]
    b[1:, :] |= sid[1:, :] != sid[:-1, :]
    # rho_f is exact where its R_s = 9 disk lies on one plane; the fit window adds R_n = 9
    far = ndimage.distance_transform_edt(~b) > FIT.R_n + 9 + 1
    checked = 0
    for s, n in BOX_NORMALS.items():
        m = far & (sid == s) & ctx.normal_valid
        if m.any():  # the box's x face is narrower than 2 x 19 px: no pixel qualifies
            assert np.abs(ctx.n_hat[m] - np.array(n, dtype=float)).max() < 1e-12, s
            checked += m.sum() > 50
    assert checked >= 4  # floor, wall, box top, box front


# ---------------------------------------------------------------- 2. checks

def test_2_camera_facing_plane():
    D = np.full((378, 504), 2.0)
    ctx = GeometryContext(D, None, K_L, TRI)
    v = ctx.normal_valid
    assert np.abs(ctx.n_hat[v] - np.array([0.0, 0.0, -1.0])).max() < 1e-12


@pytest.mark.parametrize("h", [1.5, 0.4])
def test_2_floor_normal_and_height(h, capsys):
    H, W = 378, 504
    K = K_L
    _, v = K.pixel_grid()
    rows = int(K.c_y) + 30
    with np.errstate(divide="ignore"):
        D = np.where(v > K.c_y + 1, F_PX * h / (v - K.c_y), 1.0)  # metric depth: rho = (v - c_y) / (f h)
    ctx = GeometryContext(D, _sky_top(H, W, rows), K, TRI)
    # The sky edge acts like a feature: rho_f is exact only where its R_s = 9 disk avoids the sky (the gradient
    # support sees rho = 0 there), and the fit window adds R_n = 9.
    ok = ctx.normal_valid & (v >= rows + FIT.R_n + 9 + 1)
    assert ok.sum() > 0.3 * ok.size
    assert np.abs(ctx.n_hat[ok] - np.array([0.0, -1.0, 0.0])).max() < 1e-12
    d0 = plane_distance(ctx.dense.alpha[ok], ctx.normals.mu[ok])
    assert np.abs(d0 / h - 1).max() < 1e-12
    band = ctx.normal_valid & ~ok
    with capsys.disabled():
        print(f"\n[F1 test 2 floor h={h}] max |n - n_true| within {FIT.R_n + 9 + 1} px of the sky: "
              f"{np.abs(ctx.n_hat[band] - np.array([0.0, -1.0, 0.0])).max():.2e}")


# ---------------------------------------------------------------- 3. depth scale

@pytest.mark.parametrize("c", [0.37, 2.9])
def test_3_depth_scale_invariance(c):
    D, _ = box_scene()
    D = noisy(D, 1e-3, seed=4)
    sky = _sky_top(*D.shape, 30)
    a = GeometryContext(D, sky, K_L, TRI)
    b = GeometryContext(c * D, sky, K_L, TRI)
    assert np.array_equal(a.normal_valid, b.normal_valid)
    v = a.normal_valid
    assert np.abs(a.n_hat[v] - b.n_hat[v]).max() < 1e-9
    assert np.allclose(a.sigma_theta[v], b.sigma_theta[v], rtol=1e-6, atol=0)


# ---------------------------------------------------------------- 4. focal

@pytest.mark.parametrize("kappa", [0.78, 1.22])
def test_4_focal_scales_tan_theta(kappa):
    """f -> kappa f scales mu_x, mu_y by kappa and leaves mu_z alone, so the tangent of the angle between n_hat and
    the optical axis scales by exactly kappa at every pixel (near the principal point that is the angle to the ray)."""
    D, sid = box_scene()
    ctx = GeometryContext(D, None, K_L, TRI)
    fit, sel = ctx.dense, np.nonzero(ctx.normal_valid)
    args = (fit.alpha[sel], fit.beta[sel], fit.cov[sel][:, 1:, 1:], ctx.u[sel], ctx.v[sel])
    n0, n1 = normals(*args, K_L).n_hat, normals(*args, K_L.scaled(kappa)).n_hat
    with np.errstate(divide="ignore"):  # n_z = 0 on the box's x face
        t0 = np.hypot(n0[:, 0], n0[:, 1]) / np.abs(n0[:, 2])
        t1 = np.hypot(n1[:, 0], n1[:, 1]) / np.abs(n1[:, 2])
    ok = (t0 > 1e-6) & (t0 < 1e6)
    assert ok.sum() > 1000
    assert np.abs(t1[ok] / t0[ok] / kappa - 1).max() < 1e-9
    # mu_z does not depend on f, so n_z keeps its sign
    assert np.array_equal(np.sign(n0[:, 2]), np.sign(n1[:, 2]))


# ---------------------------------------------------------------- 5. covariance

def _plane_crop(n, d0, u0, v0, size):
    """rho = (n/d0).r on the network-res K, cropped to [v0:v0+size, u0:u0+size] with the matching K."""
    r = rays(504, 378)[v0:v0 + size, u0:u0 + size]
    rho = r @ (np.asarray(n, float) / d0)
    assert rho.min() > 0
    K = Intrinsics(F_PX, F_PX, K_L.c_x - u0, K_L.c_y - v0, size, size)
    return rho, K


@pytest.mark.parametrize("case", ["slanted", "grazing_floor"])
def test_5_sigma_theta_matches_monte_carlo(case, capsys):
    if case == "slanted":
        n = np.array([0.3, -0.2, 1.0])
        rho, K = _plane_crop(n / np.linalg.norm(n), 3.0, 300, 120, 56)
    else:
        rho, K = _plane_crop([0.0, 1.0, 0.0], 1.5, 220, 300, 56)  # floor 1.5 m below, rows 300-356
    rel, n_mc = 1e-3, 200
    ground = np.ones(rho.shape, dtype=bool)
    rho_floor = 0.5 * np.percentile(rho, 1)
    rng = np.random.default_rng(0)
    ns, s2 = [], []
    for _ in range(n_mc):
        rho_f = rho * (1 + rel * rng.normal(size=rho.shape))
        fit = dense_fit(rho_f, ground, rho_floor, FIT)
        u, v = K.pixel_grid()
        nv = normals(fit.alpha, fit.beta, fit.cov[..., 1:, 1:], u, v, K)
        ns.append(nv.n_hat)
        s2.append(nv.sigma_theta**2)
    valid = fit.valid
    ns, s2 = np.stack(ns)[:, valid], np.stack(s2)[:, valid]
    nbar = ns.mean(0)
    nbar /= np.linalg.norm(nbar, axis=-1, keepdims=True)
    emp2 = np.sum((ns - nbar) ** 2, -1).mean(0)  # per pixel E|n - nbar|^2 = trace Sigma_n
    ana2 = s2.mean(0)
    ratio = math.sqrt(emp2.mean() / ana2.mean())
    per_px = np.sqrt(emp2 / ana2)
    with capsys.disabled():
        print(f"\n[F1 test 5 {case}] MC/analytic sigma_theta = {ratio:.3f} (per pixel P5/P50/P95 "
              f"{np.percentile(per_px, 5):.3f}/{np.median(per_px):.3f}/{np.percentile(per_px, 95):.3f}), "
              f"analytic median {np.degrees(np.sqrt(np.median(ana2))):.3f} deg, {valid.sum()} px, {n_mc} draws")
    assert abs(ratio - 1) < 0.2


# ---------------------------------------------------------------- 6. ripple

@pytest.mark.parametrize("case", ["plane_rows", "plane_oblique", "floor_rows"])
def test_6_ripple_through_trilateral(case, capsys):
    W, H = 504, 378
    u, v = K_L.pixel_grid()
    r = rays(W, H)
    sky = None
    if case.startswith("plane"):
        n = np.array([0.25, -0.35, 1.0])
        n /= np.linalg.norm(n)
        rho = r @ n / 3.0
    else:
        rows = int(K_L.c_y) + 30
        sky = _sky_top(H, W, rows)
        n = np.array([0.0, 1.0, 0.0])
        with np.errstate(divide="ignore"):
            rho = np.where(v > K_L.c_y + 1, (v - K_L.c_y) / (F_PX * 1.5), 1.0)
    phase = 2 * np.pi * (v / 7.0 if case.endswith("rows") else (u + v) / (7.0 * math.sqrt(2)))
    D = 1.0 / (rho * (1 + 2e-3 * np.sin(phase)))
    ctx = GeometryContext(D, sky, K_L, TRI)
    ok = ctx.normal_valid
    err = np.degrees(_angle(ctx.n_hat[ok], -n))
    raw = dense_fit(ctx.rho, ctx.omega, ctx.rho_floor, FIT)  # for the report only: fit on unfiltered rho
    nr = normals(raw.alpha, raw.beta, raw.cov[..., 1:, 1:], u, v, K_L)
    err_raw = np.degrees(_angle(nr.n_hat[raw.valid], -n))
    gu, gv = gaussian_gradient(ctx.rho, 3.0)
    with np.errstate(divide="ignore", invalid="ignore"):  # rho = 0 on the sky
        ng = normals(ctx.rho, np.stack([gu, gv], -1), np.zeros((H, W, 2, 2)), u, v, K_L).n_hat
    err_g = np.degrees(_angle(ng[ok], -n))
    with capsys.disabled():
        print(f"\n[F1 test 6 {case}] median normal error: rho_f + fit {np.median(err):.4f} deg "
              f"(P95 {np.percentile(err, 95):.4f}); raw rho + fit {np.median(err_raw):.3f} deg; "
              f"raw rho + sigma_g=3 gradient {np.median(err_g):.3f} deg")
    assert np.median(err) < 0.1


# ---------------------------------------------------------------- N1 gradient initialisation (F2 follow-up 4)

@pytest.mark.parametrize("args", [(1.1, 20, -20, 60), (1.5, 20, -20, 60), (3.0, 0, 40, 30), (2.0, -30, 10, 90)])
def test_n1_weight_sum_at_steps(args):
    """Side-window gradient init: sum W >= min_weight_sum everywhere except within 1 px of a depth step (with the
    Gaussian gradient it fell to 3-5 within 3 px of a 1.5x step); the planes stay exact away from the step."""
    from illum_baseline.geom.synthetic import step_scene

    D, sid, ns = step_scene(*args)
    ctx = GeometryContext(D, None, K_L, TRI)
    ph = math.radians(args[3])
    q = np.array([-math.sin(ph), math.cos(ph)])
    dd = (ctx.u - K_L.c_x) * q[0] + (ctx.v - K_L.c_y) * q[1]
    R = FIT.R_n
    inner = (ctx.u >= R) & (ctx.u < K_L.W - R) & (ctx.v >= R) & (ctx.v < K_L.H - R)
    assert ctx.dense.m0[inner & (np.abs(dd) > 1)].min() >= FIT.min_weight_sum
    n_true = np.where((sid == 1)[..., None], ns[1], ns[2])
    far = ctx.normal_valid & (np.abs(dd) > R + 9 + 1)
    assert np.abs(ctx.n_hat[far] - n_true[far]).max() <= 1e-9


def test_n1_weight_sum_box_scene():
    D, sid = box_scene()
    ctx = GeometryContext(D, None, K_L, TRI)
    b = np.zeros(sid.shape, dtype=bool)
    b[:, 1:] |= sid[:, 1:] != sid[:, :-1]
    b[1:, :] |= sid[1:, :] != sid[:-1, :]
    R = FIT.R_n
    inner = np.zeros(sid.shape, dtype=bool)
    inner[R:-R, R:-R] = True
    far = inner & (ndimage.distance_transform_edt(~b) > 1)
    assert ctx.dense.m0[far].min() >= FIT.min_weight_sum


def test_n1_gaussian_option_is_the_f1_fit():
    """g_init = gaussian, refit = 0 is the plain F1 fit; on a plane both inits agree to rounding."""
    D, _ = box_scene()
    a = GeometryContext(D, None, K_L, TRI, DenseFitParams(g_init="gaussian", refit=0))
    b = GeometryContext(D, None, K_L, TRI)
    v = a.normal_valid & b.normal_valid
    assert v.mean() > 0.8
    with pytest.raises(ValueError):
        dense_fit(a.rho_f, a.omega, a.rho_floor, DenseFitParams(g_init="bogus"))
