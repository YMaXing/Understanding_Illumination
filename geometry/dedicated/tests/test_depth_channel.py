"""TASK_DEPTH_CHANNEL D3: jump, gates, magnitude, weights (DepthChannel). Tests 6-14.

All synthetic depths are canonical (D = Z * 300 / f, the Stage A convention), so the channel's Delta Z is metric.
Noise: i.i.d. relative noise 5e-4 on rho before the trilateral filter, as in the fold tests, unless stated.
"""

import dataclasses
import itertools
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from illum_baseline.detector import Params
from illum_baseline.geom.camera import Intrinsics
from illum_baseline.geom.candidates import _sep, depth_candidates, depth_kernels
from illum_baseline.geom.context import GeometryContext
from illum_baseline.geom.depth import DepthChannel, DepthParams, depth_null, jump
from illum_baseline.geom.synthetic import (CANON, F_PX, affine_step_scene, pole_scene, rays, scene_objects,
                                           step_scene)

TRI = Params(prefilter="trilateral")
DP = DepthParams()
W, H = 504, 378
K = Intrinsics.centered(F_PX, F_PX, W, H)
C = np.array([(W - 1) / 2, (H - 1) / 2])
NOISE = 5e-4
CAN = CANON / F_PX  # metric Z -> canonical D


def noisy(D, rel, seed):
    return 1.0 / ((1.0 / D) * (1.0 + rel * np.random.default_rng(seed).normal(size=D.shape)))


def smear(D, sw):
    """DA3-like boundary smear: Gaussian blur of rho (odd-reflect padding keeps planes exact)."""
    if sw == 0:
        return D
    _, g0, _, _ = depth_kernels(sw)
    return 1.0 / _sep(1.0 / D, g0, g0)


def line_points(pt, direction, margin=15.0, step=0.5):
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    P = np.asarray(pt, float) + np.arange(-700, 700, step)[:, None] * d
    ok = (P[:, 0] > margin) & (P[:, 0] < W - 1 - margin) & (P[:, 1] > margin) & (P[:, 1] < H - 1 - margin)
    return P[ok]


def coverage(df, P, tol=1.5):
    acc = df.loc[df["accepted"], ["u0", "v0"]].to_numpy()
    if not len(acc):
        return 0.0
    d, _ = cKDTree(acc).query(P)
    return float((d <= tol).mean())


def plane_Z(n, P0, x0):
    """Metric depth of the plane through P0 with normal n along the rays through image points x0 (K centred)."""
    r = np.stack([(x0[:, 0] - C[0]) / F_PX, (x0[:, 1] - C[1]) / F_PX, np.ones(len(x0))], -1)
    return (n @ P0) / (r @ n)


# ---------------------------------------------------------------- 6. single steps

def against_slope_scene(r, dv=40.0, v_h=120.0, seed=0, noise=NOISE):
    """Floor rho = g (v - v_h) (camera 1.5 m above it, canonical units) with sky above the horizon v_h; a horizontal
    step at v_h + dv whose upper side (smaller background rho) is raised by Delta = r rho_floor(v_s): the step opposes
    the background slope. Returns (ctx, step line points)."""
    g = 1.0 / (CANON * 1.5)
    v_s = v_h + dv
    a = -g * v_h
    D, _ = affine_step_scene((a, 0.0, g), (a + r * g * dv, 0.0, g), (C[0], v_s), (1.0, 0.0))
    v = np.arange(H, dtype=float)[:, None].repeat(W, 1)
    sky = v < v_h + 1
    D = np.where(sky, 1.0, D)
    D = np.where(sky, 1.0, noisy(D, noise, seed))
    return GeometryContext(D, sky.astype(float), K, TRI), line_points((C[0], v_s), (1.0, 0.0))


def test_6_single_step_recall(capsys):
    rows = []
    for ratio, tilts, phi, sw in itertools.product((1.05, 1.1, 1.5, 3.0), ((20, -20), (0, 40)), (45, 210), (0.0, 0.8)):
        Z, _, _ = step_scene(ratio, *tilts, phi)
        ctx = GeometryContext(noisy(smear(Z * CAN, sw), NOISE, 7), None, K, TRI)
        df = DepthChannel().run(ctx).candidates
        rows.append(("slanted", ratio - 1, tilts, phi, sw,
                     coverage(df, line_points(C, (math.cos(math.radians(phi)), math.sin(math.radians(phi)))))))
    for r in (0.05, 0.08):
        ctx, P = against_slope_scene(r)
        rows.append(("against", r, None, 0, 0.0, coverage(DepthChannel().run(ctx).candidates, P)))
    rec = np.array([x[-1] for x in rows])
    with capsys.disabled():
        print(f"\n[D3 test 6] single steps r 5-200 %, slanted, both polarities, smear 0 / 0.8 px, against slope: recall "
              f"mean {rec.mean():.4f}, min {rec.min():.4f}; against slope r 5 % / 8 %: {rows[-2][-1]:.3f} / "
              f"{rows[-1][-1]:.3f}; smear 0.8: {np.mean([x[-1] for x in rows if x[4] == 0.8]):.4f}")
    assert rec.mean() >= 0.95
    assert rows[-2][-1] >= 0.95 and rows[-1][-1] >= 0.95


# ---------------------------------------------------------------- 7. strict silhouette recall (box scene)

def test_7_box_silhouette_recall(capsys):
    """Recall over the true box silhouette (pixel pairs box | not box with a depth ratio > 1.03), sampled at pair
    midpoints >= 15 px from the border: the share within 1.5 px of an accepted depth candidate (windows at k >= 11)."""
    Z, sid, _, _ = scene_objects()
    ctx = GeometryContext(noisy(Z * CAN, NOISE, 0), None, K, TRI)
    df = DepthChannel().run(ctx).candidates
    box = np.isin(sid, [3, 4, 5])
    pts = []
    for ax in (0, 1):
        a = [slice(None), slice(None)]
        b = [slice(None), slice(None)]
        a[ax], b[ax] = slice(None, -1), slice(1, None)
        za, zb = Z[tuple(a)], Z[tuple(b)]
        sil = (box[tuple(a)] != box[tuple(b)]) & (np.abs(za - zb) / np.minimum(za, zb) > 0.03)
        vv, uu = np.nonzero(sil)
        pts.append(np.stack([uu + 0.5 * (ax == 1), vv + 0.5 * (ax == 0)], -1).astype(float))
    P = np.concatenate(pts)
    P = P[(P[:, 0] > 15) & (P[:, 0] < W - 16) & (P[:, 1] > 15) & (P[:, 1] < H - 16)]
    rec = coverage(df, P)
    with capsys.disabled():
        print(f"\n[D3 test 7] box silhouette recall {rec:.3f} over {len(P)} silhouette samples (k_c >= {DP.k_min})")
    assert rec >= 0.90


# ---------------------------------------------------------------- 8. Delta Z accuracy

def _step_truth(ratio, tilts, phi, Z1=3.0):
    Z, _, ns = step_scene(ratio, *tilts, phi, Z1=Z1)
    return Z, ns, (np.array([0, 0, Z1]), np.array([0, 0, ratio * Z1]))


@pytest.mark.parametrize("args", [(1.05, (20, -20), 30), (1.5, (0, 40), 60), (3.0, (-30, 10), 120)])
def test_8_dZ_exact_noise_free(args, capsys):
    """Noise-free steps on planes: Delta Z at accepted candidates on the edge, >= 30 px from the border, against the
    planes' own depths on the ray through x0 (both from the same x0, so the check is the at-x0 gap and the
    conversion)."""
    ratio, tilts, phi = args
    Z, ns, (P1, P2) = _step_truth(ratio, tilts, phi)
    df = DepthChannel().run(GeometryContext(Z * CAN, None, K, TRI)).candidates
    a = df[df["accepted"]]
    x0 = a[["u0", "v0"]].to_numpy()
    inner = (x0[:, 0] > 30) & (x0[:, 0] < W - 31) & (x0[:, 1] > 30) & (x0[:, 1] < H - 31) & (a["fit_mode"] == "m8")
    x0 = x0[inner]
    truth = plane_Z(ns[2], P2, x0) - plane_Z(ns[1], P1, x0)
    err = np.abs(a["dZ"].to_numpy()[inner] / truth - 1)
    with capsys.disabled():
        print(f"\n[D3 test 8 {args}] noise-free Delta Z rel. error median {np.median(err):.2e}, max {err.max():.2e} "
              f"({inner.sum()} candidates)")
    assert inner.sum() > 100
    assert err.max() <= 1e-6


@pytest.mark.parametrize("sw", [0.5, 0.8, 1.5, 3.0])
def test_8_dZ_with_smear(sw, capsys):
    """DA3-like smear (Gaussian blur of rho, sigma_w <= 3 px), noise-free: median relative error <= 1 %. Smears above
    sigma_w_max fail the sharpness test by design; for them sigma_w_max is raised so the magnitude can be checked."""
    ratio, tilts, phi = 1.5, (20, -20), 30
    Z, ns, (P1, P2) = _step_truth(ratio, tilts, phi)
    p = DP if sw <= DP.sigma_w_max else dataclasses.replace(DP, sigma_w_max=sw + 0.6)  # + the pixel-sampling term
    df = DepthChannel().run(GeometryContext(smear(Z * CAN, sw), None, K, TRI), p).candidates
    a = df[df["accepted"]]
    x0 = a[["u0", "v0"]].to_numpy()
    inner = (x0[:, 0] > 30) & (x0[:, 0] < W - 31) & (x0[:, 1] > 30) & (x0[:, 1] < H - 31)
    truth = plane_Z(ns[2], P2, x0[inner]) - plane_Z(ns[1], P1, x0[inner])
    err = np.abs(a["dZ"].to_numpy()[inner] / truth - 1)
    with capsys.disabled():
        print(f"\n[D3 test 8 smear {sw}] Delta Z rel. error median {np.median(err):.2e}, P95 "
              f"{np.percentile(err, 95):.2e} ({inner.sum()} candidates, k_c {sorted(set(a['k_c'][inner]))})")
    assert inner.sum() > 100
    assert np.median(err) <= 0.01


# ---------------------------------------------------------------- 9. invariances

def test_9_scale_and_focal_invariance():
    Z, _, _, _ = scene_objects()
    D = noisy(Z * CAN, NOISE, 1)
    c, kappa = 2.0, 1.3
    a = DepthChannel().run(GeometryContext(D, None, K, TRI)).candidates
    b = DepthChannel().run(GeometryContext(c * D, None, K, TRI)).candidates
    k = DepthChannel().run(GeometryContext(D, None, K.scaled(kappa), TRI)).candidates
    for other in (b, k):
        np.testing.assert_array_equal(a["accepted"].to_numpy(), other["accepted"].to_numpy())
        np.testing.assert_array_equal(a["reason"].to_numpy(), other["reason"].to_numpy())
    acc = a["accepted"].to_numpy() & np.isfinite(a["dZ"].to_numpy())
    np.testing.assert_allclose(b["dZ"].to_numpy()[acc], c * a["dZ"].to_numpy()[acc], rtol=1e-9)
    np.testing.assert_allclose(k["dZ"].to_numpy()[acc], kappa * a["dZ"].to_numpy()[acc], rtol=1e-9)


# ---------------------------------------------------------------- 10. sky

def _sky_scene(v_sky=150.0, slope=0.25):
    """A slanted plane with the sky above the line v = v_sky + slope (u - c_x)."""
    n = np.array([0.1, -0.5, 1.0])
    n /= np.linalg.norm(n)
    Z = 4.0 / (rays(W, H) @ n)
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    sky = v < v_sky + slope * (u - C[0])
    D = np.where(sky, 1.0, noisy(Z * CAN, NOISE, 4))
    return GeometryContext(D, sky.astype(float), K, TRI), sky


def test_10_sky(capsys):
    ctx, sky = _sky_scene()
    res = DepthChannel().run(ctx)
    df = res.candidates
    a = df[df["accepted"]]
    assert len(a) > 300
    on_sky = a["sky_edge"].to_numpy()
    assert on_sky.mean() > 0.95
    assert np.isinf(a["dZ"].to_numpy()[on_sky]).all()
    assert (a["w_c"].to_numpy()[on_sky] == DP.w_min).all()
    ij = a.loc[on_sky, ["u", "v"]].to_numpy()
    assert (res.w_net[ij[:, 1], ij[:, 0]] == DP.w_min).all()
    # a near window in the sky is rejected: the same edges with n flipped (the sky side declared near)
    f_cands = depth_candidates(ctx, DP)
    nul = depth_null(ctx, DP, f_cands.field)
    P = f_cands.passed
    J = jump(ctx, DP, f_cands.x0[P], -f_cands.n[P], f_cands.k_c[P], f_cands.q_p[P], f_cands.q_m[P], f_cands.h_p[P], f_cands.h_m[P],
             f_cands.s_sep[P], f_cands.rho_s0[P], f_cands.field.s_q, nul, np.empty((0, 2)))
    J0 = jump(ctx, DP, f_cands.x0[P], f_cands.n[P], f_cands.k_c[P], f_cands.q_p[P], f_cands.q_m[P], f_cands.h_p[P], f_cands.h_m[P],
              f_cands.s_sep[P], f_cands.rho_s0[P], f_cands.field.s_q, nul, np.empty((0, 2)))
    edge = J0["sky_edge"]
    assert edge.sum() > 300 and J["sky_near"][edge].all() and not J0["sky_near"][edge].any()
    with capsys.disabled():
        print(f"\n[D3 test 10] sky boundary: {on_sky.sum()} accepted sky edges (Delta Z = inf, w = w_min); "
              f"flipped n: {J['sky_near'][edge].sum()}/{edge.sum()} sky_near")


# ---------------------------------------------------------------- 11. distant far side

def test_11_distant_far_side_lower_bound(capsys):
    """A near wall (Z = 3 m) above a floor whose horizon coincides with the wall's lower edge: the far side's disparity
    extrapolates to 0 at x0 (rho_hat_- ~ 0, not sky; relative noise 5e-4). Delta Z is the flagged lower bound:
    positive, and infinite where even the upper confidence bound of rho_hat_- is <= 0 (x0 sits ~0.2 px on the near
    side of the pixel edge, so the extrapolated far plane is slightly negative there; the true jump is infinite).
    The relative-jump gate uses the matching lower bound of r."""
    rho_near = 1.0 / (3.0 * CAN)
    g = 1.0 / (CANON * 1.5)  # floor 1.5 m below the camera
    v_e = 200.5              # the wall's lower edge = the floor's horizon (between pixel rows)
    D, _ = affine_step_scene((-g * v_e, 0.0, g), (rho_near, 0.0, 0.0), (C[0], v_e), (1.0, 0.0))
    res = DepthChannel().run(GeometryContext(noisy(D, NOISE, 3), None, K, TRI))
    df = res.candidates
    a = df[df["accepted"] & (np.abs(df["v0"] - v_e) < 1.5) & (df["u0"] > 30) & (df["u0"] < W - 31)]
    assert len(a) > 200
    lb = a["lower_bound"].to_numpy()
    dz = a["dZ"].to_numpy()
    with capsys.disabled():
        print(f"\n[D3 test 11] wall edge on the floor's horizon: {lb.mean():.3f} of {len(a)} accepted edge candidates "
              f"flagged lower_bound (r lower bound {a['r_lower_bound'].mean():.3f}); Delta Z lower bound: "
              f"{np.isinf(dz[lb]).mean():.3f} unbounded (inf), finite median "
              f"{np.median(dz[lb][np.isfinite(dz[lb])]) if np.isfinite(dz[lb]).any() else float('nan'):.1f} m; "
              f"rho_hat_- median {np.median(a['rho_m']):.2e} (near {rho_near:.2e}); w_c {a['w_c'].median():.2f}")
    assert lb.mean() > 0.95
    assert not a["sky_edge"].any()
    assert (dz[lb] > 0).all()
    assert (a["w_c"].to_numpy()[lb] <= DP.dZ0 / (np.where(np.isinf(dz[lb]), 0, dz[lb]) + DP.dZ0) + 1e-12).all()


# ---------------------------------------------------------------- 12. thin structures

@pytest.mark.parametrize("width", [4, 6, 8, 12, 16, 20])
def test_12_thin_structures(width, capsys):
    Z, sid, (ul, ur) = pole_scene(width)
    ctx = GeometryContext(noisy(Z * CAN, NOISE, 5), None, K, TRI)
    res = DepthChannel().run(ctx)
    df = res.candidates
    a = df[df["accepted"]]
    out = {}
    for name, u_e in (("left", ul), ("right", ur)):
        P = np.stack([np.full(300, u_e), np.linspace(30, H - 31, 300)], -1)
        rec = coverage(df, P)
        near = a[np.abs(a["u0"] - u_e) < 1.5]
        x0 = near[["u0", "v0"]].to_numpy()
        th = math.radians(20.0)
        nb = np.array([math.sin(th), 0.0, -math.cos(th)])
        truth = plane_Z(nb, np.array([0, 0, 6.0]), x0) - 3.0
        err = np.median(np.abs(near["dZ"].to_numpy() / truth - 1)) if len(near) else float("nan")
        modes = near["fit_mode"].value_counts().to_dict()
        out[name] = (rec, err, modes)
        if width < DP.k_min and len(near):
            assert near["fit_mode"].isin(["const", "doublet"]).all()  # windows at k_c lie beyond the pole: flagged
            assert (near["thin"] == (near["fit_mode"] == "doublet")).all()
    # components merging both sides of the pole (one contour component holding candidates of both edges)
    side = np.where(np.abs(a["u0"] - ul) < 1.5, 0, np.where(np.abs(a["u0"] - ur) < 1.5, 1, -1))
    both = [cid for cid in set(a["component_id"]) if {0, 1} <= set(side[a["component_id"].to_numpy() == cid])]
    with capsys.disabled():
        print(f"\n[D3 test 12 pole {width} px] " + "; ".join(
            f"{k}: recall {v[0]:.2f}, Delta Z median rel. err {v[1]:.3f}, modes {v[2]}" for k, v in out.items())
            + f"; components merging both sides: {len(both)}")
    if width > DP.link_radius:
        assert not both


# ---------------------------------------------------------------- 13. cylinder limbs

def test_13_cylinder_limb_bias(capsys):
    """Vertical cylinders (R 0.25, 0.5, 1 m): Delta Z at accepted limb candidates (near side on a cylinder 2.5 px
    along +n, the edge vertical: |n_v| < 0.5, away from the ends) against the analytic value: the limb depth (the
    ray's closest approach to the axis) and the far surface's own depth 2.5 px beyond the edge. The affine near-side
    extrapolation overestimates: reported with the bound (the cylinder's depth change within k px of the limb)."""
    radii = (0.25, 0.5, 1.0)
    Z, sid, _, _ = scene_objects(radii=radii)
    res = DepthChannel().run(GeometryContext(noisy(Z * CAN, NOISE, 6), None, K, TRI))
    a = res.candidates[res.candidates["accepted"]]
    x0, n = a[["u0", "v0"]].to_numpy(), a[["n_u", "n_v"]].to_numpy()
    k = a["k_c"].to_numpy()

    def at(arr, x):
        return arr[np.clip(np.rint(x[:, 1]).astype(int), 0, H - 1), np.clip(np.rint(x[:, 0]).astype(int), 0, W - 1)]

    near_sid, far_Z = at(sid, x0 + 2.5 * n), at(Z, x0 - 2.5 * n)
    near_k = at(Z, x0 + k[:, None] * n)
    xs = np.linspace(0.6, 2.6, len(radii))
    lines, worst, ok_bound = [], 0.0, []
    for i, (R, xc) in enumerate(zip(radii, xs)):
        zc = 5.0 + 1.5 * i
        dx = (x0[:, 0] - C[0]) / F_PX
        t_close = (xc * dx + zc) / (dx**2 + 1)
        Y = t_close * (x0[:, 1] - C[1]) / F_PX
        sel = (near_sid == 10 + i) & (np.abs(n[:, 1]) < 0.5) & (Y > -0.8) & (Y < 1.3) & (at(sid, x0 - 2.5 * n) != 10 + i)
        if not sel.any():
            lines.append(f"R {R} m: no accepted limb candidate")
            continue
        truth = far_Z[sel] - t_close[sel]
        rel = a["dZ"].to_numpy()[sel] / truth - 1
        bound = (near_k[sel] - t_close[sel]) / truth
        worst = max(worst, float(np.median(np.abs(rel))))
        ok_bound.append(abs(np.median(rel)) <= abs(np.median(bound)))
        lines.append(f"R {R} m: {sel.sum()} limb candidates, Delta Z bias median {np.median(rel):+.4f} (P10 "
                     f"{np.percentile(rel, 10):+.4f}, P90 {np.percentile(rel, 90):+.4f}); cylinder depth change within "
                     f"k px of the limb {np.median(bound):.4f} of Delta Z")
    with capsys.disabled():
        print("\n[D3 test 13] cylinder limbs: " + "; ".join(lines))
    assert all(ok_bound)  # the spec's bound: the bias is at most the occluder's depth change within k px


# ---------------------------------------------------------------- 14. noise null

def test_14_lambda_matches_monte_carlo(capsys):
    r = rays(W, H)
    out = []
    for nvec, d in (([0.2, -0.3, 1.0], 3.0), ([-0.45, 0.25, 1.0], 2.5)):
        n = np.asarray(nvec, float) / np.linalg.norm(nvec)
        ctx = GeometryContext(noisy(d / (r @ n) * CAN, NOISE, 9), None, K, TRI)
        f = depth_candidates(ctx, DP).field
        nul = depth_null(ctx, DP, f)
        mc = float(nul.z2.mean())
        out.append((nul.lambda_hat, mc))
        assert abs(nul.lambda_hat / mc - 1) <= 0.2
    with capsys.disabled():
        print("\n[D3 test 14] lambda_hat_Delta_rho vs Monte-Carlo mean(z^2): "
              + "; ".join(f"{a:.1f} vs {b:.1f} ({a / b - 1:+.1%})" for a, b in out))
