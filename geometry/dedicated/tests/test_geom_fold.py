"""TASK_FOLD_CHANNEL F2: two-sided fits and the fold measure, on synthetic scenes with exact surface IDs and normals.

Tests as specified:
1. exact fold angles (30/60/90/120 deg, several orientations, convex and concave), noise-free;
2. wrong K on single planes: dtheta = 0 and Delta rho* = 0;
3. veto: cylinders vetoed (C < C_min), box folds pass (C >= 0.95);
4. noise null: median(T_null) / (2 ln 2) within 20% of the actual covariance inflation;
5. depth_jump not set at folds away from narrow faces (silhouette recall reported only);
6. narrow faces: windows become invalid and the flags are set.
F2 decisions (2026-10-31): test 1 asserts exactness at k = R_s + 2 and <= 0.2 deg at the channel's k1; the veto's k2
windows have no contamination check (test 6 asserts only the k1 flags); Delta rho* is the smallest plane gap over
[-k, k] with planes_cross; the additive null's sigma_geo is solved per image.
"""

import math

import numpy as np
import pytest
from scipy import ndimage

from illum_baseline.geom.synthetic import F_PX, chamfer_scene, fold_scene, rays, scene_objects, step_scene
from illum_baseline.detector import Params
from illum_baseline.geom.camera import Intrinsics
from illum_baseline.geom.candidates import fold_candidates, rounding_width
from illum_baseline.geom.context import GeometryContext
from illum_baseline.geom.fold import FoldChannel, FoldParams
from illum_baseline.geom.twosided import b_tri, null_projection, quad_form, two_sided_fit

TRI = Params(prefilter="trilateral")
K = Intrinsics.centered(F_PX, F_PX, 504, 378)
FP = FoldParams()
NOISE = 5e-4  # i.i.d. relative noise on rho before the trilateral filter (synthetic stand-in for DA3 noise)
R_S = 9       # trilateral support: rho_f is exact only farther than this from a feature


def noisy(D, rel, seed):
    return 1.0 / ((1.0 / D) * (1.0 + rel * np.random.default_rng(seed).normal(size=D.shape)))


def at(a, x):
    """Nearest-pixel lookup of a (H, W) at points x (N, 2) (u, v)."""
    u = np.clip(np.rint(x[:, 0]).astype(int), 0, a.shape[1] - 1)
    v = np.clip(np.rint(x[:, 1]).astype(int), 0, a.shape[0] - 1)
    return a[v, u]


def line_candidates(ctx, sid, k):
    """Fold candidates whose two window centres (at k + m/2) lie on surfaces 1 and 2 and whose full windows
    stay on their own surface."""
    c = fold_candidates(ctx, FP.cand_sigma_d, FP.cand_sigma_T, FP.cand_rate_factor)
    ok = np.ones(len(c.x0), dtype=bool)
    t = np.arange(-FP.w_t, FP.w_t + 1)
    et = np.stack([-c.e1[:, 1], c.e1[:, 0]], -1)
    sides = []
    for sign in (+1, -1):
        centre = at(sid, c.x0 + sign * (k + FP.m / 2) * c.e1)
        for s in np.arange(k - 1, k + FP.m + 2):
            for tt in t:
                x = c.x0 + sign * s * c.e1 + tt * et
                inside = (x[:, 0] >= 0) & (x[:, 0] <= sid.shape[1] - 2) & (x[:, 1] >= 0) & (x[:, 1] <= sid.shape[0] - 2)
                ok &= inside & (at(sid, x) == centre) & at(ctx.normal_valid, x)
        sides.append(centre)
    ok &= (sides[0] != sides[1]) & (sides[0] > 0) & (sides[1] > 0)
    return c, ok


# ---------------------------------------------------------------- 1. exact fold angles

CASES_1 = [(30, 0, True), (60, 135, True), (90, 30, False), (90, 0, True), (120, 90, False), (60, 90, False)]


@pytest.mark.parametrize("dtheta, phi, convex", CASES_1, ids=[f"{a}deg_phi{b}_{'cvx' if c else 'ccv'}" for a, b, c in CASES_1])
def test_1_exact_fold_angles(dtheta, phi, convex, capsys):
    D, sid, ns, sky = fold_scene(dtheta, phi, convex)
    ctx = GeometryContext(D, sky.astype(float), K, TRI)
    truth = math.radians(dtheta)
    # windows beyond the trilateral support (k = R_S + 2: one more px for bilinear corners and the subpixel x0):
    # the two-sided fit itself is exact
    k_far = R_S + 2
    c, ok = line_candidates(ctx, sid, k_far)
    assert ok.sum() >= 50
    r = two_sided_fit(ctx, c.x0[ok], c.e1[ok], k_far, FP.m, FP.w_t, rms_max=None, fallback=False)
    assert r.valid.all()
    err_far = np.abs(r.dtheta - truth)
    # the channel's own skip k1 (from the rounding width)
    width, _ = rounding_width(ctx, c.x0[ok], c.e1[ok], FP.profile_half_len)
    k1 = np.maximum(FP.k_min, np.ceil(width / 2).astype(int) + FP.k_margin)
    c1, ok1 = line_candidates(ctx, sid, int(k1.max()))
    r1 = two_sided_fit(ctx, c1.x0[ok1], c1.e1[ok1], k1.max(), FP.m, FP.w_t, rms_max=None, fallback=False)
    err_k1 = np.abs(r1.dtheta - truth)
    with capsys.disabled():
        print(f"\n[F2 test 1 {dtheta} deg phi {phi} {'convex' if convex else 'concave'}] max |dtheta err|: "
              f"k = {k_far}: {err_far.max():.2e} rad ({ok.sum()} cands); channel k1 = {k1.min()}-{k1.max()}: "
              f"{np.degrees(err_k1.max()):.3f} deg (median {np.degrees(np.median(err_k1)):.3f}), width {np.median(width):.2f} px")
    assert err_far.max() <= 1e-9
    assert np.degrees(err_k1.max()) <= 0.2


# ---------------------------------------------------------------- 2. wrong K

@pytest.mark.parametrize("kappa", [0.78, 1.22])
def test_2_wrong_K_gives_no_fold_on_planes(kappa):
    rng = np.random.default_rng(3)
    r = rays(504, 378)
    for _ in range(3):
        while True:
            n = rng.normal(size=3)
            n[2] = abs(n[2]) + 0.5
            n /= np.linalg.norm(n)
            if (r @ n).min() > 0.05:
                break
        D = rng.uniform(1, 5) / (r @ n)
        ctx = GeometryContext(D, None, K.scaled(kappa), TRI)
        x0 = np.stack([rng.uniform(40, 464, 200), rng.uniform(40, 338, 200)], -1)
        ang = rng.uniform(-np.pi / 2, np.pi / 2, 200)
        e1 = np.stack([np.cos(ang), np.sin(ang)], -1)
        res = two_sided_fit(ctx, x0, e1, FP.k_min, FP.m, FP.w_t, rms_max=None, fallback=False)
        v = res.valid & res.p.planar & res.m.planar
        assert v.sum() > 150
        assert np.abs(res.dtheta[v]).max() < 1e-9
        assert (np.abs(res.d_rho[v]) / res.rho_p_gap[v]).max() < 1e-12


# ---------------------------------------------------------------- 3. veto, 5. depth_jump (one noisy scene)

@pytest.fixture(scope="module")
def objects_run():
    D, sid, nrm, names = scene_objects(radii=(0.5, 1.0, 2.0))
    ctx = GeometryContext(noisy(D, NOISE, 0), None, K, TRI)
    res = FoldChannel().run(ctx)
    df = res.candidates
    x0, e1 = df[["u0", "v0"]].to_numpy(), df[["e1_u", "e1_v"]].to_numpy()
    k1, k2 = df["k1"].to_numpy(), df["k2"].to_numpy()
    sp = at(sid, x0 + (k1 + FP.m / 2)[:, None] * e1)
    sm = at(sid, x0 - (k1 + FP.m / 2)[:, None] * e1)
    # silhouette: a depth jump anywhere between the two window centres (x0 can sit a few px off the true edge)
    ss = np.arange(-1.0, 1.0001, 0.02)
    Zs = np.stack([at(D, x0 + (s * (k1 + FP.m / 2))[:, None] * e1) for s in ss], 1)
    silhouette = (np.abs(np.diff(Zs, axis=1)) / np.minimum(Zs[:, 1:], Zs[:, :-1])).max(1) > 0.03
    solid = np.isin(sp, [1, 3, 4, 5]) & np.isin(sm, [1, 3, 4, 5])  # floor and box faces
    # cylinder interior: every sample of the k1 and k2 windows on one cylinder
    et = np.stack([-e1[:, 1], e1[:, 0]], -1)
    on = (sp == sm) & (sp >= 10)
    for kk in (k1, k2):
        for sign in (1, -1):
            for s in range(FP.m + 1):
                for t in range(-FP.w_t, FP.w_t + 1):
                    on &= at(sid, x0 + sign * (kk + s)[:, None] * e1 + t * et) == sp
    cls = {"cylinder": on,
           "box_fold": ~silhouette & (sp != sm) & solid & (np.isin(sp, [3, 4, 5]) | np.isin(sm, [3, 4, 5])),
           "box_silhouette": silhouette & (np.isin(sp, [3, 4, 5]) | np.isin(sm, [3, 4, 5]))}
    return ctx, res, cls, names, sid


def _veto_classes(df, names, sid):
    """(deep, near, rim) masks: candidates on a cylinder whose nominal k1 and 2 k1 windows are all >= R_s + 2 px from its
    rim (ends) and limb (side silhouettes); the other on-cylinder candidates; rim if the nearest boundary is an end."""
    x0, e1 = df[["u0", "v0"]].to_numpy(), df[["e1_u", "e1_v"]].to_numpy()
    k1, k2 = df["k1"].to_numpy(), df["k2"].to_numpy()
    et = np.stack([-e1[:, 1], e1[:, 0]], -1)
    deep = np.zeros(len(df), dtype=bool)
    on_cyl = np.zeros(len(df), dtype=bool)
    rim = np.zeros(len(df), dtype=bool)
    for nm, i in names.items():
        if not nm.startswith("cyl"):
            continue
        mask = sid == i
        dist = ndimage.distance_transform_edt(mask)
        d_min = np.full(len(df), np.inf)
        for kk in (k1, k2):
            for sign in (1, -1):
                for s in range(FP.m + 1):
                    for t in range(-FP.w_t, FP.w_t + 1):
                        d_min = np.minimum(d_min, at(dist, x0 + sign * (kk + s)[:, None] * e1 + t * et))
        here = at(mask, x0) & at(mask, x0 + (k1 + FP.m / 2)[:, None] * e1) & at(mask, x0 - (k1 + FP.m / 2)[:, None] * e1)
        on_cyl |= here
        deep |= here & (d_min >= R_S + 2)
        vv, uu = np.nonzero(mask)
        v_top, v_bot = vv.min(), vv.max()
        rows = np.clip(np.rint(x0[:, 1]).astype(int), 0, sid.shape[0] - 1)
        d_end = np.minimum(np.abs(x0[:, 1] - v_top), np.abs(x0[:, 1] - v_bot))
        u_l = np.array([uu[vv == r].min() if (vv == r).any() else -1e9 for r in rows])
        u_r = np.array([uu[vv == r].max() if (vv == r).any() else 1e9 for r in rows])
        d_side = np.minimum(np.abs(x0[:, 0] - u_l), np.abs(x0[:, 0] - u_r))
        rim |= here & (d_end < d_side)
    return deep, on_cyl & ~deep, rim


def test_3_veto_cylinders_vetoed_box_folds_pass(objects_run, capsys):
    """Deep interior (every sample of the k1 and k2 windows >= R_s + 2 px from the cylinder's rim (ends) and limb
    (side silhouettes)): 100% vetoed. Passes nearer the rim or limb are reported."""
    ctx, res, cls, names, sid = objects_run
    df = res.candidates
    deep, near_c, rim = _veto_classes(df, names, sid)
    gate, vp = df["gate"].to_numpy(), df["veto_pass"].to_numpy()
    near = near_c & gate
    box = cls["box_fold"] & gate & ~df["no_veto"].to_numpy()
    with capsys.disabled():
        print(f"\n[F2 test 3] cylinder deep interior: {(deep & gate).sum()} gate-passed, {(deep & gate & vp).sum()} pass the "
              f"veto; near rim/limb: {near.sum()} gate-passed, passes rim {(near & vp & rim).sum()} / limb "
              f"{(near & vp & ~rim).sum()} ({(near & vp & df['no_veto']).sum()} as no_veto); box folds: {box.sum()} "
              f"with veto, C min {df['C'][box].min():.3f}")
    assert (deep & gate).sum() > 50 and not (deep & gate & vp).any()
    assert box.sum() > 50 and df["C"][box].min() >= 0.95


# ---------------------------------------------------------------- F7: capped veto

FIXED_RIM_LIMB = (32, 217)  # rim / limb passes of the fixed veto (test 3), the capped veto may not exceed them


def test_7_capped_veto_cylinders_and_box_folds(objects_run, capsys):
    """Capped veto on the same scene: deep interior still 0 passes, rim / limb passes not above the fixed veto's,
    box folds with a decided veto C >= 0.95."""
    ctx, _, cls, names, sid = objects_run
    df = FoldChannel().run(ctx, {"veto_mode": "capped"}).candidates
    deep, near_c, rim = _veto_classes(df, names, sid)
    gate, vp = df["gate"].to_numpy(), df["veto_pass"].to_numpy()
    near = near_c & gate
    box = cls["box_fold"] & gate & df["decidable"].to_numpy() & ~df["no_veto"].to_numpy()
    with capsys.disabled():
        print(f"\n[F7 capped veto] cylinder deep interior: {(deep & gate).sum()} gate-passed, {(deep & gate & vp).sum()} "
              f"pass; near rim/limb: {near.sum()} gate-passed, passes rim {(near & vp & rim).sum()} / limb "
              f"{(near & vp & ~rim).sum()} (narrow_face {(near & df['narrow_face']).sum()}); box folds: {box.sum()} decided, "
              f"C min {df['C'][box].min():.3f}, narrow_face {(cls['box_fold'] & df['narrow_face']).sum()}, "
              f"veto_undecidable {(cls['box_fold'] & df['veto_undecidable']).sum()}")
    assert (deep & gate).sum() > 50 and not (deep & gate & vp).any()
    assert (near & vp & rim).sum() <= FIXED_RIM_LIMB[0] and (near & vp & ~rim).sum() <= FIXED_RIM_LIMB[1]
    assert box.sum() > 50 and df["C"][box].min() >= 0.95


@pytest.mark.parametrize("width_px", [12, 18, 24, 32])
def test_7_capped_veto_narrow_faces(width_px, capsys):
    """Faces 12-32 px wide between two folds: which edges each veto accepts. With the capped veto both edges of faces
    >= 18 px are accepted; a 12-px face is narrower than the k1 windows (k1 = 8, m = 8), its edges are reported."""
    D, sid, ns, (c_ab, c_bc) = chamfer_scene(width_px)
    ctx = GeometryContext(noisy(D, NOISE, 1), None, K, TRI)
    out = {}
    for mode in ("fixed", "capped"):
        df = FoldChannel().run(ctx, {"veto_mode": mode}).candidates
        for nm, c in (("A|B", c_ab), ("B|C", c_bc)):
            near = (np.abs(df["u0"] - c) < 2.5).to_numpy()
            g = near & df["gate"].to_numpy()
            out[mode, nm] = (int(g.sum()), int((near & df["accepted"]).sum()),
                             int((near & df["narrow_face"]).sum()) if mode == "capped" else 0)
    with capsys.disabled():
        print(f"\n[F7 narrow face {width_px} px] " + "; ".join(
            f"{m} {e}: gate {g}, accepted {a}" + (f" (narrow_face {nf})" if m == "capped" else "")
            for (m, e), (g, a, nf) in out.items()))
    if width_px >= 18:
        for e in ("A|B", "B|C"):
            g, a, _ = out["capped", e]
            assert g > 100 and a >= 0.95 * g


def test_5_depth_jump_not_at_folds(objects_run, capsys):
    """No depth_jump on folds at least 2 k1 + m px from any narrow face (a surface narrower than 2 k2 + m, i.e. twice
    its largest inscribed distance); silhouette recall is reported only (the strict recall requirement belongs to
    the depth-channel tests, k >= 11: docs/geometry_channels.md section 5)."""
    _, res, cls, _, sid = objects_run
    df = res.candidates
    valid = df["valid"].to_numpy()
    x0 = df[["u0", "v0"]].to_numpy()
    need_face = 2 * int(df["k2"].max()) + FP.m
    narrow = np.zeros(sid.shape, dtype=bool)
    for s_id in np.unique(sid):
        mask = sid == s_id
        if 2 * ndimage.distance_transform_edt(mask).max() < need_face:
            narrow |= mask
    d_narrow = ndimage.distance_transform_edt(~narrow) if narrow.any() else np.full(sid.shape, np.inf)
    away = at(d_narrow, x0) >= 2 * df["k1"].to_numpy() + FP.m
    fold = cls["box_fold"] & valid
    fold_away = fold & away
    sil = cls["box_silhouette"] & valid
    dj = df["depth_jump"].to_numpy()
    # single folds between two wide faces (noisy): every candidate within 2.5 px of the fold line
    n_single, dj_single = 0, 0
    for dth, phi, cvx in ((30, 0, True), (90, 30, False), (120, 90, False)):
        D, sid1, _, sky = fold_scene(dth, phi, cvx)
        d1 = FoldChannel().run(GeometryContext(noisy(D, NOISE, 3), sky.astype(float), K, TRI)).candidates
        ph = math.radians(phi)
        dist = (d1[["u0", "v0"]].to_numpy() - np.array([K.c_x, K.c_y])) @ np.array([-math.sin(ph), math.cos(ph)])
        line = (np.abs(dist) < 2.5) & d1["valid"].to_numpy() & ~sky[d1["v"], d1["u"]]
        n_single += line.sum()
        dj_single += (line & d1["depth_jump"]).sum()
    with capsys.disabled():
        print(f"\n[F2 test 5] single folds: {n_single} valid, depth_jump {dj_single}; box folds: {fold.sum()} valid, "
              f"{fold_away.sum()} >= 2k1+m from narrow faces (depth_jump {(fold_away & dj).sum()}), all box folds "
              f"depth_jump {(fold & dj).sum()} (reported); silhouette recall {(sil & dj).sum()}/{sil.sum()} "
              f"= {(sil & dj).sum() / max(sil.sum(), 1):.2f} (reported only)")
    assert n_single > 500 and dj_single == 0
    assert not (fold_away & dj).any()


# ---------------------------------------------------------------- 4. noise null

def test_4_noise_null_lambda_matches_inflation(capsys):
    """lambda_hat = median(T_null)/(2 ln 2) against the inflation actually present: the Monte-Carlo E[T]/2 over
    the channel's null sample (T would be chi^2_2, mean 2, if the window covariances were right)."""
    r = rays(504, 378)
    n = np.array([0.2, -0.3, 1.0])
    n /= np.linalg.norm(n)
    D0 = 3.0 / (r @ n)
    lam_hat, T_all = [], []
    for seed in range(4):
        res = FoldChannel().run(GeometryContext(noisy(D0, NOISE, 100 + seed), None, K, TRI))
        t = res.tables["null"]
        T = t["T"][t["valid"]].to_numpy()
        lam_hat.append(res.scalars["lambda_hat_raw"])
        T_all.append(T)
    T_all = np.concatenate(T_all)
    lam_mc = T_all.mean() / 2
    with capsys.disabled():
        print(f"\n[F2 test 4] lambda_hat per image {np.round(lam_hat, 1)}, MC inflation E[T]/2 = {lam_mc:.1f}, "
              f"ratio {np.mean(lam_hat) / lam_mc:.3f}; T P90/median {np.percentile(T_all, 90) / np.median(T_all):.2f} "
              f"(chi2_2: 3.32); clamp {FP.lambda_clamp}")
    assert abs(np.mean(lam_hat) / lam_mc - 1) < 0.2


def test_4b_additive_null_calibrates_median():
    """sigma_geo is solved so that median(T_add over the null sample) = 2 ln 2."""
    D, sid, nrm, names = scene_objects(radii=(0.5, 1.0, 2.0))
    res = FoldChannel().run(GeometryContext(noisy(D, NOISE, 0), None, K, TRI), {"lambda_n": 50.0})
    t = res.tables["null"]
    T_add = t["T_add"][t["valid"]].to_numpy()
    assert res.scalars["sigma_geo"] > 0
    assert np.median(T_add) == pytest.approx(2 * math.log(2), rel=1e-6)


# ---------------------------------------------------------------- Delta rho*: smallest plane gap

def test_gap_folds_cross_steps_exact():
    """Noise-free, windows beyond the trilateral support: folds have planes_cross and Delta rho* = 0; at a step the
    gap is the exact relative depth jump between the two planes at the gap point."""
    k = R_S + 2
    D, sid, ns, sky = fold_scene(90, 30, False)
    ctx = GeometryContext(D, sky.astype(float), K, TRI)
    c, ok = line_candidates(ctx, sid, k)
    r = two_sided_fit(ctx, c.x0[ok], c.e1[ok], k, FP.m, FP.w_t, rms_max=None, fallback=False)
    assert r.planes_cross.all() and np.all(r.d_rho == 0)

    # step: points on the true edge with the true e1 (fold candidates sit at the edge of the invalid-normal band of
    # the dense fit next to a step, a few px off the edge)
    phi = 60
    D, sid, ns = step_scene(1.5, 20, -20, phi)
    ctx = GeometryContext(D, None, K, TRI)
    ph = math.radians(phi)
    L, q = np.array([math.cos(ph), math.sin(ph)]), np.array([-math.sin(ph), math.cos(ph)])
    x0 = np.array([251.5, 188.5]) + np.linspace(-120, 120, 41)[:, None] * L
    e1 = np.tile(q if q[0] > 0 else -q, (len(x0), 1))
    r = two_sided_fit(ctx, x0, e1, k, FP.m, FP.w_t, rms_max=None, fallback=False)
    ok = r.p.planar & r.m.planar
    assert ok.sum() > 30 and not r.planes_cross[ok].any()
    xg = x0[ok] + r.s_gap[ok, None] * e1[ok]
    ray = K.ray(xg[:, 0], xg[:, 1])
    Z = {s: (ns[s] @ np.array([0.0, 0.0, 3.0 * (1.5 if s == 2 else 1.0)])) / (ray @ ns[s]) for s in (1, 2)}
    truth = np.abs(1 / Z[1] - 1 / Z[2]) / np.minimum(1 / Z[1], 1 / Z[2])
    assert np.abs(r.dZ_rel[ok] - truth).max() < 1e-9
    assert np.all(np.abs(r.s_gap[ok]) == k)


def test_b_tri_bounds_held_out_fold():
    """Noise-free fold not in the calibration set (45 deg, phi 60): every window of the channel's candidates at
    k >= k_min passes the contamination threshold b_tri_margin b_tri(k, m)."""
    D, sid, ns, sky = fold_scene(45, 60, True)
    ctx = GeometryContext(D, sky.astype(float), K, TRI)
    c = fold_candidates(ctx, FP.cand_sigma_d, FP.cand_sigma_T, FP.cand_rate_factor)
    for m in (FP.m, FP.m // 2):
        for k in range(FP.k_min, 16):
            r = two_sided_fit(ctx, c.x0, c.e1, k, m, FP.w_t, rms_max=None, fallback=False)
            rr = np.concatenate([r.p.rms_rel[r.p.planar], r.m.rms_rel[r.m.planar]])
            assert rr.size and (rr <= FP.b_tri_margin * b_tri(np.array([k]), m)[0]).all(), (m, k)


# ---------------------------------------------------------------- 6. narrow faces

@pytest.mark.parametrize("width_px, expect", [(12, "fallback"), (18, "retry")])
def test_6_narrow_faces_flagged(width_px, expect, capsys):
    D, sid, ns, (c_ab, c_bc) = chamfer_scene(width_px)
    res = FoldChannel().run(GeometryContext(noisy(D, NOISE, 1), None, K, TRI))
    df = res.candidates
    near = (np.abs(df["u0"] - c_ab) < 2.5).to_numpy()  # candidates on the A|B fold; side "+" (e1_u > 0) faces B
    with capsys.disabled():
        print(f"\n[F2 test 6 width {width_px}] A|B candidates {near.sum()}: planar_p {(near & df['planar_p']).sum()}, "
              f"m_used_p = 4: {(near & (df['m_used_p'] == 4)).sum()}, fallback_p {(near & df['fallback_p']).sum()}, "
              f"gate {(near & df['gate']).sum()}, no_veto {(near & df['no_veto']).sum()}, accepted {(near & df['accepted']).sum()}")
    assert near.sum() > 100
    assert (df["e1_u"][near] > 0).all()
    if expect == "fallback":  # the k1 window into B crosses the B|C fold at both m and m/2
        assert (df["fallback_p"][near]).all()
    else:                     # m/2 fits inside B
        assert (df["planar_p"][near] & (df["m_used_p"][near] == 4)).all()
    g = near & df["gate"].to_numpy()
    with capsys.disabled():
        print(f"    k2 (no contamination check): C median {np.nanmedian(df['C'][g]):.3f}, vetoed {(g & ~df['veto_pass']).sum()}"
              f"/{g.sum()}, no_veto {(g & df['no_veto']).sum()}, k2 windows that would fail the check "
              f"{(g & ~df['k2_clean'] & ~df['no_veto']).sum()}")


def test_4c_additive_local_null():
    """additive_local: with one tile covering the image it is the global additive null; with 48-px tiles the tiles
    are solved where they hold enough null samples and the gate uses the interpolated sigma_geo."""
    D, sid, nrm, names = scene_objects(radii=(0.5, 1.0, 2.0))
    ctx = GeometryContext(noisy(D, NOISE, 0), None, K, TRI)
    one = FoldChannel().run(ctx, {"null_model": "additive_local", "local_tile": 1024})
    df = one.candidates
    v = df["valid"].to_numpy()
    assert np.allclose(df["T_add_local"][v], df["T_add"][v], rtol=1e-12, atol=0)
    assert one.scalars["sigma_geo_local_tiles"] == 1
    assert np.array_equal(df["gate"].to_numpy(), df["gate_add"].to_numpy())
    loc = FoldChannel().run(ctx, {"null_model": "additive_local"})
    assert loc.scalars["sigma_geo_local_tiles"] > 10
    assert np.array_equal(loc.candidates["gate"].to_numpy(), loc.candidates["gate_add_local"].to_numpy())
