"""Synthetic scenes with exact surface IDs and normals: fold-channel tests and calibration (scripts/calibrate_fold.py).

All return metric depth Z (= canonical D up to a global scale, which the detectors ignore) at network resolution,
upright, for a pinhole camera with f = F_PX and the principal point at the image centre.
"""

import numpy as np

F_PX = 400.0


def rays(W, H, f=F_PX):
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    return np.stack([(u - (W - 1) / 2) / f, (v - (H - 1) / 2) / f, np.ones_like(u)], axis=-1)


def fold_scene(dtheta_deg, phi_deg, convex=True, W=504, H=378, Z0=4.0, f=F_PX):
    """Two planes meeting along a fronto-parallel 3D line through (0, 0, Z0) whose image is a line through the
    principal point at angle phi (direction (cos phi, sin phi) in (u, v)). Camera-facing normals
    n_pm = cos(h) (0, 0, -1) +- sin(h) q, q = (-sin phi, cos phi, 0), h = dtheta / 2, so angle(n+, n-) = dtheta.
    convex: ridge toward the camera (the visible surface is the farther plane); else a valley (the nearer one).
    Returns D (metric Z), sid (1: plane n+, 2: plane n-, 0: excluded), normals {1: n+, 2: n-}, sky (bool;
    grazing or back-facing rays)."""
    r = rays(W, H, f)
    h = np.radians(dtheta_deg) / 2
    ph = np.radians(phi_deg)
    q = np.array([-np.sin(ph), np.cos(ph), 0.0])
    n0 = np.array([0.0, 0.0, -1.0])
    ns = {1: np.cos(h) * n0 + np.sin(h) * q, 2: np.cos(h) * n0 - np.sin(h) * q}
    P0 = np.array([0.0, 0.0, Z0])
    ts = {}
    for s, n in ns.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (n @ P0) / (r @ n)
        ts[s] = np.where(t > 0, t, np.nan)
    if convex:
        pick = np.where(np.nan_to_num(ts[1], nan=-np.inf) >= np.nan_to_num(ts[2], nan=-np.inf), 1, 2)
    else:
        pick = np.where(np.nan_to_num(ts[1], nan=np.inf) <= np.nan_to_num(ts[2], nan=np.inf), 1, 2)
    t = np.where(pick == 1, ts[1], ts[2])
    nvis = np.where((pick == 1)[..., None], ns[1], ns[2])
    rhat = r / np.linalg.norm(r, axis=-1, keepdims=True)
    sky = ~np.isfinite(t) | (np.sum(nvis * rhat, -1) > -0.15)
    D = np.where(sky, 1.0, t)  # Z = t since r_z = 1
    sid = np.where(sky, 0, pick)
    return D, sid, ns, sky


def scene_objects(W=504, H=378, f=F_PX, radii=(0.25, 0.5, 1.0)):
    """Floor (camera 1.5 m high), back wall at z = 12, an axis-aligned box on the floor, and vertical cylinders
    (axis along Y) of the given radii standing on the floor. Returns D (metric Z), sid, normals (H, W, 3) toward
    the camera, and a dict name -> sid. sid: 1 floor, 2 wall, 3-5 box faces (x, top, front), 10+i cylinder i."""
    r = rays(W, H, f)
    dx, dy = r[..., 0], r[..., 1]
    t = np.full((H, W), 12.0)
    sid = np.full((H, W), 2, dtype=np.int32)
    nrm = np.zeros((H, W, 3))
    nrm[...] = (0.0, 0.0, -1.0)
    with np.errstate(divide="ignore"):
        tf = np.where(dy > 0, 1.5 / dy, np.inf)
    hit = tf < t
    t[hit], sid[hit] = tf[hit], 1
    nrm[hit] = (0.0, -1.0, 0.0)
    lo, hi = np.array([-1.6, 0.6, 3.0]), np.array([-0.4, 1.5, 4.5])  # box left of centre: its +x face is visible
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = lo / r, hi / r
    tn, tx = np.minimum(t1, t2), np.maximum(t1, t2)
    tn, tx = np.where(np.isnan(tn), -np.inf, tn), np.where(np.isnan(tx), np.inf, tx)
    t_in, t_out = tn.max(-1), tx.min(-1)
    face = tn.argmax(-1)
    hit = (t_out >= t_in) & (t_in > 0) & (t_in < t)
    t[hit], sid[hit] = t_in[hit], 3 + face[hit]
    fn = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # +x face (x = -0.4 faces the camera at x = 0), top, front
    nrm[hit] = fn[face[hit]]
    names = {"floor": 1, "wall": 2, "box_x": 3, "box_top": 4, "box_front": 5}
    xs = np.linspace(0.6, 2.6, len(radii))
    for i, (R, xc) in enumerate(zip(radii, xs)):
        zc = 5.0 + 1.5 * i
        a = dx**2 + 1.0
        b = -2 * (xc * dx + zc)
        c = xc**2 + zc**2 - R**2
        disc = b * b - 4 * a * c
        with np.errstate(invalid="ignore"):
            tc = (-b - np.sqrt(disc)) / (2 * a)
        Y = tc * dy
        hit = (disc > 0) & (tc > 0) & (tc < t) & (Y > -1.0) & (Y < 1.5)
        t[hit], sid[hit] = tc[hit], 10 + i
        X, Zc = tc * dx, tc
        nn = np.stack([X - xc, np.zeros_like(X), Zc - zc], -1) / R
        nrm[hit] = nn[hit]
        names[f"cyl_{R:g}"] = 10 + i
    return t, sid, nrm, names


def chamfer_scene(width_px, half_angle_deg=45.0, Z0=4.0, W=504, H=378, f=F_PX):
    """Convex prism with vertical edges: plane A (x < -a, normal tilted toward -x), a narrow face B (z = Z0,
    |x| < a, normal (0, 0, -1)) and plane C (x > a, tilted toward +x); B is width_px wide in the image.
    Returns D, sid (1 A, 2 B, 3 C), normals {1, 2, 3}, and the image columns of the A|B and B|C folds."""
    r = rays(W, H, f)
    a = width_px * Z0 / (2 * f)
    h = np.radians(half_angle_deg)
    ns = {1: np.array([-np.sin(h), 0, -np.cos(h)]), 2: np.array([0.0, 0, -1]), 3: np.array([np.sin(h), 0, -np.cos(h)])}
    pts = {1: np.array([-a, 0, Z0]), 2: np.array([0, 0, Z0]), 3: np.array([a, 0, Z0])}
    ts = np.stack([(ns[s] @ pts[s]) / (r @ ns[s]) for s in (1, 2, 3)], -1)
    ts = np.where(ts > 0, ts, -np.inf)
    sid = ts.argmax(-1) + 1
    D = ts.max(-1)
    c = (W - 1) / 2
    return D, sid, ns, (c - a * f / Z0, c + a * f / Z0)


def step_scene(ratio, tilt_near_deg, tilt_far_deg, phi_deg, Z1=3.0, W=504, H=378, f=F_PX):
    """Occlusion edge: a near plane through (0, 0, Z1) on the side q.(x - c) < 0 of an image line through the
    principal point at angle phi, and a far plane through (0, 0, ratio Z1) elsewhere. Each plane is tilted about the
    image-line direction by its own angle, so the normals differ and fold candidates exist on the edge.
    Returns D, sid (1 near, 2 far), normals {1, 2}."""
    r = rays(W, H, f)
    ph = np.radians(phi_deg)
    q = np.array([-np.sin(ph), np.cos(ph), 0.0])
    n0 = np.array([0.0, 0.0, -1.0])
    ns = {1: np.cos(np.radians(tilt_near_deg)) * n0 + np.sin(np.radians(tilt_near_deg)) * q,
          2: np.cos(np.radians(tilt_far_deg)) * n0 + np.sin(np.radians(tilt_far_deg)) * q}
    P = {1: np.array([0.0, 0.0, Z1]), 2: np.array([0.0, 0.0, ratio * Z1])}
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    side = (u - (W - 1) / 2) * q[0] + (v - (H - 1) / 2) * q[1]
    sid = np.where(side < 0, 1, 2)
    t = {s: (ns[s] @ P[s]) / (r @ ns[s]) for s in (1, 2)}
    D = np.where(sid == 1, t[1], t[2])
    return D, sid, ns


CANON = 300.0  # canonical focal: D = Z * CANON / f (Stage A convention), so Z = f D / CANON


def affine_step_scene(rho_far, rho_near, line_pt, line_dir, W=504, H=378):
    """Disparity step between two affine fields rho = a + b u + c v (coefficients (a, b, c)): rho_near on the side
    q.(x - line_pt) < 0 of the image line through line_pt with direction line_dir, q = line_dir rotated by +90 deg,
    rho_far elsewhere (any affine rho is a plane). Returns (D = 1/rho (canonical units), sid: 1 near, 2 far)."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    d = np.asarray(line_dir, float) / np.linalg.norm(line_dir)
    q = np.array([-d[1], d[0]])
    side = (u - line_pt[0]) * q[0] + (v - line_pt[1]) * q[1]
    sid = np.where(side < 0, 1, 2)
    rn = rho_near[0] + rho_near[1] * u + rho_near[2] * v
    rf = rho_far[0] + rho_far[1] * u + rho_far[2] * v
    with np.errstate(divide="ignore"):
        D = 1.0 / np.where(sid == 1, rn, rf)
    return D, sid


def pole_scene(width_px, Z_pole=3.0, Z_bg=6.0, bg_tilt_deg=20.0, W=504, H=378, f=F_PX):
    """A fronto-parallel vertical pole (a slab at Z_pole, width_px wide in the image, centred) in front of a background
    plane through (0, 0, Z_bg) tilted about the vertical axis by bg_tilt_deg. Returns (Z (metric), sid: 1 pole,
    2 background, (u_left, u_right) the pole's image edges)."""
    r = rays(W, H, f)
    th = np.radians(bg_tilt_deg)
    n = np.array([np.sin(th), 0.0, -np.cos(th)])
    Zb = (n @ np.array([0.0, 0.0, Z_bg])) / (r @ n)
    c = (W - 1) / 2
    u = np.arange(W, dtype=np.float64)[None, :].repeat(H, 0)
    ul, ur = c - width_px / 2, c + width_px / 2
    sid = np.where((u >= ul) & (u <= ur), 1, 2)
    Z = np.where(sid == 1, Z_pole, Zb)
    return Z, sid, (ul, ur)
