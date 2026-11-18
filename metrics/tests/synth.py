"""Synthetic scene for the §7 suite (plus the Delta Q1 textured material). Deterministic for a given seed.

Layout (W 640 x H 320; x = column, y = row). Soft shadow edge at y = 160 (logistic, scale 1 px): y < 160 lit,
y > 160 shadow, with a coloured (ISD-like, bluish) log ratio. Materials (linear reflectance):
  mat0 smooth   x < 320                       (0.50, 0.40, 0.30)
  mat4 chroma   x 200..259, y 40..99 in mat0   mat0 * exp(0.20, -0.20, 0.10)   (||ΔℓI|| = 0.173 < 2 tau: alpha n/a)
  mat1 textured 320 <= x < 480                 (0.25, 0.48, 0.30) * exp(texture), per-pixel iid log texture:
                                               grey N(0, 0.10) + per-channel N(0, 0.04)   (||ΔℓI|| ~0.4 to mat0 / mat2)
  mat2 smooth   x >= 480 (and x 480..559 below the edge)   (0.30, 0.35, 0.55)
  mat3 smooth   x >= 560, y >= 160             (0.60, 0.30, 0.25): its top boundary IS the shadow edge (b pairs)
Illumination: log E0 + mild shading 0.03 sin(2 pi x / W) cos(pi y / H) + log(1 - s(y) + s(y) rho),
rho = (0.25, 0.30, 0.42): the penumbra mixes lit and shadow light linearly (physical), so it curves in log space.
Sensor noise: var = g I + r2 (g 1.5, r2 16 DN^2), rounded to uint16. ℓI comes from the configured loader hook.
"""

import math

import numpy as np

from core import config as K
from core import io as IO

W, H, EDGE_Y = 640, 320, 160
E0 = 50000.0
SHADOW = np.log([0.25, 0.30, 0.42])
NOISE = [(1.5, 16.0)] * 3
GAP_X = 30
TEX_X = (320, 480)


def scene(seed=0, tex_grey=0.10, tex_chroma=0.04):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    R = np.empty((H, W, 3))
    R[:] = (0.50, 0.40, 0.30)
    m4 = (xx >= 200) & (xx < 260) & (yy >= 40) & (yy < 100)
    R[m4] = np.array([0.50, 0.40, 0.30]) * np.exp([0.20, -0.20, 0.10])
    m1 = (xx >= TEX_X[0]) & (xx < TEX_X[1])
    tex = rng.normal(0, tex_grey, (H, W, 1)) + rng.normal(0, tex_chroma, (H, W, 3))
    R[m1] = np.array([0.25, 0.48, 0.30]) * np.exp(tex[m1])
    m2 = xx >= 480
    R[m2] = (0.30, 0.35, 0.55)
    m3 = (xx >= 560) & (yy >= EDGE_Y)
    R[m3] = (0.60, 0.30, 0.25)
    lR = np.log(R)
    s = 1.0 / (1.0 + np.exp(-(yy - EDGE_Y) / 1.0))
    lL = math.log(E0) + 0.03 * np.sin(2 * np.pi * xx / W) * np.cos(np.pi * yy / H)
    lL = lL[..., None] + np.log(1.0 - s[..., None] + s[..., None] * np.exp(SHADOW))
    I = np.exp(lR + lL)
    g, r2 = NOISE[0]
    In = I + rng.normal(0, 1, I.shape) * np.sqrt(g * I + r2)
    In = np.clip(np.rint(In), 0, 65535).astype(np.uint16)
    return dict(I=In, lR_gt=lR, lL_true=lL, shadow=s)


def image_dict(cfg, sc):
    lI = np.asarray(K.import_hook(cfg["loader"]["log_i"])(sc["I"]), np.float64)
    img = dict(stem="synth", I=sc["I"], lI=lI, valid=IO.validity_mask(cfg, sc["I"]), H=H, W=W, noise=NOISE,
               geom_edge=None)
    return img


def ctx(img, lR):
    return dict(lI=img["lI"], lR=lR, valid=img["valid"], I=img["I"], noise=img["noise"], geom_edge=None,
                regions=None, set="synthetic")


_n = [0]


def _p(type_, a, b, r=2, **kw):
    _n[0] += 1
    p = dict(id=f"t={_n[0]}", image="synth", type=type_, A=dict(x=a[0], y=a[1], r=r), B=dict(x=b[0], y=b[1], r=r),
             confidence="sure", used_in_loss=False, overlay_seen=False, annotator="synth", created="2026-11-18T00:00:00Z",
             note="", schema_version="1.1")
    p.update(kw)
    return p


def pairs():
    """Programmatic pairs. Each carries a 'tag' (test grouping; not a schema field, stripped before scoring is
    unnecessary because measure() ignores unknown keys)."""
    _n[0] = 0
    P = []
    for x in (60, 100, 140, 180, 220, 280):
        P.append(dict(_p("i", (x, 135), (x, 185), lit_end="A"), tag="i_smooth"))
    for x in (500, 530):
        P.append(dict(_p("i", (x, 135), (x, 185), lit_end="A"), tag="i_smooth"))
    for k, x in enumerate(range(330, 471, 12)):
        P.append(dict(_p("i", (x, 135), (x, 185), r=1 + k % 3, lit_end="A"), tag="i_textured"))
    for y in (40, 80, 120, 200, 240, 280):
        P.append(dict(_p("m", (300, y), (340, y)), tag="m_tex_side"))
        P.append(dict(_p("m", (460, y), (500, y)), tag="m_tex_side"))
    for y in (200, 240, 280):
        P.append(dict(_p("m", (540, y), (580, y)), tag="m_smooth"))
    P.append(dict(_p("m", (185, 70), (215, 70)), tag="m_chroma"))
    P.append(dict(_p("m", (230, 85), (230, 115)), tag="m_chroma"))
    for x in (580, 600, 620):
        P.append(dict(_p("b", (x, 135), (x, 185)), tag="b"))
    for a, b, lit, tag in (((40, 30), (90, 30), "lit", "s_smooth"), ((120, 120), (170, 120), "lit", "s_smooth"),
                           ((60, 250), (110, 250), "shadow", "s_smooth"), ((150, 300), (200, 300), "shadow", "s_smooth"),
                           ((500, 60), (540, 90), "lit", "s_smooth"),
                           ((340, 60), (390, 60), "lit", "s_textured"), ((400, 250), (450, 250), "shadow", "s_textured")):
        P.append(dict(_p("s", a, b, lighting=lit, pool="random"), tag=tag))
    for d in (0, 20, 40, 80, 120, 180, 240):
        P.append(dict(_p("s", (GAP_X + d, 175), (GAP_X + d + 40, 175), lighting="shadow", pool="targeted",
                         gap_ref=f"synthetic gap at x={GAP_X}"), tag="s_gap", gap_dist=d))
    for a, b, mat, ill, tag in (((40, 60), (250, 290), "same", "different", "d_R_diff"),
                                ((500, 40), (520, 300), "same", "different", "d_R_diff"),
                                ((30, 130), (290, 40), "same", "same", "d_R_same"),
                                ((20, 300), (300, 220), "same", "same", "d_R_same"),
                                ((100, 60), (600, 60), "different", "same", "d_L"),
                                ((100, 60), (600, 250), "different", "different", "d_unscored"),
                                ((40, 60), (250, 290), "unsure", "different", "d_unscored")):
        P.append(dict(_p("d", a, b, r=3, material=mat, illumination=ill), tag=tag))
    return P


def gap_leak(lR_gt, shadow, scale=30.0):
    """ℓR with part of the shadow leaked into R near a gap in the shadow edge at x = GAP_X (decays exp(-|x - gap| / 30))."""
    xx = np.arange(W, dtype=np.float64)[None, :]
    k = np.exp(-np.abs(xx - GAP_X) / scale)
    return lR_gt + (k * shadow)[..., None] * SHADOW


def halo(lI, lR_gt, sigma=6.0):
    """L blurred across the shadow (ℓL = G_sigma * (ℓI - ℓR_gt)); returns the implied ℓR = ℓI - ℓL."""
    from scipy import ndimage as ndi
    lL = lI - lR_gt
    G = np.stack([ndi.gaussian_filter(lL[..., c], sigma, mode="reflect") for c in range(3)], -1)
    return lI - G
