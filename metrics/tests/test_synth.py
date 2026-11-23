"""§7 synthetic suite (+ Delta Q1 textured material). Scene and pairs: tests/synth.py (seed 0, fixed a priori).

The texture tolerance is statistical: tools/texture_mc.py measures the per-pair false-fail rate of ground truth over
seeds (reported at Checkpoint 2). The seed-0 scene below is a fixed regression case, not a coverage guarantee.
"""

import math

import numpy as np
import pytest

import synth as S
from core import baselines as BL
from core import pair_metrics as PM
from core import profiles as PF
from core import summary as SM

TAU = 0.10


@pytest.fixture(scope="module")
def world(cfg):
    sc = S.scene(0)
    img = S.image_dict(cfg, sc)
    return sc, img, S.pairs()


def score(cfg, img, lR, P):
    out = []
    for p in P:
        rec, prof = PM.measure(p, S.ctx(img, lR), cfg)
        out.append((p, rec, PM.judge(rec, TAU, cfg), prof))
    return out


def tagged(res, *tags):
    return [x for x in res if x[0]["tag"] in tags]


# ---------------------------------------------------------------- ground truth
def test_gt_every_scored_pair_passes(cfg, world):
    sc, img, P = world
    res = score(cfg, img, sc["lR_gt"], P)
    scored = [x for x in res if x[1]["E_kind"] is not None]
    assert len(scored) == len(P) - 3 - 2               # minus 3 b pairs and 2 unscored d pairs
    bad = [(p["tag"], p["id"]) for p, rec, j, _ in scored if not j["pass_wide"]]
    assert bad == []
    tex_plain = [p["id"] for p, rec, j, _ in tagged(res, "i_textured") if not j["pass_plain"]]
    print(f"\nGT textured i pairs failing under plain tau (seed 0): {len(tex_plain)}/12 {tex_plain}")
    assert not any(j["low_snr"] for _, _, j, _ in res)


def test_gt_split_fractions(cfg, world):
    sc, img, P = world
    res = score(cfg, img, sc["lR_gt"], P)
    for p, rec, j, _ in tagged(res, "i_smooth"):
        assert abs(rec["alpha"] - 1) < 1e-9                # smooth: ΔℓR = 0 exactly
    for p, rec, j, _ in tagged(res, "i_textured"):
        assert abs(rec["alpha"] - 1) < 0.1
    for p, rec, j, _ in tagged(res, "m_tex_side", "m_smooth"):
        assert j["alpha_defined"] and abs(rec["alpha"]) < 0.1
    for p, rec, j, _ in tagged(res, "m_chroma"):
        assert not j["alpha_defined"]                      # ||ΔℓI|| = 0.173 < 2 tau
    for p, rec, j, _ in tagged(res, "b"):
        assert j["both_step"]


# ---------------------------------------------------------------- degenerate baselines: exact bins
def test_all_R_bins(cfg, world):
    sc, img, P = world
    res = score(cfg, img, BL.all_R(img["lI"]), P)
    for p, rec, j, _ in res:
        t = p["tag"]
        if rec["kind"] == "i":
            assert rec["alpha"] == 0.0 and not j["pass_wide"] and j["bin_wide"] == "full_leak", t
        elif rec["kind"] == "m":
            assert j["pass_plain"] and j["pass_wide"], t
        elif rec["kind"].startswith("s_"):
            assert j["pass_wide"], t                       # H_R = H_I <= max(tau, H_I); E = ||ΔℓI|| small
        elif t == "d_R_diff":
            assert not j["pass_wide"], t
        elif t in ("d_R_same", "d_L"):
            assert j["pass_wide"], t
        elif t == "b":
            assert j["both_step"] is False


def test_all_L_bins(cfg, world):
    sc, img, P = world
    res = score(cfg, img, BL.all_L(img["lI"]), P)
    for p, rec, j, _ in res:
        t = p["tag"]
        if rec["kind"] == "i" or rec["kind"].startswith("s_") or rec["kind"] == "d_R":
            assert j["pass_plain"] and j["pass_wide"], t
        elif t in ("m_tex_side", "m_smooth"):
            assert rec["alpha"] == 1.0 and not j["pass_wide"] and j["bin_wide"] == "full_leak", t
        elif t == "m_chroma":
            assert not j["pass_wide"] and j["bin_wide"] == "n/a", t
        elif t == "d_L":
            assert not j["pass_wide"], t
        elif t == "b":
            assert j["both_step"] is False


def test_blur_L_exercises_profile(cfg, world):
    sc, img, P = world
    res = score(cfg, img, BL.blur_L(img["lI"], cfg), P)
    hr = [rec["H_R"] for p, rec, j, _ in tagged(res, "i_smooth")]
    assert all(h > TAU for h in hr)


# ---------------------------------------------------------------- injected failures
def test_gap_leak_targeted_s(cfg, world):
    sc, img, P = world
    lR = S.gap_leak(sc["lR_gt"], sc["shadow"])
    res = sorted(tagged(score(cfg, img, lR, P), "s_gap"), key=lambda x: x[0]["gap_dist"])
    E = [rec["E"] for _, rec, _, _ in res]
    d = [p["gap_dist"] for p, _, _, _ in res]
    print("\ngap leak: d / E / pass:", [(dd, round(e, 3), j["pass_wide"]) for dd, e, (_, _, j, _) in zip(d, E, res)])
    assert all(a > b for a, b in zip(E, E[1:]))            # E decays with distance from the gap
    assert not res[0][2]["pass_wide"] and not res[1][2]["pass_wide"]
    assert res[-1][2]["pass_wide"]
    gt = {p["id"]: j["pass_wide"] for p, _, j, _ in tagged(score(cfg, img, sc["lR_gt"], P), "s_gap")}
    assert all(gt.values())


def test_halo(cfg, world):
    sc, img, P = world
    res = score(cfg, img, S.halo(img["lI"], sc["lR_gt"]), P)
    for p, rec, j, _ in tagged(res, "i_smooth"):
        assert rec["H_R"] > TAU and rec["E"] < 0.02 and not j["pass_wide"], (rec["H_R"], rec["E"])


# ---------------------------------------------------------------- invariances and identities
NUM = ("dI", "dR", "dL", "nI", "nR", "nL", "alpha", "H_I", "H_R", "H_L", "E", "H", "se_tex", "se_noise")


@pytest.mark.parametrize("pred", ["gt", "blur_L"])
@pytest.mark.parametrize("c", [3.7, (1.1, -0.4, 2.5)])
def test_scale_invariance(cfg, world, pred, c):
    sc, img, P = world
    lR = sc["lR_gt"] if pred == "gt" else BL.blur_L(img["lI"], cfg)
    a = score(cfg, img, lR, P)
    b = score(cfg, img, lR + np.asarray(c, float), P)
    for (p, r1, j1, _), (_, r2, j2, _) in zip(a, b):
        for k in NUM:
            x, y = np.asarray(r1[k], float), np.asarray(r2[k], float)
            assert np.allclose(x, y, atol=1e-9, rtol=0, equal_nan=True), (p["tag"], k, x, y)
        assert j1 == j2, p["tag"]


def test_identity_endpoints_and_profiles(cfg, world):
    sc, img, P = world
    for p, rec, j, prof in score(cfg, img, BL.blur_L(img["lI"], cfg), P):
        assert np.allclose(np.asarray(rec["dL"]), np.asarray(rec["dI"]) - np.asarray(rec["dR"]), atol=1e-12)
        fin = np.isfinite(prof["L"])
        assert np.array_equal(prof["L"][fin], (prof["I"] - prof["R"])[fin])


# ---------------------------------------------------------------- guards, refusals, empties
def test_noise_guard_uses_noise_model(cfg, world):
    sc, img, P = world
    noisy = dict(img, noise=[(5000.0, 1e6)] * 3)
    rec, _ = PM.measure(P[0], S.ctx(noisy, sc["lR_gt"]), cfg)
    assert PM.judge(rec, TAU, cfg)["low_snr"]
    rec0, _ = PM.measure(P[0], S.ctx(img, sc["lR_gt"]), cfg)
    assert not PM.judge(rec0, TAU, cfg)["low_snr"] and rec0["se_tex"] == rec["se_tex"]


def test_used_in_loss_recorded(cfg, world):
    sc, img, P = world
    rec, _ = PM.measure(dict(P[0], used_in_loss=True), S.ctx(img, sc["lR_gt"]), cfg)
    assert rec["used_in_loss"] is True
    rec, _ = PM.measure(P[0], S.ctx(img, sc["lR_gt"]), cfg)
    assert rec["used_in_loss"] is False


def test_empty_types_are_na(cfg, world):
    sc, img, P = world
    only_im = [p for p in P if p["type"] in ("i", "m")]
    res = PM.score_pairs(only_im, S.ctx(img, sc["lR_gt"]), cfg)
    s = SM.by_kind(res, cfg["thresholds"]["tau"])
    assert s["s_random"] == dict(n=0, k=0, rate=None, text="n/a: 0 pairs")
    assert s["d_R"]["text"] == "n/a: 0 pairs"
    assert s["i"]["n"] > 0 and s["i"]["rate"] == 0.0


def test_excursion_basics():
    t = np.arange(21.0)
    a, b = np.zeros(3), np.array([1.0, 0.5, 0.2])
    step = np.where(t[:, None] < 10, a, b)
    assert PF.excursion(PF.smooth(t, step, 1.5), a, b) < 1e-12              # monotone step on the line: H = 0
    bump = np.zeros((21, 3))
    bump[10] = 0.3
    assert math.isclose(PF.excursion(bump, a, a), 0.3, rel_tol=1e-12)       # flat endpoints: H = bump norm
