"""TASK_GEOM_V2 W1: geometry.fold.window_cap (off default, next_ridge) and two_sided_fit's per-side window lengths."""

import dataclasses
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from illum_baseline import PROJECT_ROOT
from illum_baseline.config import Config, apply_sets, load_config
from illum_baseline.geom.fold import FoldChannel, FoldParams, params_record
from illum_baseline.geom.twosided import two_sided_fit

sys.path.insert(0, str(PROJECT_ROOT / "tests"))
from test_geom_g2g3 import _real_ctx  # noqa: E402

STAGE_A = PROJECT_ROOT / "baseline_output" / "stage_a"


def _ridges(rows, u=range(20, 60)):
    ij = np.array([[x, r] for r in rows for x in u], dtype=float)
    return SimpleNamespace(ij=ij)


def test_window_cap_rule():
    p = FoldParams(window_cap="next_ridge")
    # a narrow face: ridges at v = 20 and v = 34 (14 px); a lap pair 10 px apart (v = 60, 70); a wide face (v = 100, 120)
    cands = _ridges([20, 34, 60, 70, 100, 120])
    x0 = np.array([[40.0, 20.2], [40.0, 33.8], [40.0, 60.0], [40.0, 100.0]])
    e1 = np.tile([0.0, 1.0], (4, 1))
    k1 = np.array([8, 6, 6, 8])
    c = FoldChannel._window_cap(p, cands, x0, e1, k1)
    # 1: + side ridge at 13.8 px (first sample s within 0.75 px of a ridge pixel: 13.5) -> reach 16 >= 11.5:
    #    k 6, m = min(8, floor(11.5) - 6) = 5; - side: no ridge -> default
    assert c["d_ridge_p"][0] == 13.5 and np.isinf(c["d_ridge_m"][0])
    assert (c["k1_p"][0], c["m_p_fit"][0], c["window_capped_p"][0]) == (6, 5, True)
    assert (c["k1_m"][0], c["m_m_fit"][0], c["window_capped_m"][0]) == (8, 8, False)
    # 2: - side ridge at 13.5 (v = 20.2 is 13.6 away; samples 0.5 apart), k1 6, reach 14 -> m = 5
    assert (c["k1_m"][1], c["m_m_fit"][1], c["window_capped_m"][1]) == (6, 5, True)
    # 3: lap pair 10 px: reach 14 >= 8 but m = 8 - 6 = 2 < m_min -> default window, counted as too short
    assert not c["window_capped_p"][2] and c["k1_p"][2] == 6 and c["m_p_fit"][2] == 8
    # 4: next ridge 20 px away: beyond the search (reach 16 + margin 2) -> untouched
    assert not c["window_capped_p"][3] and c["k1_p"][3] == 8 and np.isinf(c["d_ridge_p"][3])
    assert c["n_too_short"] >= 1


def test_window_cap_config_and_record():
    assert Config().geometry.fold.window_cap == "off"
    assert apply_sets(Config(), ["geometry.fold.window_cap=next_ridge"]).geometry.fold.window_cap == "next_ridge"
    for bad in (["geometry.fold.window_cap=always"], ["geometry.fold.window_cap_margin=-1"]):
        with pytest.raises(ValueError):
            apply_sets(Config(), bad)
    rec = params_record(dataclasses.asdict(FoldParams()))
    assert not any(k.startswith("window_cap") for k in rec)
    assert params_record(dataclasses.asdict(FoldParams(window_cap="next_ridge")))["window_cap"] == "next_ridge"
    v11 = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v1_1.yaml")
    wc = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v1_1_wcap.yaml")
    a, b = v11.to_dict(), wc.to_dict()
    b["geometry"]["fold"]["window_cap"], b["run"]["name"] = "off", "geom_v1_1"
    assert a == b


def test_per_side_window_lengths_match_the_scalar_path():
    stem = "taiwei_cui_076"
    if not (STAGE_A / f"{stem}.npz").exists():
        pytest.skip("Stage A cache not present")
    cfg = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v1_1.yaml")
    ctx = _real_ctx(stem, cfg)
    rng = np.random.default_rng(0)
    H, W = ctx.shape
    x0 = np.stack([rng.uniform(30, W - 30, 200), rng.uniform(30, H - 30, 200)], -1)
    ang = rng.uniform(0, np.pi, 200)
    e1 = np.stack([np.cos(ang), np.sin(ang)], -1)
    kw = dict(w_t=4, m_min=4, max_excluded=0.2, fallback=True)
    a = two_sided_fit(ctx, x0, e1, 6, 8, **kw)
    b = two_sided_fit(ctx, x0, e1, 6, 8, m_p=np.full(200, 8), m_m=np.full(200, 8), **kw)
    for f in ("dtheta", "T", "valid", "d_rho"):
        assert np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True)
    # mixed lengths: rows with m = 4 equal a separate m = 4 fit
    m = np.where(np.arange(200) % 2 == 0, 8, 4)
    c = two_sided_fit(ctx, x0, e1, 6, 8, m_p=m, m_m=m, **kw)
    d4 = two_sided_fit(ctx, x0[m == 4], e1[m == 4], 6, 4, **kw)
    assert np.array_equal(c.dtheta[m == 4], d4.dtheta, equal_nan=True)
    assert np.array_equal(c.dtheta[m == 8], a.dtheta[m == 8], equal_nan=True)
