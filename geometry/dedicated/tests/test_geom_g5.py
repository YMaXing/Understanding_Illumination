"""TASK_GEOM_INTEGRATION G5: geometry.laplacian.pool_radius, the geom_v1 config and the combined sheet.

geom_v1 itself is a bit-for-bit reference in tests/test_baseline_repro.py.
"""

import dataclasses
import json

import numpy as np
import pytest

from illum_baseline import PROJECT_ROOT
from illum_baseline.config import Config, apply_sets, load_config
from illum_baseline.detector import Params, compute_weights

STAGE_A = PROJECT_ROOT / "baseline_output" / "stage_a"
V1 = PROJECT_ROOT / "baseline_output" / "stage_b" / "geom_v1"


def _scene():
    rng = np.random.default_rng(0)
    H, W = 80, 90
    vv, uu = np.mgrid[0:H, 0:W]
    D = np.where(uu < 45, 4.0, 6.0) + 0.002 * rng.standard_normal((H, W))
    return D


def test_pool_radius_default_is_the_detector_r():
    D = _scene()
    p = Params()
    a = compute_weights(D, None, p)
    b = compute_weights(D, None, dataclasses.replace(p, pool_radius=p.r))
    assert p.r == 3 and np.array_equal(a["w"], b["w"]) and np.array_equal(a["q_pool"], b["q_pool"])
    assert "pool_radius" not in a["scalars"] and b["scalars"]["pool_radius"] == 3
    c = compute_weights(D, None, dataclasses.replace(p, pool_radius=1))
    assert np.array_equal(a["q"], c["q"]) and a["scalars"]["s_hat"] == c["scalars"]["s_hat"]  # only pooling changes
    assert ((c["w"] < 1).sum() < (a["w"] < 1).sum()) and (c["w"] >= a["w"]).all()


def test_pool_radius_config():
    assert Config().geometry.laplacian.pool_radius is None
    assert apply_sets(Config(), ["geometry.laplacian.pool_radius=2"]).geometry.laplacian.pool_radius == 2
    for bad in ("0", "-1"):
        with pytest.raises(ValueError):
            apply_sets(Config(), [f"geometry.laplacian.pool_radius={bad}"])


def test_pool_radius_recorded_only_when_set(tmp_path):
    from illum_baseline.stage_b import process_one

    stem = "taiwei_cui_079"
    if not (STAGE_A / f"{stem}.npz").exists():
        pytest.skip("no Stage A cache")
    cfg = load_config(PROJECT_ROOT / "baseline_output" / "stage_b" / "tri_veto4" / "config_resolved.yaml")
    cfg.geometry.enabled, cfg.geometry.channels = True, ["laplacian"]
    cfg.geometry.laplacian.pool_radius = 1
    meta = process_one(cfg, stem, tmp_path)
    assert meta["detector"]["pool_radius"] == 1 and meta["params"]["geometry"]["laplacian"] == {"pool_radius": 1}
    ref = json.loads((PROJECT_ROOT / "baseline_output" / "stage_b" / "tri_veto4" / f"{stem}.json").read_text())
    assert "pool_radius" not in ref["detector"]
    w_ref = np.load(PROJECT_ROOT / "baseline_output" / "stage_b" / "tri_veto4" / f"{stem}.npz")["w_net"]
    w1 = np.load(tmp_path / f"{stem}.npz")["w_net"]
    assert (w1 >= w_ref).all() and (w1 > w_ref).any()


def test_geom_v1_config():
    v0 = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v0.yaml")
    v1 = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v1.yaml")
    assert v1.run.name == "geom_v1" and v1.geometry.channels == ["depth", "fold", "laplacian"]
    assert v1.geometry.combine == "min" and v1.geometry.laplacian.pool_radius == 2
    assert v1.geometry.depth.sky_band == "depth_and_mask" and v1.geometry.depth.band_min == 2
    f = v1.geometry.fold
    assert (f.crease_rescue, f.rescue_theta_deg, f.rescue_k_L, f.band_min) == ("laplacian_corroborated", 8.0, 3.0, 2)
    a, b = dataclasses.asdict(v0.geometry), dataclasses.asdict(v1.geometry)
    b["channels"], b["depth"]["sky_band"], b["fold"]["crease_rescue"] = ["depth", "fold"], "depth", "off"
    b["laplacian"]["pool_radius"] = None
    assert a == b and v0.detector == v1.detector and v0.sky == v1.sky
    r3 = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v1_r3.yaml")
    assert r3.run.name == "geom_v1_r3" and r3.geometry.laplacian.pool_radius is None
    c, d = dataclasses.asdict(v1.geometry), dataclasses.asdict(r3.geometry)
    d["laplacian"]["pool_radius"] = 2
    assert c == d and r3.detector == v1.detector


def test_geom_sheet_and_disagreement():
    from illum_baseline.viz_geom import disagreement, render_geom, wants_geom_sheet

    assert wants_geom_sheet(["depth", "fold", "laplacian"]) and not wants_geom_sheet(["depth", "fold"])
    wl = np.array([0.1, 0.1, 0.95, 0.6, 0.3])
    wdf = np.array([0.95, 0.2, 0.1, 0.95, 0.89])
    lo, do = disagreement(wl, wdf)
    assert lo.tolist() == [True, False, False, False, False] and do.tolist() == [False, False, True, False, False]
    if not (V1 / "taiwei_cui_079.npz").exists():
        pytest.skip("geom_v1 run not present")
    cfg = load_config(V1 / "config_resolved.yaml")
    out = render_geom(cfg, V1, "taiwei_cui_079")
    assert out.exists() and out.stat().st_size > 10_000


def test_geom_v1_1_config_is_geom_v1_plus_the_hessian_site():
    v1 = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v1.yaml")
    v11 = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v1_1.yaml")
    assert v11.run.name == "geom_v1_1" and v11.geometry.fold.site == "hessian_peak"
    a, b = v1.to_dict(), v11.to_dict()
    b["geometry"]["fold"]["site"], b["run"]["name"] = "nms", "geom_v1"
    assert a == b
