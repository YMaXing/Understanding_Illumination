"""TASK_GEOM_V2 W2: geometry.laplacian.hysteresis (off default, ridge_bridge) and illum_baseline.geom.hysteresis."""

import json

import numpy as np
import pytest

from illum_baseline import PROJECT_ROOT
from illum_baseline.config import Config, apply_sets, load_config
from illum_baseline.geom.hysteresis import ridge_hysteresis

STAGE_A = PROJECT_ROOT / "baseline_output" / "stage_a"


def _ridge(amp_gap, gap=(50, 60), W=120):
    H = 60
    vv, uu = np.mgrid[0:H, 0:W].astype(float)
    amp = np.where((uu > gap[0]) & (uu < gap[1]), amp_gap, 1.5)
    return amp * np.exp(-0.5 * ((vv - 30) / 1.5) ** 2)


def test_bridge_rule():
    # a straight ridge at 1.5 tau with a 9-px dip to 0.7 tau
    h = ridge_hysteresis(_ridge(0.7), 1.0, 0.6)
    assert h["bridged"].sum() == 9 and set(np.nonzero(h["bridged"])[0]) == {30}       # the gap row only
    assert (h["q_cont"][h["bridged"]] <= 0.5 + 1e-12).all()                          # <= the flanks' q - tau
    assert not ridge_hysteresis(_ridge(0.7), 1.0, 0.8)["bridged"].any()             # 0.7 tau < 0.8 tau: no run
    # a dip longer than max_gap is not bridged; one-sided runs only with single_ended
    assert not ridge_hysteresis(_ridge(0.7, (40, 70)), 1.0, 0.6)["bridged"].any()
    end = _ridge(0.7, (106, 200))                                                    # 13-px dip to the image border
    assert not ridge_hysteresis(end, 1.0, 0.6)["bridged"].any()
    assert ridge_hysteresis(end, 1.0, 0.6, single_ended=True)["bridged"].any()


def test_hysteresis_config():
    assert Config().geometry.laplacian.hysteresis == "off"
    c = apply_sets(Config(), ["geometry.laplacian.hysteresis=ridge_bridge", "geometry.laplacian.hysteresis_kappa=0.6"])
    assert c.geometry.laplacian.hysteresis_kappa == 0.6
    for bad in (["geometry.laplacian.hysteresis=canny"], ["geometry.laplacian.hysteresis_kappa=1.2"]):
        with pytest.raises(ValueError):
            apply_sets(Config(), bad)
    v11 = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v1_1.yaml").to_dict()
    for tag, k in (("06", 0.6), ("08", 0.8)):
        h = load_config(PROJECT_ROOT / "configs" / f"candidate_geom_v1_1_hys{tag}.yaml").to_dict()
        assert h["geometry"]["laplacian"]["hysteresis_kappa"] == k
        h["geometry"]["laplacian"]["hysteresis"], h["run"]["name"] = "off", "geom_v1_1"
        h["geometry"]["laplacian"]["hysteresis_kappa"] = 0.8
        assert h == v11


def test_hysteresis_in_stage_b(tmp_path):
    from illum_baseline.stage_b import load_upright, process_one

    stem = "taiwei_cui_079"
    if not (STAGE_A / f"{stem}.npz").exists():
        pytest.skip("no Stage A cache")
    ref_dir = PROJECT_ROOT / "baseline_output" / "stage_b" / "geom_v1_1"
    cfg0 = load_config(ref_dir / "config_resolved.yaml")
    cfg = apply_sets(cfg0, ["geometry.laplacian.hysteresis=ridge_bridge", "geometry.laplacian.hysteresis_kappa=0.6"])
    meta = process_one(cfg, stem, tmp_path)
    a, _ = load_upright(cfg, tmp_path, stem)
    b, _ = load_upright(cfg0, ref_dir, stem)
    wl, wl0 = a["laplacian_w_net"].astype(np.float64), b["laplacian_w_net"].astype(np.float64)
    assert (wl <= wl0 + 1e-6).all()                                    # w only goes down
    assert np.array_equal(a["depth_w_net"], b["depth_w_net"]) and np.array_equal(a["fold_w_net"], b["fold_w_net"])
    assert a["laplacian_continued"].dtype == bool
    assert meta["params"]["geometry"]["laplacian"] == {"pool_radius": 2, "hysteresis": "ridge_bridge",
                                                       "hysteresis_kappa": 0.6, "hysteresis_max_gap": 16}
    ref = json.loads((ref_dir / f"{stem}.json").read_text())
    assert ref["params"]["geometry"]["laplacian"] == {"pool_radius": 2}   # the default records stay unchanged
