"""TASK_GEOM_V2 V5: geometry.laplacian.role fallback (default channel) and illum_baseline.geom.fallback."""

import json

import numpy as np
import pandas as pd
import pytest

from illum_baseline import PROJECT_ROOT
from illum_baseline.config import Config, apply_sets, load_config
from illum_baseline.geom.fallback import apply_fallback, centrelines, exclusion_zone

STAGE_A = PROJECT_ROOT / "baseline_output" / "stage_a"


def test_centrelines_and_zone():
    depth = pd.DataFrame({"u": [10, 20, 30, 40, 70], "v": [5, 5, 5, 5, 5], "accepted": [True, True, True, False, True],
                          "thin": [False, True, True, False, False], "sky_edge": [False, False, True, False, False],
                          "w_c": [0.1, 0.1, 0.05, 0.1, 0.8]})
    fold = pd.DataFrame({"u": [50, 60, 65], "v": [8, 8, 8], "u_site": [51, 60, 66], "v_site": [8, 9, 8],
                         "accepted": [True, False, True], "w_c": [0.3, 0.3, 0.6]})
    c = centrelines(depth, fold, (20, 80))
    # depth: 10 (not thin), 30 (thin but a sky edge); 20 thin, 40 not accepted, 70 weak (w_c >= 0.5) -> out;
    # fold at its site (51, 8); the weak fold at 66 -> out
    assert set(zip(*np.nonzero(c))) == {(5, 10), (5, 30), (8, 51)}
    assert centrelines(depth, fold, (20, 80), include_thin=True)[5, 20]
    w = centrelines(depth, fold, (20, 80), include_weak=True)
    assert w[5, 70] and w[8, 66]
    z = exclusion_zone(depth, fold, (20, 80), 2.0)
    assert z[5, 12] and not z[5, 13] and z[6, 11] and not z[7, 11] and not z[5, 20]
    w = np.full((20, 80), 0.3)
    assert (apply_fallback(w, z)[z] == 1).all() and (apply_fallback(w, z)[~z] == 0.3).all()


def test_role_config():
    assert Config().geometry.laplacian.role == "channel"
    for bad in (["geometry.laplacian.role=mask"], ["geometry.laplacian.d_excl=-1"]):
        with pytest.raises(ValueError):
            apply_sets(Config(), bad)


def test_fallback_in_stage_b(tmp_path):
    from illum_baseline.stage_b import load_upright, process_one

    stem = "taiwei_cui_079"
    if not (STAGE_A / f"{stem}.npz").exists():
        pytest.skip("no Stage A cache")
    cfg = apply_sets(load_config(PROJECT_ROOT / "configs" / "candidate_geom_v1.yaml"),
                     ["geometry.fold.site=hessian_peak", "geometry.laplacian.role=fallback", "geometry.laplacian.d_excl=3"])
    meta = process_one(cfg, stem, tmp_path)
    a, _ = load_upright(cfg, tmp_path, stem)
    zone, wfb = a["laplacian_exclusion"], a["laplacian_w_fallback"].astype(np.float64)
    assert zone.dtype == bool and zone.any()
    assert (wfb[zone] == 1).all() and np.array_equal(wfb[~zone], a["laplacian_w_net"].astype(np.float64)[~zone])
    comb = np.minimum.reduce([a["depth_w_net"], a["fold_w_net"], a["laplacian_w_fallback"]])
    assert np.array_equal(comb, a["w_net"])                             # the fusion is the unchanged min
    dc = pd.read_parquet(tmp_path / f"{stem}_depth_candidates.parquet")
    fc = pd.read_parquet(tmp_path / f"{stem}_fold_candidates.parquet")
    assert np.array_equal(exclusion_zone(dc, fc, zone.shape, 3), zone)  # the sweep's function, same zone
    assert meta["params"]["geometry"]["laplacian"] == {"pool_radius": 2, "role": "fallback", "d_excl": 3}
    lap = meta["channels"]["laplacian"]
    assert lap["role"] == "fallback" and lap["exclusion_share"] == pytest.approx(float(zone.mean()))
    ref = json.loads((PROJECT_ROOT / "baseline_output" / "stage_b" / "geom_v1" / f"{stem}.json").read_text())
    assert ref["params"]["geometry"]["laplacian"] == {"pool_radius": 2}  # the default records stay unchanged
