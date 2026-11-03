"""TASK_GEOM_INTEGRATION G4: the stratified pps estimator (scripts/geom_g4.py) and the re-evaluation config."""

import sys

import numpy as np
import pandas as pd

from illum_baseline import PROJECT_ROOT
from illum_baseline.config import load_config

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from geom_g4 import Estimator  # noqa: E402


def test_stratified_share_and_bootstrap():
    # two strata: A = 300 (4 draws, 1 labelled c) and A = 100 (2 draws, both c); one stratum with a single draw
    strata = pd.DataFrame({"stem": ["a", "b", "c"], "A_h": [300, 100, 50], "n_h": [4, 2, 1]})
    draws = pd.DataFrame({"stem": ["a", "a", "a", "a", "b", "b", "c"]})
    ind = {"c": np.array([1, 0, 0, 0, 1, 1, 0], bool), "all": np.ones(7, bool)}
    E = Estimator(draws, strata, total_px=450.0, image_px=4500.0, rng=np.random.default_rng(0))
    df, reps = E.run(ind)
    row = df.set_index("quantity").loc["c"]
    want = (300 * 0.25 + 100 * 1.0 + 50 * 0.0) / 450
    assert np.isclose(row["share"], want) and np.isclose(row["pooled_fraction"], 3 / 7)
    assert np.isclose(row["area_px"], want * 450) and np.isclose(row["share_of_image_px"], want * 450 / 4500)
    assert row["ci_lo"] <= want <= row["ci_hi"] and row["ci_lo"] < row["ci_hi"]
    assert np.allclose(reps["all"], 1.0) and E.n_single == 1          # shares of the whole are exactly 1
    # the single-draw stratum contributes no bootstrap variance
    ind2 = {"x": np.array([0, 0, 0, 0, 0, 0, 1], bool)}
    _, r2 = E.run(ind2)
    assert np.allclose(r2["x"], 50 / 450)


def test_reeval_config_is_geom_v0_plus_the_approved_options():
    a = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v0.yaml")
    b = load_config(PROJECT_ROOT / "configs" / "candidate_geom_g4_reeval.yaml")
    assert b.run.name == "geom_g4_reeval"
    assert (b.geometry.depth.sky_band, a.geometry.depth.sky_band) == ("depth_and_mask", "depth")
    f = b.geometry.fold
    assert (f.crease_rescue, f.rescue_theta_deg, f.rescue_k_L) == ("laplacian_corroborated", 8.0, 3.0)
    import dataclasses
    da, db = dataclasses.asdict(a.geometry), dataclasses.asdict(b.geometry)
    db["depth"]["sky_band"] = "depth"
    db["fold"]["crease_rescue"] = "off"
    assert da == db and a.detector == b.detector and a.sky == b.sky
