"""TASK_FOLD_CHANNEL F3: channel framework and Stage B integration.

- combine is the elementwise min (and only min);
- geometry.enabled with channels [laplacian] is bit-identical to tri_veto4 (w and every detector array);
- the saved keys and the candidate parquet schema (docs/geometry_channels.md section 3.5);
- config: built-in default off, validation, nested --set, config <-> params field parity.
(baseline_v0 and tri_veto4 bit-for-bit: tests/test_baseline_repro.py.)
"""

import dataclasses
import json

import numpy as np
import pytest

from illum_baseline import DEFAULT_CONFIG, PROJECT_ROOT
from illum_baseline.config import Config, DenseFitCfg, FoldCfg, apply_sets, load_config
from illum_baseline.geom.channels import CHANNELS, ChannelResult, combine
from illum_baseline.geom.fold import FoldParams
from illum_baseline.geom.planefit import DenseFitParams

TRI_RUN = PROJECT_ROOT / "baseline_output" / "stage_b" / "tri_veto4"

# per-candidate columns of <stem>_fold_candidates.parquet (docs/geometry_channels.md section 3.5)
FOLD_COLUMNS = [
    "u", "v", "u0", "v0", "e1_u", "e1_v", "r_n", "width", "profile_ok", "k1", "k2",
    "alpha_p", "rms_rel_p", "n_p", "excl_p", "m_used_p", "planar_p", "fallback_p", "sigma_theta_p",
    "alpha_m", "rms_rel_m", "n_m", "excl_m", "m_used_m", "planar_m", "fallback_m", "sigma_theta_m",
    "n_p_x", "n_p_y", "n_p_z", "n_m_x", "n_m_y", "n_m_z",
    "valid", "dtheta", "T", "T_over_lambda", "T_add", "T_add_local", "sigma_geo_local", "gate_mult", "gate_add",
    "gate_add_local", "gate", "dtheta_gate", "dtheta_gate_add", "dtheta_gate_add_local", "dtheta_k2", "C", "no_veto", "veto_pass", "k2_clean", "component_px", "accepted", "reason",
    "J", "chord", "J_lo", "J_hi", "w_c", "band",
    "d_rho", "d_rho_se", "s_gap", "planes_cross", "d_rho_x0", "rho_p_gap", "rho_m_gap", "dZ_rel", "depth_jump",
    "convexity", "L_over_rho_x0",
]


def _w(a):
    return ChannelResult(name="x", w_net=np.asarray(a, float), strength=None, candidates=None)


def test_combine_is_elementwise_min():
    rng = np.random.default_rng(0)
    a, b, c = (rng.uniform(0.05, 1, (7, 5)) for _ in range(3))
    assert np.array_equal(combine([_w(a), _w(b), _w(c)]), np.minimum(np.minimum(a, b), c))
    one = _w(a)
    assert combine([one]) is one.w_net  # a single channel passes through bit for bit
    with pytest.raises(ValueError, match="only 'min'"):
        combine([_w(a), _w(b)], mode="max")
    with pytest.raises(ValueError):
        combine([])


def test_config_defaults_validation_and_nested_set():
    cfg = Config()
    assert cfg.geometry.enabled is False and cfg.geometry.channels == ["laplacian"]
    assert load_config(DEFAULT_CONFIG).geometry.enabled is False
    assert set(CHANNELS) >= {"laplacian", "fold"}
    apply_sets(cfg, ["geometry.fold.p_fa=1.0e-4", "geometry.dense_fit.refit=2", "geometry.channels=[fold]"])
    assert cfg.geometry.fold.p_fa == 1e-4 and cfg.geometry.dense_fit.refit == 2 and cfg.geometry.channels == ["fold"]
    # float fields accept YAML strings that look like floats ("1e-4" has no decimal point), and fail clearly otherwise
    c2 = apply_sets(Config(), ["detector.k_tau=1e-4", "geometry.fold.p_fa=1e-4", "geometry.fold.lambda_clamp=[1e-1, 5]"])
    assert c2.detector.k_tau == 1e-4 and c2.geometry.fold.p_fa == 1e-4 and c2.geometry.fold.lambda_clamp == [0.1, 5]
    with pytest.raises(ValueError, match="expected a number"):
        apply_sets(Config(), ["geometry.fold.p_fa=abc"])
    for bad in (["geometry.channels=[bogus]"], ["geometry.combine=max"], ["geometry.fold.null_model=x"],
                ["geometry.fold.lambda_clamp=[5, 1]"], ["geometry.fold.nope=1"], ["geometry.fold=1"]):
        with pytest.raises(ValueError):
            apply_sets(Config(), bad)


def test_config_params_field_parity():
    """Every fold / dense-fit config field maps onto the geom params dataclasses, with the same defaults."""
    assert {f.name for f in dataclasses.fields(FoldCfg)} == {f.name for f in dataclasses.fields(FoldParams)}
    assert {f.name for f in dataclasses.fields(DenseFitCfg)} == {f.name for f in dataclasses.fields(DenseFitParams)}
    assert FoldParams.from_dict(dataclasses.asdict(FoldCfg())) == FoldParams()
    assert DenseFitParams(**dataclasses.asdict(DenseFitCfg())) == DenseFitParams()
    for name in ("candidate_fold_only.yaml", "candidate_lap_fold.yaml"):
        cfg = load_config(PROJECT_ROOT / "configs" / name)
        assert cfg.geometry.enabled
        assert FoldParams.from_dict(dataclasses.asdict(cfg.geometry.fold)) == FoldParams()


def _tri_cfg():
    if not (TRI_RUN / "config_resolved.yaml").exists():
        pytest.skip(f"{TRI_RUN} not present")
    return load_config(TRI_RUN / "config_resolved.yaml")


def test_laplacian_channel_bit_identical_to_tri_veto4(tmp_path):
    from illum_baseline.stage_b import image_stems, process_one

    cfg = _tri_cfg()
    cfg.geometry.enabled = True
    cfg.geometry.channels = ["laplacian"]
    stems = image_stems(TRI_RUN)
    assert stems
    for stem in stems:
        process_one(cfg, stem, tmp_path)
        old, new = np.load(TRI_RUN / f"{stem}.npz"), np.load(tmp_path / f"{stem}.npz")
        for k in old.files:  # w, w_net, q, L_over_rho, D, sky_mask
            assert old[k].dtype == new[k].dtype and np.array_equal(old[k], new[k]), (stem, k)
        assert np.array_equal(new["laplacian_w_net"], old["w_net"])
        assert "n_hat" not in new.files  # nothing computed the normals
        assert json.loads((tmp_path / f"{stem}.json").read_text())["normals_computed"] is False
        a, b = json.loads((TRI_RUN / f"{stem}.json").read_text()), json.loads((tmp_path / f"{stem}.json").read_text())
        assert a["detector"] == b["detector"] and a["focal"] == b["focal"]
        assert b["provenance"]["geometry_version"] == 1 and b["params"]["geometry"]["channels"] == ["laplacian"]


def test_lap_fold_saved_keys_and_candidate_schema(tmp_path):
    from illum_baseline.stage_b import process_one

    cfg = load_config(PROJECT_ROOT / "configs" / "candidate_lap_fold.yaml")
    stem = "taiwei_cui_079"
    if not (PROJECT_ROOT / "baseline_output" / "stage_a" / f"{stem}.npz").exists():
        pytest.skip("no Stage A cache")
    meta = process_one(cfg, stem, tmp_path)
    z = np.load(tmp_path / f"{stem}.npz")
    expected = {"w", "w_net", "q", "L_over_rho", "laplacian_w_net", "laplacian_strength", "laplacian_q",
                "laplacian_L_over_rho", "fold_w_net", "fold_strength", "fold_r_n", "n_hat", "sigma_theta",
                "normal_valid", "D", "sky_mask"}
    assert set(z.files) == expected
    W_s, H_s = meta["geometry"]["stored_WxH"]
    H_n, W_n = meta["geometry"]["H_n"], meta["geometry"]["W_n"]
    assert z["w"].shape == (H_s, W_s) and z["w"].dtype == np.float16
    for k in expected - {"w"}:
        assert z[k].shape[:2] == (H_n, W_n), k  # orientation 1: stored == upright
    assert z["n_hat"].shape == (H_n, W_n, 3) and z["normal_valid"].dtype == bool
    assert z["D"].dtype == np.float32 and z["sky_mask"].dtype == bool
    w_net = z["w_net"].astype(np.float64)
    assert np.array_equal(z["w_net"], np.minimum(z["laplacian_w_net"], z["fold_w_net"]))
    assert w_net.min() >= 0.05 - 1e-3 and w_net.max() <= 1.0
    assert set(meta["channels"]) == {"laplacian", "fold"} and "combined" in meta and meta["normals_computed"] is True

    import pandas as pd
    cand = pd.read_parquet(tmp_path / f"{stem}_fold_candidates.parquet")
    assert list(cand.columns) == FOLD_COLUMNS
    assert len(cand) == meta["channels"]["fold"]["n_cand"]
    assert set(cand["reason"]) <= {"accepted", "component", "veto", "gate", "invalid_window"}
    assert int(cand["accepted"].sum()) == meta["channels"]["fold"]["n_accepted"]
    null = pd.read_parquet(tmp_path / f"{stem}_fold_null.parquet")
    assert list(null.columns) == ["u", "v", "T", "T_add", "T_add_local", "valid", "dtheta", "fallback_p", "fallback_m"]
