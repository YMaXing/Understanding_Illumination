"""TASK_DEPTH_CHANNEL D4: depth-channel integration (registry, config, Stage B saving, manifest).

- config <-> params field parity, the two run configs, nested --set (geometry.depth.null), validation, YAML round trip;
- a [depth] run never builds the dense fit or the normals (no n_hat saved, normals_computed false);
- [depth, fold] saves the elementwise min of the two channels' w;
- saved npz keys and dtypes, the candidate / null parquet schemas, the JSON channel record, the manifest columns;
  strength is inf at sky edges (docs/depth_channel.md: saved outputs).
(The bit-for-bit references, including fold_v2 and tri_veto4_fold_v2: tests/test_baseline_repro.py.)
"""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from illum_baseline import PROJECT_ROOT
from illum_baseline.config import Config, DepthCfg, apply_sets, load_config
from illum_baseline.geom.channels import CHANNELS
from illum_baseline.geom.depth import DEPTH_REASONS, DepthChannel, DepthParams

CONFIGS = PROJECT_ROOT / "configs"
STAGE_A = PROJECT_ROOT / "baseline_output" / "stage_a"
STEMS = ("taiwei_cui_070", "taiwei_cui_079")  # 070 has a sky boundary

DEPTH_ARRAYS = {"depth_w_net": np.float16, "depth_strength": np.float16, "depth_dZ": np.float32,
                "depth_q_over_s": np.float16, "depth_edge_mask": np.bool_}
# per-candidate columns of <stem>_depth_candidates.parquet (docs/depth_channel.md: saved outputs)
DEPTH_COLUMNS = [
    "u", "v", "u0", "v0", "n_u", "n_v", "g_n", "rho_s0", "r_doublet", "q_p", "q_m", "s_p", "s_m", "s_sep", "centring",
    "sigma_wc", "k_c", "fit_mode", "mode_p", "mode_m", "d_next_p", "d_next_m", "rho_p", "rho_m", "se_p", "se_m",
    "d_rho", "d_rho_se", "d_rho_min_span", "m_used_p", "m_used_m", "rms_rel_p", "rms_rel_m", "excl_p", "excl_m",
    "doublet", "doublet_ratio", "signif", "signif_add", "r", "r_gate", "sky_share_near", "sky_share_far",
    "sky_edge", "sky_near", "thin", "planes_cross", "planar8_p", "planar8_m", "r_lower_bound",
    "component_id", "component_px", "comp_r_max", "rho_p_s", "rho_m_s", "dZ", "dZ_lo", "dZ_hi", "lower_bound",
    "w_c", "band", "accepted", "reason",
]
DEPTH_SCALARS = {"s_q", "sigma_g", "lambda_hat", "sigma_geo_rho", "null", "sigma_w_max", "s_lo", "s_hi", "n_cand",
                 "n_passed", "n_accepted", "n_fit_m8", "n_fit_doublet", "r_gate_P10", "r_gate_P50", "r_gate_P90",
                 "share_signif_binding", "frac_w_lt_0.5", "frac_w_lt_0.9", "params", "context_s", "runtime_s"}


def test_depth_config_parity_and_options():
    assert "depth" in CHANNELS and CHANNELS["depth"] is DepthChannel
    assert {f.name for f in dataclasses.fields(DepthCfg)} == {f.name for f in dataclasses.fields(DepthParams)}
    assert DepthParams.from_dict(dataclasses.asdict(DepthCfg())) == DepthParams()
    for name, channels, run in (("candidate_depth_only.yaml", ["depth"], "depth_v1"),
                                ("candidate_depth_fold.yaml", ["depth", "fold"], "depth_fold_v1")):
        cfg = load_config(CONFIGS / name)
        assert cfg.geometry.enabled and cfg.geometry.channels == channels and cfg.run.name == run
        assert DepthParams.from_dict(dataclasses.asdict(cfg.geometry.depth)) == DepthParams()
    fold_v2 = load_config(CONFIGS / "candidate_fold_v2.yaml").geometry.fold
    assert load_config(CONFIGS / "candidate_depth_fold.yaml").geometry.fold == fold_v2  # the fold_v2 settings
    assert load_config(CONFIGS / "default.yaml").geometry.enabled is False
    cfg = apply_sets(Config(), ["geometry.depth.null=additive", "geometry.depth.dZ0=2e-1",
                                "geometry.depth.lambda_clamp=[1, 1e3]"])
    assert cfg.geometry.depth.null == "additive" and cfg.geometry.depth.dZ0 == 0.2
    back = yaml.safe_load(yaml.safe_dump(cfg.to_dict(), sort_keys=False))  # "null" survives the YAML round trip
    assert back["geometry"]["depth"]["null"] == "additive"
    for bad in (["geometry.depth.null=bogus"], ["geometry.depth.gap_mode=min_span"], ["geometry.depth.disparity=x"],
                ["geometry.depth.tau_low=0.1"], ["geometry.depth.lambda_clamp=[5, 1]"], ["geometry.depth.nope=1"],
                ["geometry.depth.w_min=1.5"]):
        with pytest.raises(ValueError):
            apply_sets(Config(), bad)


def _run(cfg_name, tmp_path, stems=STEMS, sets=()):
    from illum_baseline.stage_b import process_one

    if not (STAGE_A / f"{stems[0]}.npz").exists():
        pytest.skip("Stage A cache not present")
    cfg = apply_sets(load_config(CONFIGS / cfg_name), list(sets))
    return cfg, {s: process_one(cfg, s, tmp_path) for s in stems}


def test_depth_only_run_no_normals_and_schema(tmp_path):
    from illum_baseline.manifest import write_manifest
    from illum_baseline.stage_b import run_geometry

    cfg, metas = _run("candidate_depth_only.yaml", tmp_path)
    for stem, meta in metas.items():
        z = np.load(tmp_path / f"{stem}.npz")
        assert set(z.files) == {"w", "w_net", "D", "sky_mask", *DEPTH_ARRAYS}  # no n_hat / normal_valid
        for k, dt in DEPTH_ARRAYS.items():
            assert z[k].dtype == dt, k
        assert np.array_equal(z["w_net"], z["depth_w_net"])  # one channel: its w unchanged
        assert meta["normals_computed"] is False
        sc = meta["channels"]["depth"]
        assert DEPTH_SCALARS <= set(sc) and sc["null"] == "multiplicative"
        assert "depth" in meta["params"]["geometry"]
        cand = pd.read_parquet(tmp_path / f"{stem}_depth_candidates.parquet")
        assert list(cand.columns) == DEPTH_COLUMNS
        assert set(cand["reason"]) <= set(DEPTH_REASONS)
        assert (cand["accepted"] == (cand["reason"] == "accepted")).all()
        assert cand["accepted"].sum() == sc["n_accepted"] > 0
        null = pd.read_parquet(tmp_path / f"{stem}_depth_null.parquet")
        assert list(null.columns) == ["u", "v", "d_rho", "d_rho_se", "planar_p", "planar_m"]
        # both significance columns are always recorded
        ok = cand["reason"].isin(["accepted", "signif", "rel", "component"])
        assert cand.loc[ok, "signif"].notna().all() and cand.loc[ok, "signif_add"].notna().all()
    # strength is inf at sky edges (070), finite elsewhere on the bands
    z = np.load(tmp_path / "taiwei_cui_070.npz")
    cand = pd.read_parquet(tmp_path / "taiwei_cui_070_depth_candidates.parquet")
    assert (cand["accepted"] & cand["sky_edge"]).sum() > 100
    assert np.isinf(z["depth_strength"]).any() and np.isinf(z["depth_dZ"]).any()
    assert (z["depth_w_net"][np.isinf(z["depth_strength"])] == np.float16(0.05)).all()
    # the context never builds the dense fit
    meta_a = json.loads((STAGE_A / "taiwei_cui_079.json").read_text())
    c = np.load(STAGE_A / "taiwei_cui_079.npz")
    ctx, _, _ = run_geometry(cfg, c["D"], c["sky_prob"].astype(np.float64), meta_a)
    assert "dense" not in ctx.__dict__ and "_normals" not in ctx.__dict__
    # manifest: depth columns for depth runs
    m = pd.read_parquet(write_manifest(tmp_path))
    assert {"depth_n_accepted", "depth_lambda_hat", "depth_null", "depth_frac_w_lt_0.5"} <= set(m.columns)


def test_depth_fold_is_elementwise_min(tmp_path):
    cfg, metas = _run("candidate_depth_fold.yaml", tmp_path, stems=("taiwei_cui_079",))
    z = np.load(tmp_path / "taiwei_cui_079.npz")
    assert np.array_equal(z["w_net"], np.minimum(z["depth_w_net"], z["fold_w_net"]))
    assert {"n_hat", "normal_valid", "fold_w_net", "fold_r_n", *DEPTH_ARRAYS} <= set(z.files)
    meta = metas["taiwei_cui_079"]
    assert meta["normals_computed"] is True and set(meta["channels"]) == {"depth", "fold"}
    assert meta["combined"]["channels"] == ["depth", "fold"]
    for t in ("depth_candidates", "depth_null", "fold_candidates", "fold_null"):
        assert (tmp_path / f"taiwei_cui_079_{t}.parquet").exists()


def test_additive_null_option(tmp_path):
    """geometry.depth.null = additive gates on signif_add; the multiplicative column is still recorded."""
    _, metas = _run("candidate_depth_only.yaml", tmp_path, stems=("taiwei_cui_079",),
                    sets=("geometry.depth.null=additive",))
    sc = metas["taiwei_cui_079"]["channels"]["depth"]
    assert sc["null"] == "additive" and sc["sigma_geo_rho"] > 0
    cand = pd.read_parquet(tmp_path / "taiwei_cui_079_depth_candidates.parquet")
    rej = cand["reason"] == "signif"
    assert (cand.loc[rej, "signif_add"] <= DepthParams().z).all()
    ok = cand["reason"].isin(["accepted", "rel", "component"]) & ~cand["sky_edge"] & ~cand["thin"]
    assert (cand.loc[ok, "signif_add"] > DepthParams().z).all() and cand.loc[ok, "signif"].notna().all()
