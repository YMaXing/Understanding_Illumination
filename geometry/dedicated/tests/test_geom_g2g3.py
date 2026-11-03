"""TASK_GEOM_INTEGRATION G2 / G3 options: geometry.depth.sky_band, geometry.fold.crease_rescue, ctx.s_hat_L.

Each option defaults to the old behaviour (the bit-for-bit references are in tests/test_baseline_repro.py).
"""

import dataclasses
import json
import sys

import numpy as np
import pytest

from illum_baseline import PROJECT_ROOT
from illum_baseline.config import Config, apply_sets, detector_params, load_config
from illum_baseline.geom.camera import Intrinsics
from illum_baseline.geom.context import GeometryContext
from illum_baseline.geom.depth import DepthChannel, DepthParams, params_record, sky_boundary_ring
from illum_baseline.geom.fold import FoldChannel, FoldParams
from illum_baseline.geom.fold import params_record as fold_params_record

sys.path.insert(0, str(PROJECT_ROOT / "tests"))
from test_depth_channel import _sky_scene  # noqa: E402

STAGE_A = PROJECT_ROOT / "baseline_output" / "stage_a"


def test_sky_band_depth_and_mask():
    ctx, sky = _sky_scene()
    a = DepthChannel().run(ctx)
    b = DepthChannel().run(ctx, dataclasses.replace(DepthParams(), sky_band="depth_and_mask"))
    ring = sky_boundary_ring(ctx.S, 2)
    assert (b.w_net <= a.w_net).all()
    changed = b.w_net < a.w_net
    assert changed.any() and ring[changed].all()  # only ring pixels change
    assert (b.w_net[changed] == DepthParams().w_min).all()
    assert b.scalars["n_sky_ring_px"] == int(changed.sum())
    cand_a, cand_b = a.candidates, b.candidates
    assert cand_a.equals(cand_b)  # the option paints only; candidates and decisions are unchanged
    assert "sky_band" not in params_record(dataclasses.asdict(DepthParams()))
    assert params_record(dataclasses.asdict(dataclasses.replace(DepthParams(), sky_band="depth_and_mask")))[
        "sky_band"] == "depth_and_mask"
    with pytest.raises(ValueError):
        DepthChannel().run(ctx, dataclasses.replace(DepthParams(), sky_band="bogus"))


def test_sky_band_needs_an_accepted_sky_edge():
    """A sky region without accepted sky edges stays unpainted (e.g. a misclassified sky patch inside a plane)."""
    ctx, _ = _sky_scene()
    p = dataclasses.replace(DepthParams(), sky_band="depth_and_mask", k_h=1e12)  # no candidate passes D-3
    b = DepthChannel().run(ctx, p)
    assert b.scalars["n_accepted"] == 0 and b.scalars["n_sky_ring_px"] == 0 and (b.w_net == 1).all()


def _real_ctx(stem, cfg):
    meta = json.loads((STAGE_A / f"{stem}.json").read_text())
    z = np.load(STAGE_A / f"{stem}.npz")
    from illum_baseline.config import dense_fit_params
    return GeometryContext(z["D"], z["sky_prob"].astype(np.float64), Intrinsics.from_stage_a(meta),
                           detector_params(cfg), dense_fit_params(cfg))


@pytest.mark.parametrize("stem", ["taiwei_cui_076", "taiwei_cui_088"])
def test_crease_rescue_adds_only_gate_failures(stem):
    if not (STAGE_A / f"{stem}.npz").exists():
        pytest.skip("Stage A cache not present")
    cfg = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v0.yaml")
    ctx = _real_ctx(stem, cfg)
    fp = FoldParams.from_dict(dataclasses.asdict(cfg.geometry.fold))
    off = FoldChannel().run(ctx, fp)
    on = FoldChannel().run(ctx, dataclasses.replace(fp, crease_rescue="laplacian_corroborated"))
    a, b = off.candidates, on.candidates
    assert "rescued" not in a.columns and "crease_rescue" not in off.scalars["params"]
    assert (b["accepted"] >= a["accepted"]).all()                       # nothing accepted before is lost
    new = b["accepted"] & ~a["accepted"]
    assert (new == b["rescued"]).all() and (a.loc[new, "reason"] == "gate").all()
    assert (b.loc[new, "dtheta"] >= np.radians(8.0) - 1e-12).all()
    assert (b.loc[new, "rescue_comp_px"] >= 10).all() and (b.loc[new, "rescue_support"] >= 0.7).all()
    assert on.scalars["n_rescued"] == int(new.sum())
    assert (on.w_net <= off.w_net).all()
    assert fold_params_record(dataclasses.asdict(fp)) == off.scalars["params"]


def test_s_hat_L_equals_the_detectors_s_hat():
    cfg = load_config(PROJECT_ROOT / "baseline_output" / "stage_b" / "tri_veto4" / "config_resolved.yaml")
    for stem in ("taiwei_cui_070", "taiwei_cui_079"):
        if not (STAGE_A / f"{stem}.npz").exists():
            pytest.skip("Stage A cache not present")
        rec = json.loads((PROJECT_ROOT / "baseline_output" / "stage_b" / "tri_veto4" / f"{stem}.json").read_text())
        assert _real_ctx(stem, cfg).s_hat_L == rec["detector"]["s_hat"]


def test_option_config_validation():
    cfg = apply_sets(Config(), ["geometry.depth.sky_band=depth_and_mask", "geometry.fold.crease_rescue=laplacian_corroborated",
                                "geometry.fold.rescue_theta_deg=6", "geometry.fold.rescue_k_L=2"])
    assert cfg.geometry.depth.sky_band == "depth_and_mask" and cfg.geometry.fold.crease_rescue == "laplacian_corroborated"
    for bad in (["geometry.depth.sky_band=mask"], ["geometry.fold.crease_rescue=on"],
                ["geometry.fold.rescue_support=1.5"]):
        with pytest.raises(ValueError):
            apply_sets(Config(), bad)
