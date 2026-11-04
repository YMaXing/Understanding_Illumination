"""TASK_GEOM_V2 V4: geometry.fold.rescue_undecidable (reject default, corroborate) and ctx.L_veto_over_rho_f."""

import dataclasses
import sys

import numpy as np
import pytest

from illum_baseline import PROJECT_ROOT
from illum_baseline.config import Config, apply_sets, load_config
from illum_baseline.geom.fold import FoldChannel, FoldParams, params_record

sys.path.insert(0, str(PROJECT_ROOT / "tests"))
from test_geom_g2g3 import _real_ctx  # noqa: E402

STAGE_A = PROJECT_ROOT / "baseline_output" / "stage_a"


def test_option_config_and_record():
    assert apply_sets(Config(), ["geometry.fold.veto_mode=capped", "geometry.fold.rescue_undecidable=corroborate"]
                      ).geometry.fold.rescue_undecidable == "corroborate"
    for bad in (["geometry.fold.rescue_undecidable=accept"],
                ["geometry.fold.veto_mode=fixed", "geometry.fold.rescue_undecidable=corroborate"]):
        with pytest.raises(ValueError):
            apply_sets(Config(), bad)
    on = dataclasses.asdict(FoldParams(crease_rescue="laplacian_corroborated", rescue_undecidable="corroborate"))
    assert params_record(on)["rescue_undecidable"] == "corroborate"
    assert "rescue_undecidable" not in params_record(dataclasses.asdict(FoldParams(crease_rescue="laplacian_corroborated")))


@pytest.mark.parametrize("stem", ["taiwei_cui_076"])
def test_corroborate_adds_only_undecidable_gate_failures(stem):
    if not (STAGE_A / f"{stem}.npz").exists():
        pytest.skip("no Stage A cache")
    cfg = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v1.yaml")
    ctx = _real_ctx(stem, cfg)
    fp = FoldParams.from_dict(dataclasses.asdict(cfg.geometry.fold))
    a = FoldChannel().run(ctx, fp)
    b = FoldChannel().run(ctx, dataclasses.replace(fp, rescue_undecidable="corroborate"))
    A, B = a.candidates, b.candidates
    assert "rescue_undecidable" not in A.columns
    for col in ("gate", "veto_pass", "J", "dtheta", "w_c", "rescue_eligible", "rescue_veto_pass"):
        assert A[col].equals(B[col]), col
    new = B["accepted"] & ~A["accepted"]
    lost = A["accepted"] & ~B["accepted"]
    assert new.any() and not lost.any()
    assert (A.loc[new, "reason"] == "gate").all() and B.loc[new, "rescued"].all()
    # every new component holds an undecidable member; decided-fail candidates are never rescued
    assert (B.loc[new, "rescue_undecidable"] | B.loc[new, "rescue_veto_pass"]).all()
    assert not (B["rescued"] & B["rescue_eligible"] & ~B["rescue_veto_pass"] & ~B["rescue_undecidable"]).any()
    assert b.scalars["n_rescued_undecidable"] == int((B["rescued"] & B["rescue_undecidable"]).sum())


def test_vetoed_response():
    stem = "taiwei_cui_083"
    if not (STAGE_A / f"{stem}.npz").exists():
        pytest.skip("no Stage A cache")
    cfg = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v1.yaml")
    ctx = _real_ctx(stem, cfg)
    Lv, L = ctx.L_veto_over_rho_f, ctx.L_over_rho_f
    assert (np.abs(Lv) <= np.abs(L) + 1e-15).all()
    nz = Lv != 0
    assert (np.sign(Lv[nz]) == np.sign(L[nz])).all()
    # on the bollard's smooth body (the trimmed interior ROI) the veto removes most of the response
    m = np.zeros(L.shape, bool)
    m[260:319, 175:206] = True
    assert np.median(np.abs(Lv[m])) < 0.5 * np.median(np.abs(L[m]))
