"""TASK_GEOM_V2 V2: geometry.fold.site (nms default, plane_intersection)."""

import dataclasses
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from illum_baseline import PROJECT_ROOT
from illum_baseline.config import Config, apply_sets, load_config
from illum_baseline.geom.fold import FoldChannel, FoldParams, params_record

sys.path.insert(0, str(PROJECT_ROOT / "tests"))
from test_geom_g2g3 import _real_ctx  # noqa: E402

STAGE_A = PROJECT_ROOT / "baseline_output" / "stage_a"


def test_site_smoothing_and_validity():
    # a straight chain of 7 candidates (vertical fold line, e1 = +x) and a separate chain of 3
    x0 = np.array([[50.0, 10.0 + k] for k in range(7)] + [[120.0, 40.0 + k] for k in range(3)])
    e1 = np.tile([1.0, 0.0], (10, 1))
    s_gap = np.array([1.0, 1.2, 0.8, 2.9, 1.1, 1.0, 9.0, 2.0, -2.5, 0.5])
    cross = np.ones(10, bool)
    cross[2] = False                             # no intersection: keeps its NMS pixel
    r1 = SimpleNamespace(planes_cross=cross, s_gap=s_gap)
    u, v = np.rint(x0[:, 0]).astype(int), np.rint(x0[:, 1]).astype(int)
    p = FoldParams(site="plane_intersection")
    s = FoldChannel._site(p, x0, e1, r1, np.ones(10, bool), u, v, (200, 200))
    assert s["site_valid"].tolist() == [True, True, False, True, True, True, False, True, True, True]  # |t| <= 3
    t = s["site_t_applied"]
    assert t[2] == 0 and t[6] == 0 and s["u_site"][2] == 50 and s["u_site"][6] == 50     # invalid: NMS pixel kept
    # chain of 7: running median over 5 valid neighbours (NaN ignored), window shrunk at the ends
    assert t[3] == pytest.approx(np.nanmedian([1.2, np.nan, 2.9, 1.1, 1.0]))
    assert t[0] == pytest.approx(1.0)                                                      # chain end: own value
    assert t[7:].tolist() == [2.0, -2.5, 0.5]                                              # chain of 3: own t
    assert s["u_site"][7] == 122 and s["u_site"][8] == 118


def test_site_option_config_and_record():
    cfg = apply_sets(Config(), ["geometry.fold.site=plane_intersection"])
    assert cfg.geometry.fold.site == "plane_intersection"
    for bad in (["geometry.fold.site=centre"], ["geometry.fold.site_max_t=0"]):
        with pytest.raises(ValueError):
            apply_sets(Config(), bad)
    assert "site" not in params_record(dataclasses.asdict(FoldParams()))
    assert params_record(dataclasses.asdict(FoldParams(site="plane_intersection")))["site"] == "plane_intersection"


def test_site_moves_only_the_band():
    stem = "taiwei_cui_079"
    if not (STAGE_A / f"{stem}.npz").exists():
        pytest.skip("no Stage A cache")
    cfg = load_config(PROJECT_ROOT / "configs" / "candidate_geom_v1.yaml")
    ctx = _real_ctx(stem, cfg)
    fp = FoldParams.from_dict(dataclasses.asdict(cfg.geometry.fold))
    a = FoldChannel().run(ctx, fp)
    b = FoldChannel().run(ctx, dataclasses.replace(fp, site="plane_intersection"))
    A, B = a.candidates, b.candidates
    for col in ("accepted", "reason", "J", "dtheta", "w_c", "band", "gate", "veto_pass"):
        assert A[col].equals(B[col]), col
    assert "site_t" not in A.columns and "site" not in a.scalars["params"]
    acc = B["accepted"]
    mv = acc & B["site_valid"]
    assert (B.loc[B["site_valid"], "site_t"].abs() <= 3).all()
    assert (B.loc[~mv, "u_site"] == B.loc[~mv, "u"]).all() and (B.loc[~mv, "v_site"] == B.loc[~mv, "v"]).all()
    xs = B.loc[mv, "u0"] + B.loc[mv, "site_t_applied"] * B.loc[mv, "e1_u"]
    assert (np.clip(np.rint(xs), 0, ctx.shape[1] - 1).astype(int) == B.loc[mv, "u_site"]).all()
    assert b.scalars["n_site_valid"] == int(mv.sum()) and not np.array_equal(a.w_net, b.w_net)
    # same number of relaxed pixels up to the band overlap; the default reproduces the saved geom_v1 fold w
    saved = json.loads((PROJECT_ROOT / "baseline_output" / "stage_b" / "geom_v1" / f"{stem}.json").read_text())
    assert saved["channels"]["fold"]["n_accepted"] == a.scalars["n_accepted"]
