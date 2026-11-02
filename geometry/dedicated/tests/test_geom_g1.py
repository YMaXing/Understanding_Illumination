"""TASK_GEOM_INTEGRATION G1: the audit's component rules and quota allocation, and the evaluation metrics."""

import sys

import numpy as np

from illum_baseline import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from geom_audit import allocate, components  # noqa: E402
from geom_eval import roi_metrics  # noqa: E402


def test_components_merge_within_2px_and_drop_small():
    m = np.zeros((40, 60), bool)
    m[5:10, 5:10] = True     # 25 px
    m[5:10, 12:15] = True    # 15 px, gap of 2 columns (10, 11) -> merged with the first
    m[20:22, 5:9] = True     # 8 px -> dropped
    m[30:35, 30:33] = True   # 15 px, far away -> its own component
    m[30:35, 37:40] = True   # 15 px, gap of 4 columns -> separate
    lab, n = components(m)
    assert n == 3
    assert len(set(lab[5:10, 5:15][m[5:10, 5:15]])) == 1  # merged
    assert (lab[20:22, 5:9] == 0).all()                    # dropped
    assert lab[32, 31] != lab[32, 38]
    assert ((lab > 0) == (m & (lab > 0))).all()            # labels only on original pixels


def test_allocate_proportional_min_one_exact_total():
    a = np.array([1000.0, 10.0, 0.0, 300.0, 5.0])
    n = allocate(a, 140)
    assert n.sum() == 140 and n[2] == 0 and (n[[0, 1, 3, 4]] >= 1).all()
    assert n[0] > n[3] > n[1]


def test_roi_metrics_polyline_coverage_and_area():
    w = np.ones((50, 50))
    w[20, 10:40] = 0.05  # a horizontal relaxed line
    on = {"kind": "polyline", "poly": [(12.0, 20.0), (37.0, 20.0)]}
    off_by_1 = {"kind": "polyline", "poly": [(12.0, 21.0), (37.0, 21.0)]}
    off_by_2 = {"kind": "polyline", "poly": [(12.0, 22.0), (37.0, 22.0)]}
    m = roi_metrics(w, on)
    assert m["share_w<0.5"] == 1.0 and m["coverage"] == 1.0 and m["median_w"] == 0.05
    m1 = roi_metrics(w, off_by_1)
    assert m1["share_w<0.5"] == 0.0 and m1["coverage"] == 1.0  # within 1 px
    assert roi_metrics(w, off_by_2)["coverage"] == 0.0
    box = {"kind": "area", "box": (10, 39, 19, 21)}
    ma = roi_metrics(w, box)
    assert ma["coverage"] is None and abs(ma["share_w<0.5"] - 30 / 90) < 1e-12


def test_split_units_partition_large_components():
    from geom_audit import split_units

    lab = np.zeros((64, 96), np.int32)
    lab[4:60, 4:90] = 1      # 4816 px, spans 2 x 3 tiles of 32
    lab[62:64, 0:20] = 2     # 40 px, small: kept whole
    units, n = split_units(lab, split_px=1000, tile=32)
    assert n == 2 * 3 + 1
    big = units[lab == 1]
    assert len(np.unique(big)) == 6 and (big > 0).all()
    assert len(np.unique(units[lab == 2])) == 1
    assert ((units > 0) == (lab > 0)).all()  # a partition of the original pixels (no piece under 10 px here)
    same, n0 = split_units(lab, split_px=0, tile=32)
    assert n0 == 2 and np.array_equal(same, lab)
