"""TASK_GEOM_V2 V3: geometry.depth.short_thin (off default, strong, linked)."""

import dataclasses

import numpy as np
import pytest

from illum_baseline.config import Config, apply_sets
from illum_baseline.geom.depth import DepthParams, params_record, short_thin_accept


def _pieces():
    """Surviving candidates on a vertical line u = 100 (normal +x), split into pieces; plus an off-line and a
    flipped-polarity piece. Returns x, n, lab, sizes, thin, r."""
    xs, labs = [], []
    layout = [(0, [10, 11, 12]),          # piece 0: 3 px
              (1, [16, 17]),               # piece 1: 2 px, gap 4 to piece 0
              (2, [30]),                   # piece 2: 1 px, gap 13 (no link)
              (3, [21.5, 22.5])]           # piece 3: 2 px, gap 4.5 to piece 1
    for k, vs in layout:
        for v in vs:
            xs.append([100.0, v])
            labs.append(k)
    x = np.array(xs)
    n = np.tile([1.0, 0.0], (len(x), 1))
    lab = np.array(labs)
    sizes = np.bincount(lab)
    thin = np.ones(len(x), bool)
    r = np.full(len(x), 0.1)
    return x, n, lab, sizes, thin, r


def test_strong_waives_the_minimum_for_strong_thin_pieces():
    x, n, lab, sizes, thin, r = _pieces()
    p = DepthParams(short_thin="strong")
    acc, link = short_thin_accept(p, x, n, lab, sizes, thin, r)
    assert acc[lab == 0].all() and acc[lab == 1].all() and acc[lab == 3].all() and not acc[lab == 2].any()  # >= 2 px
    r2 = r.copy()
    r2[lab == 1] = [0.1, 0.03]           # one member below tau_high: the piece is not waived
    acc2, _ = short_thin_accept(p, x, n, lab, sizes, thin, r2)
    assert not acc2[lab == 1].any()
    thin2 = thin.copy()
    thin2[0] = False                     # not all thin: piece 0 is not a thin piece
    acc3, _ = short_thin_accept(p, x, n, lab, sizes, thin2, r)
    assert not acc3[lab == 0].any()
    assert (link == -1).all()


def test_linked_joins_aligned_pieces_within_the_gap():
    x, n, lab, sizes, thin, r = _pieces()
    acc4, link4 = short_thin_accept(DepthParams(short_thin="linked", short_thin_gap_px=4.0), x, n, lab, sizes, thin, r)
    # pieces 0 + 1 link (gap 4): 5 px -> accepted; 3 is 4.5 px from 1 (not linked at 4 px); 2 alone
    assert acc4[lab <= 1].all() and not acc4[lab == 2].any() and not acc4[lab == 3].any()
    assert len(set(link4[lab <= 1])) == 1 and (link4[lab == 2] == -1).all()
    acc6, _ = short_thin_accept(DepthParams(short_thin="linked", short_thin_gap_px=6.0), x, n, lab, sizes, thin, r)
    assert acc6[lab == 3].all() and not acc6[lab == 2].any()
    # opposite polarity or an off-line endpoint blocks the link
    n2 = n.copy()
    n2[lab == 1] = [-1.0, 0.0]
    accp, _ = short_thin_accept(DepthParams(short_thin="linked", short_thin_gap_px=4.0), x, n2, lab, sizes, thin, r)
    assert not accp[lab <= 1].any()
    x2 = x.copy()
    x2[lab == 1, 0] += 2.0               # 2 px lateral offset > 1.5
    accl, _ = short_thin_accept(DepthParams(short_thin="linked", short_thin_gap_px=6.0), x2, n, lab, sizes, thin, r)
    assert not accl[lab == 0].any()


def test_short_thin_config_and_record():
    assert apply_sets(Config(), ["geometry.depth.short_thin=linked"]).geometry.depth.short_thin == "linked"
    for bad in (["geometry.depth.short_thin=on"], ["geometry.depth.short_thin_angle_deg=95"]):
        with pytest.raises(ValueError):
            apply_sets(Config(), bad)
    assert "short_thin" not in params_record(dataclasses.asdict(DepthParams()))
    rec = params_record(dataclasses.asdict(DepthParams(short_thin="strong")))
    assert rec["short_thin"] == "strong" and rec["short_thin_min_px"] == 2
