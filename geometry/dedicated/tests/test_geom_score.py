"""TASK_GEOM_V2 V1: scripts/geom_score.py reproduces the geom_v1 numbers of TASK_GEOM_V2.md (the G5 / G6 values).

Tolerance 0.0008: the task table is rounded to 3 decimals, and two of its values (texture class 0.198, 079 grass 0.219)
come from the G5 radius sweep, which recomputed the Laplacian in float64; the saved geom_v1 weights are float16
(0.19744, 0.21844). RGB-edge coverage and runtime are not in the table and are not run here.
"""

import sys

import pytest

from illum_baseline import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

TOL = 8e-4


@pytest.fixture(scope="module")
def rep():
    if not (PROJECT_ROOT / "baseline_output" / "stage_b" / "geom_v1" / "taiwei_cui_079.npz").exists():
        pytest.skip("geom_v1 run not present")
    from geom_score import score

    return score(["geom_v1"], rgb=False)["geom_v1"]


def test_geom_v1_numbers_of_the_task_file(rep):
    a = rep["audit"]["combined"]["tri-only"]
    core = rep["audit"]["core"]["tri-only"]
    g = rep["g1"]["combined"]
    want = {
        "missed per pixel": (a["missed true geometry | per pixel"]["share"], 0.079),
        "missed per unit": (a["missed true geometry | per unit"]["share"], 0.007),
        "band excess": (a["tri_band_excess | relaxed per pixel"]["share"], 0.214),
        "band excess lo": (a["tri_band_excess | relaxed per pixel"]["ci_lo"], 0.159),
        "band excess hi": (a["tri_band_excess | relaxed per pixel"]["ci_hi"], 0.271),
        "tri_fp": (a["tri_fp | relaxed per pixel"]["share"], 0.069),
        "tri_fp lo": (a["tri_fp | relaxed per pixel"]["ci_lo"], 0.038),
        "tri_fp hi": (a["tri_fp | relaxed per pixel"]["ci_hi"], 0.104),
        "cov step": (g["classes"]["step"]["coverage"], 0.900),
        "cov limb": (g["classes"]["limb"]["coverage"], 1.00),
        "cov fold": (g["classes"]["fold"]["coverage"], 0.982),
        "cov crease": (g["classes"]["crease"]["coverage"], 0.574),
        "texture class": (g["classes"]["texture"]["share_w<0.5"], 0.198),
        "079 sunlit": (g["texture_rois"]["079 grass sunlit"], 0.383),
        "079 grass": (g["texture_rois"]["079 grass"], 0.219),
        "w<0.5 batch1": (rep["wfrac"]["combined"]["batch1"]["deliverable"], 0.216),
        "w<0.5 batch2": (rep["wfrac"]["combined"]["batch2"]["deliverable"], 0.277),
        "core missed per unit": (core["missed true geometry | per unit"]["share"], 0.584),
    }
    bad = {k: (round(x, 5), y) for k, (x, y) in want.items() if abs(x - y) > TOL}
    assert not bad, bad


def test_core_is_min_of_depth_and_fold(rep):
    # the Laplacian relaxes everything the core does on the audit units (combined <= core)
    for u in rep["units"]:
        assert u["frac_combined"] >= u["frac_core"] - 1e-12
