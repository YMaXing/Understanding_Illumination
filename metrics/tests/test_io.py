"""§7 identity and the prediction contract (Delta B1)."""

import numpy as np
import pytest

from core import io as IO
from core import patches as PT


def write_pred(tmp_path, stem, **arrays):
    np.savez(tmp_path / f"{stem}.npz", **arrays)
    return tmp_path


def test_identity_random_prediction(cfg, dev_images, tmp_path):
    img = dev_images["taiwei_cui_070"]
    rng = np.random.default_rng(0)
    lR = (img["lI"] + rng.normal(0, 0.5, img["lI"].shape) + 3.0).astype(np.float32)
    lR2, lL = IO.load_prediction(cfg, write_pred(tmp_path, "taiwei_cui_070", log_R=lR), "taiwei_cui_070", img)
    assert lL.dtype == np.float64
    assert np.max(np.abs(lL - (img["lI"] - lR.astype(np.float64)))) <= cfg["identity_tol"]
    assert np.max(np.abs((lR2 + lL) - img["lI"])) <= cfg["identity_tol"]


@pytest.mark.parametrize("which", ["all_R", "all_L"])
def test_identity_degenerate(cfg, dev_images, tmp_path, which):
    img = dev_images["taiwei_cui_084"]
    lR = (img["lI"] if which == "all_R" else np.zeros_like(img["lI"])).astype(np.float32)
    _, lL = IO.load_prediction(cfg, write_pred(tmp_path, "taiwei_cui_084", log_R=lR), "taiwei_cui_084", img)
    want = img["lI"] - lR.astype(np.float64)
    assert np.max(np.abs(lL - want)) <= cfg["identity_tol"]


def test_log_L_key_checked(cfg, dev_images, tmp_path):
    img = dev_images["taiwei_cui_070"]
    lR = (0.5 * img["lI"]).astype(np.float32)
    lL = (img["lI"] - lR.astype(np.float64)).astype(np.float32)
    IO.load_prediction(cfg, write_pred(tmp_path, "taiwei_cui_070", log_R=lR, log_L=lL), "taiwei_cui_070", img)
    bad = lL.copy()
    bad[100, 100] += 1e-3
    with pytest.raises(IO.PredictionError, match="identity"):
        IO.load_prediction(cfg, write_pred(tmp_path, "taiwei_cui_070", log_R=lR, log_L=bad), "taiwei_cui_070", img)


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "key"])
def test_prediction_contract(cfg, dev_images, tmp_path, kind):
    img = dev_images["taiwei_cui_076"]
    lR = np.zeros((img["H"], img["W"], 3), np.float32)
    arrs = dict(log_R=lR)
    if kind == "shape":
        arrs = dict(log_R=lR.transpose(1, 0, 2).copy())
    elif kind == "dtype":
        arrs = dict(log_R=lR.astype(np.float64))
    elif kind == "nan":
        lR[5, 5, 1] = np.nan
    elif kind == "key":
        arrs = dict(R=lR)
    with pytest.raises(IO.PredictionError):
        IO.load_prediction(cfg, write_pred(tmp_path, "taiwei_cui_076", **arrs), "taiwei_cui_076", img)


def test_patch_shapes():
    assert len(PT.offsets("square", 1)[0]) == 9
    assert len(PT.offsets("square", 4)[0]) == 81
    assert len(PT.offsets("disc", 1)[0]) == 5
    assert len(PT.offsets("disc", 4)[0]) == 49
    with pytest.raises(ValueError):
        PT.offsets("hexagon", 2)


def test_patch_out_of_bounds_counts_invalid(cfg):
    X = np.zeros((20, 20, 3))
    v = np.ones((20, 20), bool)
    s = PT.patch_stats(X, v, 0, 0, 2, cfg["patch"])          # corner: 9 of 25 in bounds
    assert s["n_total"] == 25 and s["n_valid"] == 9 and not s["ok"]
    s = PT.patch_stats(X, v, 10, 10, 2, cfg["patch"])
    assert s["frac_valid"] == 1.0 and s["ok"]
