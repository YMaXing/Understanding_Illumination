"""§7 log-I parity (Delta B1): ℓI from the configured loader hook vs the frozen exporter CSV, on all 10 dev images.

The CSV (illumination_edge/outputs/annotate/frozen_v2_pairs_fullres.csv) holds per-pair square-patch medians of
log(max(I, 1)) over valid pixels with 5 decimals, so the tolerance is 5e-6 (re-run at 1e-6 once a real loader exists).
"""

import csv

import numpy as np
import pytest

from conftest import DEV
from core import config as K
from core import io as IO
from core import patches as PT


def rows(cfg):
    p = cfg["parity"]["csv"]
    if not p or not K.repo_path(p).exists():
        pytest.skip("parity CSV not available on this machine (illumination_edge/outputs/ is git-ignored); "
                    "set parity.csv in metrics/config.local.yaml to run the parity test")
    with open(K.repo_path(p), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_parity_csv_covers_dev(cfg):
    R = rows(cfg)
    assert len(R) == 871
    assert sorted({r["image"] for r in R}) == DEV


def test_log_i_patch_medians_match_frozen_csv(cfg, dev_images):
    assert cfg["patch"]["shape"] == "square"
    tol = cfg["parity"]["tol"]
    worst, n, exact = 0.0, 0, 0
    for r in rows(cfg):
        img = dev_images[r["image"]]
        for e in "ab":
            h = int(r[f"half_{e}"] or r["half"])
            s = PT.patch_stats(img["lI"], img["valid"], int(r[f"{e}_col"]), int(r[f"{e}_row"]), h, cfg["patch"])
            assert s["n_valid"] == int(r[f"{e}_n_valid"]), (r["t"], e)
            ref = np.array([float(r[f"{e}_log{c}"]) for c in "RGB"])
            if s["n_valid"] == 0:
                continue
            d = float(np.max(np.abs(s["median"] - ref)))
            worst = max(worst, d)
            n += 1
            # stronger than the tolerance: rounded as the exporter rounds (round(float(x), 5)), every value is identical
            exact += all(round(float(s["median"][i]), 5) == float(r[f"{e}_log{c}"]) for i, c in enumerate("RGB"))
    print(f"\nparity: {n} patch medians x 3 channels, max |dev| {worst:.3g} (tol {tol:g}); "
          f"identical after the exporter's rounding: {exact}/{n}")
    assert worst <= tol
    assert exact == n


def test_validity_mask_matches_detectors(cfg, dev_images):
    """At the defaults (eps 0, sat 65000, no loader mask) the mask equals illumination_edge common.valid_mask."""
    vm = K.import_hook("illumination_edge/common.py:valid_mask")
    for s, img in dev_images.items():
        assert np.array_equal(img["valid"], vm(img["I"])), s


def test_hook_is_imported_not_copied(cfg):
    f = K.import_hook(cfg["loader"]["log_i"])
    assert f.__module__ in ("common",) or f.__module__.startswith("_metrics_hook_common")
    assert f.__code__.co_filename.endswith("illumination_edge/common.py")


@pytest.mark.parametrize("sat", [64224])
def test_sat_threshold_is_config(cfg, dev_images, sat):
    c2 = K.load_config(overrides={"validity": {"sat_dn": sat}})
    img = dev_images["taiwei_cui_083"]
    v2 = IO.validity_mask(c2, img["I"])
    assert (img["valid"] & ~v2).sum() == img["sat_band"]
