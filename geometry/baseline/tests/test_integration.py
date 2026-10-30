"""Spec test 13: Stage A + B on 3 real images (slow, needs CUDA and model weights).

Run with: env -u PYTHONPATH uv run pytest -m slow
"""

import dataclasses
import json

import numpy as np
import pytest

from illum_baseline import DEFAULT_CONFIG
from illum_baseline.config import load_config
from illum_baseline.io_tiff import load_tiff, to_stored

pytestmark = [pytest.mark.slow, pytest.mark.gpu]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = load_config(DEFAULT_CONFIG)
    if not cfg.data_dir.exists():
        pytest.skip(f"data_dir {cfg.data_dir} missing")
    out = tmp_path_factory.mktemp("integration")
    cfg = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, out_dir=str(out)))
    from illum_baseline.stage_a import run_stage_a
    from illum_baseline.stage_b import run_dir, run_stage_b

    sa = run_stage_a(cfg, limit=3)
    sb = run_stage_b(cfg, limit=3)
    return cfg, sa, sb, run_dir(cfg)


def test_outputs_shapes_frames_range(run):
    cfg, sa, sb, rd = run
    assert sa["n_done"] == 3 and sb["n_done"] == 3
    w_min = cfg.detector.w_min
    orientations = set()
    from illum_baseline.stage_b import image_stems

    stems = image_stems(rd)
    assert len(stems) == 3
    for stem in stems:
        meta = json.loads((rd / f"{stem}.json").read_text())
        assert meta["stem"] == stem
        raw, _ = load_tiff(cfg.data_dir / meta["file"])
        z = np.load(rd / f"{stem}.npz")
        g = meta["geometry"]
        o = g["orientation_applied"]
        orientations.add(g["upright_WxH"][0] > g["upright_WxH"][1])

        assert z["w"].shape == raw.shape[:2] and z["w"].dtype == np.float16
        assert z["D"].dtype == np.float32 and z["sky_mask"].dtype == bool
        net_stored = to_stored(np.zeros((g["H_n"], g["W_n"])), o).shape
        for k, dt in (("w_net", np.float16), ("q", np.float16), ("L_over_rho", np.float16)):
            assert z[k].shape == net_stored and z[k].dtype == dt
        w = z["w"].astype(np.float64)
        assert w.min() >= w_min - 1e-3 and w.max() <= 1.0
        D_a = np.load(cfg.out_dir / "stage_a" / f"{stem}.npz")["D"]
        np.testing.assert_array_equal(z["D"], to_stored(D_a, o))
        assert (rd / "viz" / f"{stem}.png").exists()
    assert orientations == {True, False}  # both landscape and portrait covered
    assert (rd / "manifest.parquet").exists() or (rd / "manifest.csv").exists()
