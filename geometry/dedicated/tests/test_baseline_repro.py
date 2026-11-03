"""baseline_v0, tri_veto4 and the fold runs must reproduce bit-for-bit (TASK_070_STRIPES, TASK_FOLD_CHANNEL working
rules; TASK_FOLD_V2 adds fold_only and tri_veto4_fold; TASK_DEPTH_CHANNEL D4 adds fold_v2 and
tri_veto4_fold_v2; TASK_GEOM_INTEGRATION G5 adds geom_v1 on both batches).

Stage B: recompute every image of each reference run from its Stage A cache and compare all saved arrays (NaN equal
to NaN), the JSON blocks and, for geometry runs, the candidate / null tables and the channel scalars (per-run
timings excluded).
Stage A (slow, GPU): rerun preprocessing + DA3 on one image and compare D and sky_prob to the cache.
"""

import json
import warnings

import numpy as np
import pytest

from illum_baseline import PROJECT_ROOT
from illum_baseline.config import load_config

RUN = "baseline_v0"
# (output folder, run): the 10 taiwei_cui images, and the 10 batch2 images
REFERENCE_RUNS = [("baseline_output", "baseline_v0"), ("baseline_output", "tri_veto4"),
                  ("baseline_output_batch2", "tri_veto4"),
                  ("baseline_output", "fold_only"), ("baseline_output", "tri_veto4_fold"),
                  ("baseline_output", "fold_v2"), ("baseline_output", "tri_veto4_fold_v2"),
                  ("baseline_output", "geom_v1"), ("baseline_output_batch2", "geom_v1")]
TIMING_KEYS = {"context_s", "runtime_s"}  # wall-clock, differs between runs


def _run_cfg(out: str = "baseline_output", run: str = RUN):
    rd = PROJECT_ROOT / out / "stage_b" / run
    if not (rd / "config_resolved.yaml").exists():
        pytest.skip(f"{rd} not present")
    cfg = load_config(rd / "config_resolved.yaml")
    if cfg.out_dir.resolve() != (PROJECT_ROOT / out).resolve():
        warnings.warn(f"{rd}/config_resolved.yaml points at {cfg.out_dir}, not {PROJECT_ROOT / out}; "
                      "the Stage A cache is read from there")
    return cfg, rd


@pytest.mark.parametrize("out, run", REFERENCE_RUNS, ids=[f"{o}/{r}" for o, r in REFERENCE_RUNS])
def test_stage_b_reproduces_bit_for_bit(tmp_path, out, run):
    from illum_baseline.stage_b import image_stems, process_one

    cfg, rd = _run_cfg(out, run)
    stems = image_stems(rd)
    assert stems
    for stem in stems:
        process_one(cfg, stem, tmp_path)
        old, new = np.load(rd / f"{stem}.npz"), np.load(tmp_path / f"{stem}.npz")
        assert sorted(old.files) == sorted(new.files)
        for k in old.files:
            nan_ok = np.issubdtype(old[k].dtype, np.floating)
            assert old[k].dtype == new[k].dtype and np.array_equal(old[k], new[k], equal_nan=nan_ok), (stem, k)
        a, b = json.loads((rd / f"{stem}.json").read_text()), json.loads((tmp_path / f"{stem}.json").read_text())
        for sec in ("detector", "focal", "geometry", "params"):
            assert a[sec] == b[sec], (stem, sec)
        if "channels" in a:  # geometry run
            assert a["combined"] == b["combined"] and a["normals_computed"] == b["normals_computed"], stem
            for ch in a["channels"]:
                strip = lambda d: {k: v for k, v in d.items() if k not in TIMING_KEYS}  # noqa: E731
                assert strip(a["channels"][ch]) == strip(b["channels"][ch]), (stem, ch)
            import pandas as pd

            tables = sorted(p.name for p in rd.glob(f"{stem}_*.parquet"))
            assert tables == sorted(p.name for p in tmp_path.glob(f"{stem}_*.parquet")), stem
            for t in tables:
                pd.testing.assert_frame_equal(pd.read_parquet(rd / t), pd.read_parquet(tmp_path / t), check_exact=True)


@pytest.mark.slow
@pytest.mark.gpu
def test_stage_a_reproduces_bit_for_bit():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from illum_baseline.depth import DepthEstimator
    from illum_baseline.stage_a import prepare, stage_a_dir

    cfg, _ = _run_cfg()
    stem = "taiwei_cui_070"
    meta = json.loads((stage_a_dir(cfg) / f"{stem}.json").read_text())
    cache = np.load(stage_a_dir(cfg) / f"{stem}.npz")
    net = prepare(cfg, cfg.data_dir / meta["file"])["net"]
    de = DepthEstimator(cfg.da3.model_id, cfg.da3.process_res, cfg.da3.process_res_method,
                        torch.device("cuda"), cfg.seed)
    pred = de.predict([net])[0]
    assert np.array_equal(pred["D"], cache["D"])
    assert np.array_equal(pred["sky_prob"], cache["sky_prob"])
