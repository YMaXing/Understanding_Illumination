"""TASK_FOLD_CHANNEL F4: evaluation hook, fold sheet and profile tool."""

import json

import numpy as np
import pandas as pd
import pytest

from illum_baseline import PROJECT_ROOT
from illum_baseline.config import load_config
from illum_baseline.geom.evaluate import evaluate_folds, evaluate_run, load_annotations

STEM = "taiwei_cui_079"


def test_evaluate_folds_precision_recall(tmp_path):
    ann = {"tolerance_px": 1.0, "polylines": [
        {"label": "fold", "points": [[10, 10], [30, 10]]},
        {"label": "not_fold", "points": [[10, 50], [20, 50]]}]}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(ann))
    a = load_annotations(p)
    on_fold = [(u, 10.5) for u in range(10, 31)]          # 21 TP, cover the whole fold polyline
    on_not = [(12, 50.2), (15, 49.5), (18, 50.0)]         # 3 FP
    elsewhere = [(100, 100), (200, 5)]                    # unannotated: neither
    pts = on_fold + on_not + elsewhere + [(40, 40)]
    df = pd.DataFrame({"u0": [x for x, _ in pts], "v0": [y for _, y in pts],
                       "accepted": [True] * (len(pts) - 1) + [False]})
    r = evaluate_folds(df, a)
    assert (r["tp"], r["fp"], r["n_accepted"]) == (21, 3, 26)
    assert r["precision"] == pytest.approx(21 / 24) and r["precision_all"] == pytest.approx(21 / 26)
    assert r["recall"] == 1.0
    half = df[df["u0"] <= 20].copy()  # only the left half of the fold covered
    assert evaluate_folds(half, a)["recall"] == pytest.approx(0.5, abs=0.05)
    with pytest.raises(ValueError):
        p.write_text(json.dumps({"polylines": [{"label": "edge", "points": [[0, 0], [1, 1]]}]}))
        load_annotations(p)


@pytest.fixture(scope="module")
def lap_fold_run(tmp_path_factory):
    from illum_baseline.stage_b import process_one

    if not (PROJECT_ROOT / "baseline_output" / "stage_a" / f"{STEM}.npz").exists():
        pytest.skip("no Stage A cache")
    cfg = load_config(PROJECT_ROOT / "configs" / "candidate_lap_fold.yaml")
    rd = tmp_path_factory.mktemp("lap_fold")
    process_one(cfg, STEM, rd)
    return cfg, rd


def test_fold_sheet_renders(lap_fold_run):
    from illum_baseline.viz_fold import render_fold

    cfg, rd = lap_fold_run
    out = render_fold(cfg, rd, STEM)
    assert out == rd / "viz_fold" / f"{STEM}.png" and out.stat().st_size > 50_000


def test_profile_tool_and_eval_run(lap_fold_run, tmp_path):
    from illum_baseline.diag_fold import diag_fold

    cfg, rd = lap_fold_run
    out = diag_fold(cfg, rd, STEM, 197.0, 230.0)
    assert out.exists() and out.stat().st_size > 30_000
    cand = pd.read_parquet(rd / f"{STEM}_fold_candidates.parquet")
    acc = cand[cand["accepted"]].iloc[:5]
    pts = acc[["u0", "v0"]].to_numpy().tolist()
    (tmp_path / f"{STEM}.json").write_text(json.dumps({"tolerance_px": 0.5, "polylines": [
        {"label": "fold", "points": pts[:2]}]}))
    df = evaluate_run(rd, tmp_path)
    assert list(df["stem"]) == [STEM] and df["tp"].iloc[0] >= 2 and np.isfinite(df["recall"].iloc[0])
