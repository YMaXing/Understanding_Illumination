"""Phase 3: aggregation (§5 + deltas), dense diagnostics (§6), and the CLI prediction path."""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import synth as S
from core import aggregate as AG
from core import baselines as BL
from core import dense as DN
from core import pair_metrics as PM

TAU = 0.10


@pytest.fixture(scope="module")
def world(cfg):
    sc = S.scene(0)
    img = S.image_dict(cfg, sc)
    return sc, img, S.pairs()


def rows_for(cfg, img, lR, P, image="synth"):
    out = []
    for rec, J, _ in PM.score_pairs(P, S.ctx(img, lR), cfg, [0.05, 0.10, 0.20]):
        rec = dict(rec, image=image)
        out.append(dict(rec=rec, J=J))
    return out


def test_headline_na_and_split(cfg, world):
    sc, img, P = world
    only_i = [p for p in P if p["type"] == "i"]
    h = AG.headline(rows_for(cfg, img, BL.all_R(img["lI"]), only_i), TAU)
    assert h["split"] is None and h["text"]["split"] == "n/a: 0 m pairs"
    assert h["text"]["leak"] == "n/a: 0 pairs" and h["composite"] is None
    h = AG.headline(rows_for(cfg, img, BL.all_R(img["lI"]), P), TAU)
    assert h["ER"]["i"]["rate"] == 1.0 and h["ER"]["m"]["rate"] == 0.0 and h["split"] == 0.5
    assert h["leak"] == 0.0 and h["composite"] is not None


def test_low_contrast_excluded_from_split(cfg, world):
    sc, img, P = world
    R = rows_for(cfg, img, BL.all_L(img["lI"]), P)
    chroma = [r for r in R if r["rec"]["id"] in {p["id"] for p in P if p["tag"] == "m_chroma"}]
    assert all(not r["J"][0.20]["exclusions"] == [] for r in chroma)         # ||ΔℓI|| 0.173 <= 0.20
    assert all("low_contrast" in r["J"][0.20]["exclusions"] for r in chroma)
    assert all("low_contrast" not in r["J"][0.10]["exclusions"] for r in chroma)
    assert all("low_contrast" not in r["J"][0.20]["exclusions"] for r in R if r["rec"]["kind"].startswith("s_"))


def test_bootstrap_deterministic_and_covers(cfg, world):
    sc, img, P = world
    R = []
    for k, seed in enumerate(range(4)):                                     # 4 "images" from 4 scenes
        sc2 = S.scene(seed)
        im2 = S.image_dict(cfg, sc2)
        R += rows_for(cfg, im2, BL.blur_L(im2["lI"], cfg), P, image=f"img{k}")
    imgs = [f"img{k}" for k in range(4)]
    a = AG.bootstrap(R, TAU, imgs, 500, 7)
    b = AG.bootstrap(R, TAU, imgs, 500, 7)
    assert a == b
    h = AG.headline(R, TAU)
    c = a["ci"]["split"]
    assert c["lo"] <= h["split"] <= c["hi"]
    assert a["ci"]["d_R"]["n_undefined"] == 0
    one = AG.bootstrap(R, TAU, ["img0"], 200, 1)["ci"]["i"]
    assert one["lo"] == one["hi"]                                           # one cluster: no spread


class FakeMaps:
    def __init__(self, img, illum, mat):
        self.w = np.ones(img["valid"].shape, np.float32)
        self.illum, self.mat, self.geom = illum, mat, np.zeros_like(illum)
        self.gI, self.okI = DN.grad_norm(img["lI"], img["valid"])
        self.d3_mask = mat


def test_dense_degenerate_values(cfg, world):
    sc, img, P = world
    yy, xx = np.mgrid[0:S.H, 0:S.W]
    illum = (yy == S.EDGE_Y) & (xx < 300)
    mat = (xx == S.TEX_X[1]) & (yy < 140)
    maps = FakeMaps(img, illum, mat)
    lI = img["lI"]
    dR = DN.diagnostics(cfg, img, maps, BL.all_R(lI), lI - BL.all_R(lI))
    dL = DN.diagnostics(cfg, img, maps, BL.all_L(lI), lI - BL.all_L(lI))
    assert dR["D1"] == 0.0 and dR["D2"] == pytest.approx(1.0) and dR["D3"] == 0.0
    assert dL["D2"] == 0.0 and dL["D3"] == pytest.approx(1.0)
    dG = DN.diagnostics(cfg, img, maps, sc["lR_gt"], lI - sc["lR_gt"])
    assert dG["D2"] < 0.05                                                  # GT: shadow stays out of R
    assert dG["D4"] == DN.NA_REGION and dG["A1_q0.90"] == DN.NA_REGION


def test_cli_prediction_path(cfg, tmp_path):
    """A prediction folder holding all_R as float32 must score like the all_R baseline."""
    from core import io as IO
    from conftest import need_dataset
    stem = "taiwei_cui_070"
    need_dataset(cfg, [stem])
    img = IO.load_image(cfg, stem)
    pred = tmp_path / "pred"
    pred.mkdir()
    np.savez(pred / f"{stem}.npz", log_R=img["lI"].astype(np.float32))
    out = tmp_path / "out"
    M = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-B", str(M / "run_metrics.py"), "--pred", str(pred), "--images", stem,
                        "--out", str(out), "--test"], cwd=M, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = list(csv.DictReader(open(out / "per_pair.csv", newline="")))
    mod = {x["id"]: x for x in rows if x["method"] == "model"}
    base = {x["id"]: x for x in rows if x["method"] == "all_R"}
    assert mod.keys() == base.keys() and len(mod) == 39
    for k in mod:
        assert (mod[k]["pass_wide"], mod[k]["bin_wide"], mod[k]["exclusions"]) == \
               (base[k]["pass_wide"], base[k]["bin_wide"], base[k]["exclusions"]), k
    for f in ("per_pair.csv", "profiles.npz", "per_image.csv", "summary.json", "summary.md"):
        assert (out / f).exists()


def test_fixed_population_sweep_and_floor(cfg, world):
    sc, img, P = world
    R = rows_for(cfg, img, BL.blur_L(img["lI"], cfg), P)
    for t in (0.05, 0.10, 0.20):
        f = AG.headline_fixed_pop(R, t, 0.10)
        n_fixed = sum(1 for r in R if r["rec"]["kind"] == "i" and not r["J"][0.10]["exclusions"])
        assert f["ER"]["i"]["n"] == n_fixed                           # population frozen at tau 0.10
    assert AG.headline_fixed_pop(R, 0.10, 0.10) == AG.headline(R, 0.10)
    ref = {"floor": {"0.1": {"o|i|textured": dict(n=100, fail_wide=2), "o|i|not_textured": dict(n=100, fail_wide=1),
                             "o|m|textured": dict(n=10, fail_wide=0)},
                     "0.2": {"o|i|textured": dict(n=100, fail_wide=20)}}}
    cnt = {"i": dict(smooth=50, moderate=30, strong=20), "m": dict(smooth=10, moderate=0, strong=0)}
    fl = AG.gt_floor(cnt, ref, "o")
    assert fl["lo"]["ER"]["i"] == pytest.approx((30 * 0.02 + 20 * 0.20) / 100)
    assert fl["hi"]["ER"]["i"] == pytest.approx((50 * 0.01 + 30 * 0.02 + 20 * 0.20) / 100)
    assert fl["lo"]["split"] == pytest.approx(fl["lo"]["ER"]["i"] / 2)


def test_report_blocks_by_split(cfg, tmp_path):
    """train and val are separate blocks with a gap table; test images are not loaded without --test."""
    import json as J
    from conftest import need_dataset
    from core import config as K
    imgs = ["taiwei_cui_080", "taiwei_cui_070", "taiwei_cui_084"]          # train, val, test
    need_dataset(cfg, imgs)
    assert [K.split_of(cfg, s) for s in imgs] == ["train", "val", "test"]
    M = Path(__file__).resolve().parent.parent
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-B", str(M / "run_metrics.py"), "--images", ",".join(imgs), "--out", str(out)],
                       cwd=M, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    s = J.loads((out / "summary.json").read_text())
    assert set(s["summary"]) == {"train", "val"} and s["meta"]["skipped_test_images"] == 1
    assert set(s["meta"]["train_val_gap"]["all_L"]) == {"split", "leak", "long_range", "composite"}
    assert "taiwei_cui_084" not in s["meta"]["images"]
    md = (out / "summary.md").read_text()
    assert "Train - val gap" in md and "Data split: train" in md and "Data split: val" in md and "Data split: test" not in md
    rows = list(csv.DictReader(open(out / "per_pair.csv", newline="")))
    assert {x["data_split"] for x in rows} == {"train", "val"} and {x["used_in_loss"] for x in rows} == {"False"}
