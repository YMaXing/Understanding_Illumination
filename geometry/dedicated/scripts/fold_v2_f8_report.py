"""TASK_FOLD_V2 F8: gate arms with the F7 veto (capped, veto_curv_max 0.8) on the 10 images.

A0 multiplicative (reference); A1 additive_cand (candidate-centred sigma_geo); A2 additive_local, 24-px tiles, >= 30
samples. Per arm: ROI shares of w_fold < 0.5, the 083 bollard median w_fold, the 079 box rows (w_fold, u 180-215, rows
230/254), target coverage (crease target accepted iff coverage >= 0.5), the accepted dtheta histogram, the share of
pixels with w_fold in [0.5, 0.9) per image, and the 075 class table; then the switching rule.
Writes baseline_output/diag/fold_v2/f8/f8_report.json.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fold_f4_report import classify  # noqa: E402
from fold_null_rois import TARGETS, rois, target_coverage, targets_for  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.geom.fold import FoldChannel, FoldParams  # noqa: E402
from illum_baseline.stage_b import run_geometry  # noqa: E402

OUT = ROOT / "baseline_output" / "diag" / "fold_v2" / "f8"
ARMS = {"A0 multiplicative": {"null_model": "multiplicative"},
        "A1 additive_cand": {"null_model": "additive_cand"},
        "A2 additive_local 24/30": {"null_model": "additive_local", "local_tile": 24, "local_min_samples": 30}}
FOLD_U, FOLD_ROWS = (180, 215), (230, 254)
BINS = ((0, 5), (5, 20), (20, 50), (50, 181))
CREASE_ACCEPTED = 0.5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "candidate_fold_v2.yaml")
    base = dataclasses.asdict(cfg.geometry.fold)
    rep = {a: {"rois": {}, "targets": {}, "hist": np.zeros(len(BINS), int).tolist(), "mid_share": {}, "images": {}}
           for a in ARMS}
    for f in sorted((ROOT / "baseline_output/stage_a").glob("taiwei_cui_*.npz")):
        stem, s = f.stem, f.stem[-3:]
        meta_a = json.loads(f.with_suffix(".json").read_text())
        c = np.load(f)
        ctx, _, _ = run_geometry(cfg, c["D"], c["sky_prob"].astype(np.float64), meta_a)
        for a, prm in ARMS.items():
            p = FoldParams.from_dict(base | prm)
            r = FoldChannel().run(ctx, p)
            d, sc, w = r.candidates, r.scalars, r.w_net
            R = rep[a]
            dth = np.degrees(d.loc[d.accepted, "dtheta"].to_numpy())
            h = [int(((dth >= lo) & (dth < hi)).sum()) for lo, hi in BINS]
            R["hist"] = [x + y for x, y in zip(R["hist"], h)]
            R["mid_share"][s] = float(((w >= 0.5) & (w < 0.9)).mean())
            R["images"][s] = {"n_gate": sc["n_gate"], "n_accepted": sc["n_accepted"], "hist": h,
                              "w_fold<0.5": float((w < 0.5).mean()), "w_fold<0.9": float((w < 0.9).mean())}
            for name, m in rois(stem, ctx.shape).items():
                R["rois"][name] = float((w[m] < 0.5).mean())
                if name == "083 bollard":
                    R["bollard_median"] = float(np.median(w[m]))
            if s == "079":
                R["box_rows"] = [float(w[v, FOLD_U[0]:FOLD_U[1] + 1].min()) for v in FOLD_ROWS]
            for name, (_, kind, poly) in targets_for(stem).items():
                R["targets"][name] = {"kind": kind, "coverage": target_coverage(d, poly)}
            if s == "075":
                _, counts = classify(d, w, ctx.D.astype(float), p.m / 2)
                R["075_classes"] = counts | {"share": float((w < 0.5).mean())}
        print(s, {a: rep[a]["images"][s]["n_accepted"] for a in ARMS}, flush=True)
    a0 = rep["A0 multiplicative"]
    for a, R in rep.items():
        creases = [t["coverage"] >= CREASE_ACCEPTED for t in R["targets"].values() if t["kind"] == "crease"]
        R["rule"] = {
            "rois_le_A0_plus_0.02": all(R["rois"][k] <= a0["rois"][k] + 0.02 for k in R["rois"]),
            "box_rows_le_0.25": max(R["box_rows"]) <= 0.25,
            "bollard_median_ge_0.95": R["bollard_median"] >= 0.95,
            "creases_accepted": f"{sum(creases)}/{len(creases)}",
            "creases_half": sum(creases) >= len(creases) / 2}
        R["rule"]["meets"] = all(v for k, v in R["rule"].items() if k != "creases_accepted")
    (OUT / "f8_report.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
