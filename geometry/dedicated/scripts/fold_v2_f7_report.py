"""TASK_FOLD_V2 F7: the capped veto on the 10 images (multiplicative gate), against the fixed veto (= fold_only).

Arms: fixed; capped with veto_curv_max 0.7 (task) and 0.8 (calibration: one-side-capped skips have C_curv 0.75-0.77).
Per image: counts by stage, narrow_face / veto_undecidable / C_high, the split of the fixed veto's no_veto acceptances,
the 083 bollard-limb acceptances; per target: coverage (scripts/fold_null_rois.py TARGETS); ROIs: share of
w_fold < 0.5; 079 box rows 230/254 (w_fold and combined with tri_veto4).
Writes baseline_output/diag/fold_v2/f7/f7_report.json.
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

from fold_null_rois import rois, target_coverage, targets_for  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.geom.fold import FoldChannel, FoldParams  # noqa: E402
from illum_baseline.stage_b import load_upright, run_geometry  # noqa: E402

OUT = ROOT / "baseline_output" / "diag" / "fold_v2" / "f7"
ARMS = {"fixed": {}, "capped_0.7": {"veto_mode": "capped", "veto_curv_max": 0.7},
        "capped_0.8": {"veto_mode": "capped", "veto_curv_max": 0.8}}
LIMBS_083 = ((145, 168, 180, 330), (215, 240, 180, 330))  # u0 u1 v0 v1 of the bollard's side silhouettes
FOLD_U, FOLD_ROWS = (180, 215), (230, 254)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "candidate_fold_only.yaml")
    tcfg = load_config(ROOT / "baseline_output/stage_b/tri_veto4/config_resolved.yaml")
    base = dataclasses.asdict(cfg.geometry.fold)
    rep = {"images": {}, "targets": {}, "rois": {}, "box_rows": {}}
    for f in sorted((ROOT / "baseline_output/stage_a").glob("taiwei_cui_*.npz")):
        stem, s = f.stem, f.stem[-3:]
        meta_a = json.loads(f.with_suffix(".json").read_text())
        c = np.load(f)
        ctx, _, _ = run_geometry(cfg, c["D"], c["sky_prob"].astype(np.float64), meta_a)
        wt = load_upright(tcfg, ROOT / "baseline_output/stage_b/tri_veto4", stem)[0]["w_net"].astype(float)
        res = {a: FoldChannel().run(ctx, FoldParams.from_dict(base | prm)) for a, prm in ARMS.items()}
        fx = res["fixed"].candidates
        nv_fixed = (fx.accepted & fx.no_veto).to_numpy()
        img = {}
        for a, r in res.items():
            d, sc = r.candidates, r.scalars
            row = {k: sc[k] for k in ("n_cand", "n_valid", "n_gate", "n_veto", "n_accepted")}
            row |= {"w_fold<0.5": float((r.w_net < 0.5).mean()), "w_fold<0.9": float((r.w_net < 0.9).mean())}
            if a != "fixed":
                row |= {k: sc[k] for k in ("n_narrow_face", "n_veto_undecidable", "n_C_high")}
                dd = d[nv_fixed]
                row["former_no_veto"] = {"n": int(nv_fixed.sum()), "narrow_face": int(dd.narrow_face.sum()),
                                         "veto_undecidable": int(dd.veto_undecidable.sum()),
                                         "decided_pass": int((dd.decidable & dd.veto_pass & ~dd.no_veto).sum()),
                                         "decided_veto": int((dd.reason == "veto").sum()),
                                         "accepted": int(dd.accepted.sum())}
            if s == "083":
                lim = np.zeros(len(d), bool)
                for u0, u1, v0, v1 in LIMBS_083:
                    lim |= (d.u.between(u0, u1) & d.v.between(v0, v1)).to_numpy()
                row["083_limb_accepted"] = int((lim & d.accepted).sum())
            img[a] = row
            for name, (_, kind, poly) in targets_for(stem).items():
                rep["targets"].setdefault(name, {"kind": kind})[a] = target_coverage(d, poly)
            for name, m in rois(stem, ctx.shape).items():
                rep["rois"].setdefault(name, {})[a] = float((r.w_net[m] < 0.5).mean())
            if s == "079":
                rep["box_rows"][a] = {"w_fold": [float(r.w_net[v, FOLD_U[0]:FOLD_U[1] + 1].min()) for v in FOLD_ROWS],
                                      "combined": [float(np.minimum(r.w_net, wt)[v, FOLD_U[0]:FOLD_U[1] + 1].min())
                                                   for v in FOLD_ROWS]}
        rep["images"][s] = img
        print(s, {a: (v["n_accepted"], v.get("n_narrow_face"), v.get("n_veto_undecidable"), v.get("n_C_high"))
                  for a, v in img.items()}, flush=True)
    (OUT / "f7_report.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
