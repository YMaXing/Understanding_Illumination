"""TASK_FOLD_V2 F9: per-image table of the runs fold_v2 and tri_veto4_fold_v2 (from saved outputs), with the v1 runs
fold_only / tri_veto4_fold and tri_veto4 for reference. Writes baseline_output/diag/fold_v2/f9/f9_report.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.stage_b import image_stems, load_upright  # noqa: E402

SB = ROOT / "baseline_output" / "stage_b"
OUT = ROOT / "baseline_output" / "diag" / "fold_v2" / "f9"


def load(run, stem):
    cfg = load_config(SB / run / "config_resolved.yaml")
    return load_upright(cfg, SB / run, stem)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in image_stems(SB / "fold_v2"):
        t, _ = load("tri_veto4", stem)
        wt = (t["w_net"].astype(float) < 0.5).mean()
        r = {"stem": stem[-3:]}
        for run, tag in (("fold_v2", "v2"), ("fold_only", "v1")):
            a, meta = load(run, stem)
            fo = meta["channels"]["fold"]
            d = pd.read_parquet(SB / run / f"{stem}_fold_candidates.parquet")
            w = a["fold_w_net"].astype(float)
            r |= {f"{tag}_stages": f"{fo['n_cand']}>{fo['n_valid']}>{fo['n_gate']}>{fo['n_veto']}>{fo['n_accepted']}",
                  f"{tag}_w<0.5": float((w < 0.5).mean()), f"{tag}_w<0.9": float((w < 0.9).mean()),
                  f"{tag}_depth_jump_share": float(d.loc[d.accepted, "depth_jump"].mean()) if fo["n_accepted"] else 0.0,
                  f"{tag}_t_s": fo["context_s"] + fo["runtime_s"]}
            if tag == "v2":
                r |= {"narrow_face": fo["n_narrow_face"], "veto_undecidable": fo["n_veto_undecidable"],
                      "C_high": fo["n_C_high"]}
        for run, tag in (("tri_veto4_fold_v2", "comb_v2"), ("tri_veto4_fold", "comb_v1")):
            a, _ = load(run, stem)
            w = (a["w_net"].astype(float) < 0.5).mean()
            r |= {f"{tag}_w<0.5": float(w), f"{tag}_d_vs_tri_veto4": float(w - wt)}
        rows.append(r)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    (OUT / "f9_report.json").write_text(df.to_json(orient="records", indent=1))


if __name__ == "__main__":
    main()
