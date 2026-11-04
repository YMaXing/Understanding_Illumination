"""TASK_GEOM_V2 V2b check: hessian_peak sites with |t| > 2 px and same-sign neighbours (diagnostic only).

For each accepted fold candidate with a valid site in a run (default geom_v1_hessian): the other accepted fold
centrelines along its normal: accepted candidates j with |(x_j - x0) . tau| <= 1 px (tau = e1 rotated by 90 deg) and
1.5 <= |(x_j - x0) . e1| <= 6 px. Reports the share of valid candidates with |t| > 2 px, how many of those have such a
neighbour, and how many land within 1 px of it (a jump). Writes baseline_output/diag/geom/v2/site_neighbours_<run>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
OUT = ROOT / "baseline_output" / "diag" / "geom" / "v2"


def neighbours(fc: pd.DataFrame, lat_max=1.0, s_min=1.5, s_max=6.0):
    a = fc[fc["accepted"]].reset_index(drop=True)
    x = a[["u0", "v0"]].to_numpy()
    e = a[["e1_u", "e1_v"]].to_numpy()
    tau = np.stack([-e[:, 1], e[:, 0]], -1)
    tree = cKDTree(x)
    nb = [[] for _ in range(len(a))]
    for i, js in enumerate(tree.query_ball_point(x, s_max + lat_max)):
        d = x[js] - x[i]
        s = d @ e[i]
        lat = np.abs(d @ tau[i])
        ok = (lat <= lat_max) & (np.abs(s) >= s_min) & (np.abs(s) <= s_max)
        nb[i] = s[ok].tolist()
    return a, nb


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "geom_v1_hessian"
    tot = {"valid": 0, "t_gt_2": 0, "t_gt_2_with_nb": 0, "jump": 0, "valid_with_nb": 0}
    for batch, root in BATCHES.items():
        rd = root / "stage_b" / run
        for p in sorted(rd.glob("*_fold_candidates.parquet")):
            a, nb = neighbours(pd.read_parquet(p))
            t = a["site_t"].to_numpy()
            v = a["site_valid"].to_numpy()
            for i in np.nonzero(v)[0]:
                tot["valid"] += 1
                has = bool(nb[i])
                tot["valid_with_nb"] += has
                if abs(t[i]) > 2:
                    tot["t_gt_2"] += 1
                    if has:
                        tot["t_gt_2_with_nb"] += 1
                        tot["jump"] += bool(np.min(np.abs(np.asarray(nb[i]) - t[i])) <= 1.0)
    rep = tot | {"share_t_gt_2": tot["t_gt_2"] / tot["valid"],
                 "share_of_t_gt_2_with_nb": tot["t_gt_2_with_nb"] / max(tot["t_gt_2"], 1),
                 "share_of_t_gt_2_jump": tot["jump"] / max(tot["t_gt_2"], 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"site_neighbours_{run}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
