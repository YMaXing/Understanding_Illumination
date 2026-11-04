"""TASK_GEOM_V2 wrap-up: what changes for a default run if geom_v1_1 replaces geom_v1 (report only; the G6 table).

Per batch: the w < 0.5 change (deliverable and network resolution), each channel's marginal w < 0.5 area (the area that
exists only because of it: w < 0.5 while the min over the other channels is >= 0.5) in both runs, the area the re-siting
adds and removes (combined w < 0.5 in one run and not the other), runtime per image (from the scorer's --timing pass,
baseline_output/diag/geom/v1_1/score.json), the saved outputs (files, npz keys, candidate-table columns, JSON scalars) and
the config keys that differ. Writes baseline_output/diag/geom/v1_1/default_proposal.json.
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

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
CUR, NEW = "geom_v1", "geom_v1_1"
OUT = ROOT / "baseline_output" / "diag" / "geom" / "v1_1"


def flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out |= flat(v, f"{pre}{k}.") if isinstance(v, dict) else {f"{pre}{k}": v}
    return out


def main():
    sc = json.loads((OUT / "score.json").read_text())
    rep = {"per_batch": {}}
    for batch, root in BATCHES.items():
        cfg = {r: load_config(root / "stage_b" / r / "config_resolved.yaml") for r in (CUR, NEW)}
        acc = {k: 0 for k in ("px_net", "px_full", "cur_net", "new_net", "cur_full", "new_full", "added", "removed")}
        marg = {r: {"depth": 0, "fold": 0, "laplacian": 0, "w_lt_0.5": 0} for r in (CUR, NEW)}
        for stem in image_stems(root / "stage_b" / NEW):
            A = {r: load_upright(cfg[r], root / "stage_b" / r, stem)[0] for r in (CUR, NEW)}
            for r in (CUR, NEW):
                wd, wf, wl = (A[r][k].astype(np.float64) for k in ("depth_w_net", "fold_w_net", "laplacian_w_net"))
                lo = A[r]["w_net"] < 0.5
                marg[r]["w_lt_0.5"] += int(lo.sum())
                marg[r]["depth"] += int((lo & (np.minimum(wf, wl) >= 0.5)).sum())
                marg[r]["fold"] += int((lo & (np.minimum(wd, wl) >= 0.5)).sum())
                marg[r]["laplacian"] += int((lo & (np.minimum(wd, wf) >= 0.5)).sum())
            c, n = A[CUR]["w_net"] < 0.5, A[NEW]["w_net"] < 0.5
            acc["px_net"] += c.size
            acc["cur_net"] += int(c.sum())
            acc["new_net"] += int(n.sum())
            acc["added"] += int((n & ~c).sum())
            acc["removed"] += int((c & ~n).sum())
            cf, nf = A[CUR]["w"] < 0.5, A[NEW]["w"] < 0.5
            acc["px_full"] += cf.size
            acc["cur_full"] += int(cf.sum())
            acc["new_full"] += int(nf.sum())
        rt = {r: sc[r]["runtime"][batch]["median_s"] for r in (CUR, NEW)}
        rep["per_batch"][batch] = {
            "w_lt_0.5": {"deliverable": [acc["cur_full"] / acc["px_full"], acc["new_full"] / acc["px_full"]],
                         "network": [acc["cur_net"] / acc["px_net"], acc["new_net"] / acc["px_net"]]},
            "resiting_added_px": acc["added"], "resiting_removed_px": acc["removed"], "network_px": acc["px_net"],
            "marginal": marg, "runtime_s_per_image": rt}
    stem = image_stems(BATCHES["batch1"] / "stage_b" / NEW)[0]
    files = {r: sorted(p.name.replace(stem, "<stem>") for p in (BATCHES["batch1"] / "stage_b" / r).glob(f"{stem}*"))
             for r in (CUR, NEW)}
    keys = {r: sorted(np.load(BATCHES["batch1"] / "stage_b" / r / f"{stem}.npz").files) for r in (CUR, NEW)}
    cols = {r: list(pd.read_parquet(BATCHES["batch1"] / "stage_b" / r / f"{stem}_fold_candidates.parquet").columns)
            for r in (CUR, NEW)}
    js = {r: json.loads((BATCHES["batch1"] / "stage_b" / r / f"{stem}.json").read_text()) for r in (CUR, NEW)}
    rep["outputs"] = {"files_same": files[CUR] == files[NEW], "npz_keys_same": keys[CUR] == keys[NEW],
                      "new_fold_columns": [c for c in cols[NEW] if c not in cols[CUR]],
                      "new_fold_scalars": sorted(set(js[NEW]["channels"]["fold"]) - set(js[CUR]["channels"]["fold"])),
                      "new_fold_params": sorted(set(js[NEW]["params"]["geometry"]["fold"]) -
                                                set(js[CUR]["params"]["geometry"]["fold"]))}
    a = flat(load_config(ROOT / "configs" / "candidate_geom_v1.yaml").to_dict())
    b = flat(load_config(ROOT / "configs" / "candidate_geom_v1_1.yaml").to_dict())
    rep["config_changed"] = {k: [a[k], b[k]] for k in sorted(a) if k in b and a[k] != b[k] and not k.startswith("paths.")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "default_proposal.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
