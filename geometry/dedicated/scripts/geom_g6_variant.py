"""TASK_GEOM_INTEGRATION G6 follow-up (report only, no reference): the [fold, laplacian] variant geom_v1_fold_lap vs geom_v1.

Per batch: the share of w < 0.5 (network w_net and the full-resolution deliverable w); the G1 ROI table (batch1,
scripts/geom_eval.py) for geom_v1 and the variant; the per-pixel difference w_variant - w_geom_v1 (>= 0 by construction:
the variant drops one term of the min), on w_net and on w: share of pixels that differ (> 0, > 0.01, > 0.1), mean and
max, the pixels whose w < 0.5 status changes; and a check that the variant's fold and Laplacian w equal geom_v1's.
Runtime: scripts/geom_g6_timing.py. Writes baseline_output/diag/geom/g6/variant.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geom_eval import CLASSES, evaluate  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.stage_b import image_stems, load_upright  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
V1, VAR = "geom_v1", "geom_v1_fold_lap"
OUT = ROOT / "baseline_output" / "diag" / "geom" / "g6"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"per_batch": {}, "per_image": []}
    for batch, root in BATCHES.items():
        acc = {k: 0 for k in ("n_net", "n_full", "v1_net", "var_net", "v1_full", "var_full", "diff0_net", "diff001_net",
                              "diff01_net", "flip_net", "diff0_full", "diff001_full", "diff01_full", "flip_full")}
        sums = {"net": 0.0, "full": 0.0}
        mx = {"net": 0.0, "full": 0.0}
        same = []
        for stem in image_stems(root / "stage_b" / VAR):
            arr = {}
            for run in (V1, VAR):
                rd = root / "stage_b" / run
                arr[run], _ = load_upright(load_config(rd / "config_resolved.yaml"), rd, stem)
            same.append(all(np.array_equal(arr[V1][k], arr[VAR][k]) for k in ("fold_w_net", "laplacian_w_net")))
            row = {"batch": batch, "stem": stem}
            for res, key in (("net", "w_net"), ("full", "w")):
                a, b = arr[V1][key].astype(np.float64), arr[VAR][key].astype(np.float64)
                d = b - a
                acc[f"n_{res}"] += a.size
                acc[f"v1_{res}"] += int((a < 0.5).sum())
                acc[f"var_{res}"] += int((b < 0.5).sum())
                acc[f"diff0_{res}"] += int((d > 0).sum())
                acc[f"diff001_{res}"] += int((d > 0.01).sum())
                acc[f"diff01_{res}"] += int((d > 0.1).sum())
                acc[f"flip_{res}"] += int(((a < 0.5) != (b < 0.5)).sum())
                sums[res] += float(d.sum())
                mx[res] = max(mx[res], float(d.max()))
                row[f"{res}: share differing"] = float((d > 0).mean())
                row[f"{res}: max diff"] = float(d.max())
                row[f"{res}: min diff"] = float(d.min())
            rep["per_image"].append(row)
        out = {"fold_and_laplacian_w_identical": all(same)}
        for res in ("net", "full"):
            n = acc[f"n_{res}"]
            out[res] = {"w_lt_0.5 geom_v1": acc[f"v1_{res}"] / n, "w_lt_0.5 variant": acc[f"var_{res}"] / n,
                        "share diff > 0": acc[f"diff0_{res}"] / n, "share diff > 0.01": acc[f"diff001_{res}"] / n,
                        "share diff > 0.1": acc[f"diff01_{res}"] / n, "px w<0.5 status changes": acc[f"flip_{res}"],
                        "share w<0.5 status changes": acc[f"flip_{res}"] / n, "mean diff": sums[res] / n,
                        "max diff": mx[res]}
        rep["per_batch"][batch] = out
    ev = evaluate([V1, VAR])
    rep["g1_classes"] = ev["classes"]
    (OUT / "variant.json").write_text(json.dumps(rep, indent=1))
    cls = pd.DataFrame(ev["classes"])
    comb = cls[cls["w"] == "combined"].copy()
    comb["class"] = pd.Categorical(comb["class"], CLASSES, ordered=True)
    pd.set_option("display.width", 220)
    print(comb.pivot_table(index="class", columns="run", values=["share_w<0.5", "coverage", "median_w"],
                           observed=False).round(4).to_string())
    print(json.dumps(rep["per_batch"], indent=1))
    pi = pd.DataFrame(rep["per_image"])
    print(pi[["batch", "stem", "net: share differing", "net: max diff", "net: min diff"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
