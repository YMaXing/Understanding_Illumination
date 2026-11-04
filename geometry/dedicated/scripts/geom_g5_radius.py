"""TASK_GEOM_INTEGRATION G5 follow-up (report only): geom_v1 at geometry.laplacian.pool_radius r in {1, 2, 3}.

w_r = min(depth, fold, Laplacian_r), the depth and fold w from the saved geom_v1_r3 run, the Laplacian channel rerun from
the Stage A cache with pool_radius = r (r = 3 reproduces the saved laplacian_w_net and w_net; checked).
Reports: the G1 ROI table (scripts/geom_eval.py: per class share of w < 0.5, median w, polyline coverage; batch1) and the
per-batch fraction of network px with w < 0.5 (and < 0.9).
Rule (review): r = 2 if its step, limb and fold coverage are each within 0.02 of r = 3; otherwise 3.
Writes baseline_output/diag/geom/g5/radius.json.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geom_eval import CLASSES, roi_list, roi_metrics  # noqa: E402
from illum_baseline.config import dense_fit_params, detector_params, load_config  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.laplacian import LaplacianChannel  # noqa: E402
from illum_baseline.stage_b import image_stems, load_upright  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
RUN = "geom_v1_r3"   # the r = 3 run (the depth and fold channels do not depend on the pooling radius)
RADII = (1, 2, 3)
OUT = ROOT / "baseline_output" / "diag" / "geom" / "g5"


def main():
    frac = {r: {} for r in RADII}
    maps = {}  # (r, stem suffix) -> w for batch1
    checks = []
    for batch, root in BATCHES.items():
        rd = root / "stage_b" / RUN
        cfg = load_config(rd / "config_resolved.yaml")
        base = dataclasses.asdict(detector_params(cfg))
        cnt = {r: [0, 0, 0] for r in RADII}
        for stem in image_stems(rd):
            a, _ = load_upright(cfg, rd, stem)
            wd, wf, wl = (a[k].astype(np.float64) for k in ("depth_w_net", "fold_w_net", "laplacian_w_net"))
            meta_a = json.loads((root / "stage_a" / f"{stem}.json").read_text())
            z = np.load(root / "stage_a" / f"{stem}.npz")
            ctx = GeometryContext(z["D"], z["sky_prob"].astype(np.float64), Intrinsics.from_stage_a(meta_a),
                                  detector_params(cfg), dense_fit_params(cfg))
            for r in RADII:
                lr = LaplacianChannel().run(ctx, base | {"pool_radius": r}).w_net
                w = np.minimum.reduce([wd, wf, lr])
                if r == 3:
                    checks.append(bool(np.array_equal(lr.astype(np.float16), a["laplacian_w_net"]) and
                                       np.array_equal(w.astype(np.float16), a["w_net"])))
                cnt[r][0] += int((w < 0.5).sum())
                cnt[r][1] += int((w < 0.9).sum())
                cnt[r][2] += int(w.size)
                if batch == "batch1":
                    maps[(r, stem[-3:])] = w
        for r in RADII:
            frac[r][batch] = {"w_lt_0.5": cnt[r][0] / cnt[r][2], "w_lt_0.9": cnt[r][1] / cnt[r][2]}

    classes, rois = [], []
    for r in RADII:
        pooled = {}
        for roi in roi_list():
            m = roi_metrics(maps[(r, roi["stem"])], roi)
            rois.append({"r": r, "roi": roi["name"], "class": roi["class"], "legacy": bool(roi.get("legacy", False)),
                         **{k: v for k, v in m.items() if not k.startswith("_")}})
            if roi.get("legacy"):
                continue
            p = pooled.setdefault(roi["class"], {"vals": [], "cov": []})
            p["vals"].append(m["_vals"])
            if m["_cov"] is not None:
                p["cov"].append(m["_cov"])
        for cls, p in pooled.items():
            v = np.concatenate(p["vals"])
            classes.append({"r": r, "class": cls, "share_w<0.5": float((v < 0.5).mean()),
                            "median_w": float(np.median(v)),
                            "coverage": float(np.concatenate(p["cov"]).mean()) if p["cov"] else None})
    C = pd.DataFrame(classes)
    cov = C.set_index(["class", "r"])["coverage"]
    within = {c: abs(cov[(c, 2)] - cov[(c, 3)]) <= 0.02 for c in ("step", "limb", "fold")}
    choice = 2 if all(within.values()) else 3
    rep = {"classes": classes, "rois": rois, "w_fraction": frac, "checks_r3_bit_identical": checks,
           "rule": {"coverage_within_0.02_of_r3": within, "pool_radius": choice}}
    (OUT / "radius.json").write_text(json.dumps(rep, indent=1, default=float))
    C["class"] = pd.Categorical(C["class"], CLASSES, ordered=True)
    pd.set_option("display.width", 250)
    print(C.pivot_table(index="class", columns="r", values=["share_w<0.5", "coverage", "median_w"],
                        observed=False).round(3).to_string())
    print("\nw fractions:", json.dumps(frac, indent=1))
    print("r=3 bit-identical:", all(checks), f"({sum(checks)}/{len(checks)})")
    print("rule:", rep["rule"])


if __name__ == "__main__":
    main()
