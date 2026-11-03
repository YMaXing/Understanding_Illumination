"""TASK_GEOM_INTEGRATION G6: what changes for a default run if geom_v1 becomes the default (report only).

Current default: configs/default.yaml (run tri_veto4, the detector path). Proposed: configs/candidate_geom_v1.yaml.
Per batch: the w < 0.5 change (network px, and the full-resolution deliverable `w`), each channel's / option's marginal
area (from baseline_output/diag/geom/g5/g5.json), runtime per image (Stage B process_one timed on this machine, CPU,
no viz, into a temporary folder, the two configs alternating per image), the saved outputs per image (files, npz keys,
sizes), and the config keys that differ.
Writes baseline_output/diag/geom/g6/default_proposal.json.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.stage_b import image_stems, load_upright, process_one  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
CUR, NEW = "tri_veto4", "geom_v1"
OUT = ROOT / "baseline_output" / "diag" / "geom" / "g6"


def flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out |= flat(v, f"{pre}{k}.")
        else:
            out[f"{pre}{k}"] = v
    return out


def per_image_files(rd: Path, stem: str) -> dict:
    files = sorted(p for p in rd.glob(f"{stem}*") if p.is_file())
    viz = sorted(p for p in rd.glob(f"viz*/{stem}.png"))
    z = np.load(rd / f"{stem}.npz")
    return {"files": [p.name.replace(stem, "<stem>") for p in files] + [str(p.relative_to(rd)).replace(stem, "<stem>")
                                                                        for p in viz],
            "npz_keys": sorted(z.files), "mb_data": sum(p.stat().st_size for p in files) / 1e6,
            "mb_viz": sum(p.stat().st_size for p in viz) / 1e6}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g5 = json.loads((ROOT / "baseline_output" / "diag" / "geom" / "g5" / "g5.json").read_text())
    rep = {"per_batch": {}, "config": {}}
    for batch, root in BATCHES.items():
        rc, rn = root / "stage_b" / CUR, root / "stage_b" / NEW
        cc, cn = load_config(rc / "config_resolved.yaml"), load_config(rn / "config_resolved.yaml")
        stems = image_stems(rn)
        cnt = {k: [0, 0] for k in ("cur_net", "new_net", "cur_full", "new_full")}
        times = {CUR: [], NEW: []}
        files = {}
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for stem in stems:
                for run, rd, cfg in ((CUR, rc, cc), (NEW, rn, cn)):
                    a, _ = load_upright(cfg, rd, stem)
                    key = "cur" if run == CUR else "new"
                    cnt[f"{key}_net"][0] += int((a["w_net"] < 0.5).sum())
                    cnt[f"{key}_net"][1] += int(a["w_net"].size)
                    cnt[f"{key}_full"][0] += int((a["w"] < 0.5).sum())
                    cnt[f"{key}_full"][1] += int(a["w"].size)
                    (td / run).mkdir(exist_ok=True)
                    t0 = time.perf_counter()
                    process_one(cfg, stem, td / run)
                    times[run].append(time.perf_counter() - t0)
                    if stem == stems[0]:
                        files[run] = per_image_files(rd, stem)
        frac = {k: v[0] / v[1] for k, v in cnt.items()}
        rep["per_batch"][batch] = {
            "w_lt_0.5": {"network": {"current": frac["cur_net"], "proposed": frac["new_net"],
                                     "change": frac["new_net"] - frac["cur_net"]},
                         "deliverable": {"current": frac["cur_full"], "proposed": frac["new_full"],
                                         "change": frac["new_full"] - frac["cur_full"]}},
            "marginal": g5["marginal"][batch],
            "runtime_s_per_image": {run: {"median": float(np.median(t)), "mean": float(np.mean(t)),
                                          "min": float(np.min(t)), "max": float(np.max(t))} for run, t in times.items()},
            "outputs_first_image": files}
    cur = flat(load_config(ROOT / "configs" / "default.yaml").to_dict())
    new = flat(load_config(ROOT / "configs" / "candidate_geom_v1.yaml").to_dict())
    rep["config"] = {"changed": {k: [cur[k], new[k]] for k in sorted(cur) if k in new and cur[k] != new[k]
                                 and not k.startswith("paths.")},
                     "added": sorted(k for k in new if k not in cur)}
    (OUT / "default_proposal.json").write_text(json.dumps(rep, indent=1, default=str))
    for b, v in rep["per_batch"].items():
        print(b, json.dumps({k: v[k] for k in ("w_lt_0.5", "runtime_s_per_image")}, indent=1))
        for run, f in v["outputs_first_image"].items():
            print(" ", run, f["files"], f"{f['mb_data']:.1f} MB data, {f['mb_viz']:.1f} MB viz")
            print("    npz:", f["npz_keys"])
    print("config changed:", json.dumps(rep["config"]["changed"], indent=1, default=str))
    print("config added:", rep["config"]["added"])


if __name__ == "__main__":
    main()
