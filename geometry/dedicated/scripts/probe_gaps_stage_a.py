"""Gap-class probe, Stage A (report only): DA3 at network long side 1008 for the images of the probed units.

Cached alongside the 504 cache, never replacing it: <batch out_dir>/stage_a/res1008/<stem>.npz + .json (the
paths.stage_a_variant layout). Input prepared as Stage A does (network.long_side 1008), DA3 process_res 1008; GeoCalib
not rerun (its block copied from the 504 cache, focal rescaled with the new scale factors).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_res1008_stage_a import timed  # noqa: E402
from illum_baseline.config import apply_sets, load_config  # noqa: E402
from illum_baseline.depth import DepthEstimator  # noqa: E402
from illum_baseline.geometry import focal_at_network  # noqa: E402
from illum_baseline.stage_a import prepare, resolve_device  # noqa: E402

BATCH_ROOT = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
VARIANT = "res1008"


def probe_stems() -> dict[str, str]:
    u = pd.read_csv(ROOT / "baseline_output" / "diag" / "geom" / "g4" / "units.csv")
    t = u[u.direction == "tri-only"]
    sel = t[(t.bucket == "gap_unresolved") | (t.stage == "component") | (t.bucket == "tri_fp") | t.id.isin([15, 37, 92])]
    stems = dict(zip(sel.stem, sel.batch))
    for s in ("taiwei_cui_070", "taiwei_cui_079", "taiwei_cui_083"):   # texture ROIs
        stems[s] = "batch1"
    stems["acharya_mrudula_068"] = "batch2"                             # laps
    return dict(sorted(stems.items()))


def main():
    est, timing = None, {}
    for stem, batch in probe_stems().items():
        sa = BATCH_ROOT[batch] / "stage_a"
        out = sa / VARIANT
        out.mkdir(parents=True, exist_ok=True)
        if (out / f"{stem}.npz").exists() and (out / f"{stem}.json").exists():
            continue
        cfg = apply_sets(load_config(sa / "config_resolved.yaml"), ["network.long_side=1008", "da3.process_res=1008"])
        meta504 = json.loads((sa / f"{stem}.json").read_text())
        if est is None:
            est = DepthEstimator(cfg.da3.model_id, 1008, cfg.da3.process_res_method, resolve_device(cfg.device), cfg.seed)
        item = prepare(cfg, cfg.data_dir / meta504["file"])
        pred, t, mem_a, mem_r = timed(est, item["net"])
        meta = dict(item["meta"])
        meta.update({"stem": stem, "file": meta504["file"], "geocalib": meta504["geocalib"], "da3": pred["record"]})
        gc, nw = meta504["geocalib"], meta["network"]
        fo = dict(meta504["focal"])
        fo["f_n"] = focal_at_network(gc["fx"], gc["fy"], nw["s_x"], nw["s_y"])
        fo["note"] = "probe: GeoCalib block copied from the 504 cache, focal rescaled to this network size"
        meta["focal"] = fo
        meta["probe"] = {"network_long_side": 1008, "source_cache": str(sa / f"{stem}.json")}
        arrays = {"D": pred["D"]} | ({"sky_prob": pred["sky_prob"]} if pred["sky_prob"] is not None else {})
        np.savez_compressed(out / f"{stem}.npz", **arrays)
        (out / f"{stem}.json").write_text(json.dumps(meta, indent=1))
        timing[stem] = {"da3_s": t, "peak_alloc_MiB": mem_a, "peak_reserved_MiB": mem_r}
        print(stem, timing[stem], flush=True)
    torch.cuda.empty_cache()
    p = ROOT / "baseline_output" / "diag" / "geom" / "gaps1008"
    p.mkdir(parents=True, exist_ok=True)
    (p / "stage_a_timing.json").write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
