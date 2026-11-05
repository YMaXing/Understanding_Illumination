"""Resolution probe (report only, no Stage B change, no reference): DA3 at network long side 1008 on five images.

For acharya_mrudula_068 / 069 / 070 (batch2) and taiwei_cui_079 / 083 (batch1): the network input is prepared exactly as
Stage A does (illum_baseline.stage_a.prepare) with network.long_side 1008 (short side to a multiple of 14), DA3 runs with
process_res 1008 (upper_bound_resize: a no-op on that input), and GeoCalib is not rerun: its block is copied from the
image's 504 cache and the focal rescaled to the new network size (f_n = focal_at_network(fx, fy, s_x, s_y)). DA3 is also
rerun at 504 on the same input path, for runtime / memory on equal terms and as a check against the cached D.
Runtime: one warm-up pass, then the median of 3 timed passes (CUDA synchronized); memory: peak allocated / reserved.
Writes baseline_output/diag/geom/res1008/stage_a/<stem>.npz + .json (the Stage A cache format) and timing.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from illum_baseline.config import apply_sets, load_config  # noqa: E402
from illum_baseline.depth import DepthEstimator  # noqa: E402
from illum_baseline.geometry import focal_at_network  # noqa: E402
from illum_baseline.stage_a import prepare, resolve_device  # noqa: E402

STEMS = {"acharya_mrudula_068": "baseline_output_batch2", "acharya_mrudula_069": "baseline_output_batch2",
         "acharya_mrudula_070": "baseline_output_batch2", "taiwei_cui_079": "baseline_output",
         "taiwei_cui_083": "baseline_output"}
OUT = ROOT / "baseline_output" / "diag" / "geom" / "res1008"


def timed(depth: DepthEstimator, net: np.ndarray, n: int = 3) -> tuple[dict, float, float, float]:
    torch.cuda.synchronize()
    depth.predict([net])                                   # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = depth.predict([net])[0]
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return pred, float(np.median(ts)), torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20


def main():
    (OUT / "stage_a").mkdir(parents=True, exist_ok=True)
    timing = {}
    est = {}
    for res in (504, 1008):
        for stem, out_dir in STEMS.items():
            sa = ROOT / out_dir / "stage_a"
            cfg = load_config(sa / "config_resolved.yaml")
            cfg = apply_sets(cfg, [f"network.long_side={res}", f"da3.process_res={res}"])
            meta504 = json.loads((sa / f"{stem}.json").read_text())
            if res not in est:
                est[res] = DepthEstimator(cfg.da3.model_id, res, cfg.da3.process_res_method, resolve_device(cfg.device),
                                          cfg.seed)
            item = prepare(cfg, cfg.data_dir / meta504["file"])
            net = item["net"]
            pred, t, mem_a, mem_r = timed(est[res], net)
            rec = {"net_HxW": list(net.shape[:2]), "da3_s": t, "peak_alloc_MiB": mem_a, "peak_reserved_MiB": mem_r}
            if res == 504:
                Dc = np.load(sa / f"{stem}.npz")["D"]
                rec["max_abs_diff_vs_cache"] = float(np.abs(pred["D"].astype(np.float64) - Dc).max())
                rec["max_rel_diff_vs_cache"] = float(np.max(np.abs(pred["D"] / Dc - 1)))
            else:
                meta = dict(item["meta"])
                meta["stem"] = stem
                meta["file"] = meta504["file"]
                meta["geocalib"] = meta504["geocalib"]           # same GeoCalib, not rerun
                meta["da3"] = pred["record"]
                gc, nw = meta504["geocalib"], meta["network"]
                fo = dict(meta504["focal"])
                fo["f_n"] = focal_at_network(gc["fx"], gc["fy"], nw["s_x"], nw["s_y"])
                fo["note"] = "probe: GeoCalib block copied from the 504 cache, focal rescaled to this network size"
                meta["focal"] = fo
                meta["probe"] = {"network_long_side": res, "source_cache": str(sa / f"{stem}.json")}
                arrays = {"D": pred["D"]}
                if pred["sky_prob"] is not None:
                    arrays["sky_prob"] = pred["sky_prob"]
                np.savez_compressed(OUT / "stage_a" / f"{stem}.npz", **arrays)
                (OUT / "stage_a" / f"{stem}.json").write_text(json.dumps(meta, indent=1))
                rec["f_n"] = fo["f_n"]
                rec["f_n_504"] = meta504["focal"]["f_n"]
            timing.setdefault(stem, {})[str(res)] = rec
            print(stem, res, rec, flush=True)
        del est[res]
        torch.cuda.empty_cache()
    (OUT / "timing.json").write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
