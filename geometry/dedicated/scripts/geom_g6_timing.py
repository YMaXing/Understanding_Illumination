"""TASK_GEOM_INTEGRATION G6 follow-up: Stage B runtime per image, broken down by channel and option (report only).

Per image and config (geom_v1, and the [fold, laplacian] variant geom_v1_fold_lap), on this machine's CPU:
  total   process_one (as Stage B runs it, no viz) into a temporary folder; its IO measured inside the same call by
          timing np.load (Stage A cache in), np.savez_compressed, DataFrame.to_parquet and the JSON write;
  parts   a second pass on a fresh context, timed step by step in Stage B's order:
          disparity + sky (shared), rho_f trilateral (shared by depth and fold), depth channel (incl. its
          rasterization and the sky ring), dense fit + normals (fold context), fold channel (incl. the crease rescue,
          its rasterization and the L/rho_f record field), laplacian channel (tri_veto4's detector with its own
          trilateral), combine (min) + deliverable upsample.
  options the depth channel rerun with sky_band depth (ring cost = difference) and the fold channel rerun with
          crease_rescue off (rescue cost = difference), on the same context;
  raster  the channels' rasterization re-timed from their tables (scripts/geom_g5.py depth_w / fold_w; included in
          the channel times above, reported as "of which").
Medians per batch. Writes baseline_output/diag/geom/g6/timing.json.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geom_g5 import depth_w, fold_w  # noqa: E402
import illum_baseline.stage_b as sb  # noqa: E402
from illum_baseline.config import dense_fit_params, detector_params, load_config  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.channels import combine  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.depth import DepthChannel, DepthParams  # noqa: E402
from illum_baseline.geom.fold import FoldChannel, FoldParams  # noqa: E402
from illum_baseline.geom.laplacian import LaplacianChannel  # noqa: E402
from illum_baseline.outputs import deliverable_w  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
RUNS = ("geom_v1", "geom_v1_fold_lap")
OUT = ROOT / "baseline_output" / "diag" / "geom" / "g6"


class IOTimer:
    """Accumulates wall time spent in the IO calls Stage B makes (patched for the duration of one process_one)."""

    def __init__(self):
        self.t = 0.0
        self._orig = {}

    def _wrap(self, f):
        def g(*a, **k):
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                self.t += time.perf_counter() - t0
        return g

    def __enter__(self):
        self._orig = {"load": np.load, "savez": np.savez_compressed, "parquet": pd.DataFrame.to_parquet,
                      "write_text": Path.write_text}
        np.load = self._wrap(self._orig["load"])
        np.savez_compressed = self._wrap(self._orig["savez"])
        pd.DataFrame.to_parquet = self._wrap(self._orig["parquet"])
        Path.write_text = self._wrap(self._orig["write_text"])
        return self

    def __exit__(self, *exc):
        np.load = self._orig["load"]
        np.savez_compressed = self._orig["savez"]
        pd.DataFrame.to_parquet = self._orig["parquet"]
        Path.write_text = self._orig["write_text"]


def tick(store: dict, key: str, f):
    t0 = time.perf_counter()
    out = f()
    store[key] = store.get(key, 0.0) + time.perf_counter() - t0
    return out


def parts(cfg, root: Path, stem: str) -> dict:
    t: dict[str, float] = {}
    g = cfg.geometry
    meta_a = json.loads((root / "stage_a" / f"{stem}.json").read_text())
    z = np.load(root / "stage_a" / f"{stem}.npz")
    D, sky = z["D"], z["sky_prob"].astype(np.float64)
    ctx = GeometryContext(D, sky, Intrinsics.from_stage_a(meta_a), detector_params(cfg), dense_fit_params(cfg))
    tick(t, "shared: disparity + sky", lambda: ctx.disparity)
    tick(t, "shared: rho_f trilateral", lambda: ctx.rho_f)
    res = []
    dp = DepthParams.from_dict(dataclasses.asdict(g.depth))
    fp = FoldParams.from_dict(dataclasses.asdict(g.fold))
    if "depth" in g.channels:
        rd_ = tick(t, "depth channel", lambda: DepthChannel().run(ctx, dp))
        res.append(rd_)
        tick(t, "option: depth without sky ring", lambda: DepthChannel().run(ctx, dataclasses.replace(dp, sky_band="depth")))
        c = rd_.candidates
        tick(t, "of which: depth rasterization", lambda: depth_w(c, ctx.shape, ctx.S, dp.sky_band != "depth",
                                                                  dataclasses.asdict(g.depth)))
    if "fold" in g.channels:
        tick(t, "fold context: dense fit + normals", lambda: ctx.normals)
        tick(t, "fold context: L/rho_f, s_hat_L", lambda: (ctx.L_over_rho_f, ctx.s_hat_L))
        rf = tick(t, "fold channel (incl. rescue)", lambda: FoldChannel().run(ctx, fp))
        res.append(rf)
        tick(t, "option: fold without rescue", lambda: FoldChannel().run(ctx, dataclasses.replace(fp, crease_rescue="off")))
        c = rf.candidates
        tick(t, "of which: fold rasterization", lambda: fold_w(c, ctx.shape, True))
    if "laplacian" in g.channels:
        prm = sb._channel_params(cfg, "laplacian")
        res.append(tick(t, "laplacian channel", lambda: LaplacianChannel().run(ctx, prm)))
    fr = meta_a["frames"]
    w = tick(t, "combine + deliverable upsample", lambda: combine(res, g.combine))
    tick(t, "combine + deliverable upsample", lambda: deliverable_w(w, tuple(fr["upright_WxH"]),
                                                                       fr["orientation_applied"], 0.05))
    if "depth" in g.channels:
        t["option: sky ring"] = t["depth channel"] - t.pop("option: depth without sky ring")
    if "fold" in g.channels:
        t["option: crease rescue"] = t["fold channel (incl. rescue)"] - t.pop("option: fold without rescue")
    return t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    for run in RUNS:
        rep[run] = {}
        for batch, root in BATCHES.items():
            rd = root / "stage_b" / run
            cfg = load_config(rd / "config_resolved.yaml")
            rows = []
            with tempfile.TemporaryDirectory() as td:
                for stem in sb.image_stems(rd):
                    with IOTimer() as io:
                        t0 = time.perf_counter()
                        sb.process_one(cfg, stem, Path(td))
                        total = time.perf_counter() - t0
                    p = parts(cfg, root, stem)
                    compute = sum(v for k, v in p.items() if not k.startswith(("of which", "option")))
                    rows.append({"stem": stem, "total (process_one)": total, "IO (in + out)": io.t, **p,
                                 "sum of parts": compute,
                                 "other (total - IO - parts)": total - io.t - compute})
            df = pd.DataFrame(rows)
            rep[run][batch] = {"median": df.drop(columns="stem").median().to_dict(), "per_image": rows}
    (OUT / "timing.json").write_text(json.dumps(rep, indent=1))
    pd.set_option("display.width", 200)
    for run in RUNS:
        tab = pd.DataFrame({b: v["median"] for b, v in rep[run].items()})
        print(f"\n{run} (median s per image)\n" + tab.round(3).to_string())


if __name__ == "__main__":
    main()
