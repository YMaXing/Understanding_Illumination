"""Resolution probe extra (report only): DA3 at 1008, downsampled to the 504 grid (2 x 2 block mean of the metric
disparity 1 / Z, then canonical D = 300 Z / f_n at 504; sky score block-mean), run through the same 504 measures as
scripts/probe_res1008.py (risers, laps, fold acceptance, ripple, normal noise). Writes res1008/probe_down.json."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import probe_res1008 as P  # noqa: E402
from geom_v1_1_laps import shadow_069  # noqa: E402
from geom_w2_hysteresis import lap_lines  # noqa: E402
from illum_baseline.config import dense_fit_params, detector_params, load_config  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402


def block(a):
    H, W = a.shape
    return a.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))


def main():
    lines = lap_lines()
    _, _, _, sh, band, _ = shadow_069(None)
    base = json.loads((P.OUT / "probe.json").read_text())
    rep = {}
    for stem, root in P.STEMS.items():
        batch_root = ROOT / root
        meta = json.loads((batch_root / "stage_a" / f"{stem}.json").read_text())
        m1008 = json.loads((P.OUT / "stage_a" / f"{stem}.json").read_text())
        z = np.load(P.OUT / "stage_a" / f"{stem}.npz")
        Z1008 = m1008["focal"]["f_n"] * z["D"].astype(np.float64) / 300.0
        Z = 1.0 / block(1.0 / Z1008)
        D = (300.0 * Z / meta["focal"]["f_n"]).astype(np.float32)
        sky = block(z["sky_prob"].astype(np.float64)) if "sky_prob" in z else None
        cfg = load_config(batch_root / "stage_b" / "geom_v1_1" / "config_resolved.yaml")
        K = Intrinsics.from_stage_a(meta)
        ctx = GeometryContext(D, sky, K, detector_params(cfg), dense_fit_params(cfg))
        Zm = meta["focal"]["f_n"] * D.astype(np.float64) / 300.0
        r = {"Z_median_m": float(np.median(Zm))}
        fold_acc = None
        if stem.endswith(("068", "069", "070")):
            import dataclasses

            from scipy.spatial import cKDTree

            from illum_baseline.geom.fold import FoldChannel, FoldParams
            fr = FoldChannel().run(ctx, FoldParams.from_dict(dataclasses.asdict(cfg.geometry.fold)))
            acc = fr.candidates[fr.candidates.accepted]
            r["fold"] = {"lambda_hat": fr.scalars["lambda_hat"], "n_cand": fr.scalars["n_cand"],
                         "n_accepted": fr.scalars["n_accepted"]}
            tree = cKDTree(acc[["u0", "v0"]].to_numpy())

            def fold_acc(x0, tol, tree=tree):
                return np.isfinite(tree.query(x0, distance_upper_bound=tol)[0]).astype(float)
        if stem.endswith("070"):
            r["risers"] = P.risers(ctx, K, Zm, 1)
            for name, vf in (("middle upper", 394), ("middle lower", 408), ("top upper", 353), ("bottom lower", 470)):
                u504 = np.arange(100, 281, 1.0)
                x0 = np.stack([u504, np.full(len(u504), float(vf))], -1)
                r["risers"].setdefault("fold_accepted_share", {})[name] = float(fold_acc(x0, 2.0).mean())
        if stem in lines:
            conv504 = [pl["convexity"] for pl in base[stem]["504"]["laps"]["per_line"]]
            r["laps"] = P.laps(stem, cfg, ctx, K, Zm, 1, lines[stem], (sh, band) if stem.endswith("069") else None,
                               conv504, fold_acc)
        r["ripple"] = P.ripple(cfg, ctx, D.astype(np.float64), sky, 1)
        r["normal_noise"] = P.noise(stem, ctx, 1)
        rep[stem] = r
        print(stem, "done", flush=True)
        del ctx, fold_acc
        import gc
        gc.collect()
    (P.OUT / "probe_down.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
