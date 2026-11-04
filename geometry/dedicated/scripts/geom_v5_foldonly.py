"""TASK_GEOM_V2 wrap-up (report only): the fallback with the exclusion around relaxing fold candidates only.

Exclusion edges: the accepted fold candidates (rescued included, at the Hessian site) with their own w_c < 0.5; no depth
and no sky edges. d_excl in {2, 3, 4}, pool radius 2, band_min 2, on geom_v1_hessian's tables and weights
(scripts/geom_v5_fallback.py's functions). Reports, per setting: paired CIs vs geom_v1 (band excess, per-pixel missed
geometry, tri_fp) and the RGB-edge coverages (depth edges, fold targets, fold candidates), plus the first rule's other
constraints. Pass: band excess reduced (paired CI < 0) and every original constraint met.
Writes baseline_output/diag/geom/v2/fallback_foldonly.json.
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

from geom_score import OUT, Run, load_labels, rgb_edge_set  # noqa: E402
from geom_v5_fallback import (BASE, BATCHES, TOL, V1, SettingRun, depth_w, flat, fold_w, metrics,  # noqa: E402
                              paired)
from illum_baseline.geom.fallback import apply_fallback, exclusion_zone  # noqa: E402
from illum_baseline.stage_b import load_upright  # noqa: E402

D_EXCL = (2, 3, 4)


def main():
    lab, draws, strata, totals, masks = load_labels()
    edges = rgb_edge_set()
    base = Run(BASE)
    b1, b2 = base.stems["batch1"], base.stems["batch2"]
    pre, up = {}, {}
    for batch, stems in base.stems.items():
        cfg = base.cfg[batch]
        import dataclasses
        prm = dataclasses.asdict(cfg.geometry.depth)
        for stem in stems:
            a, meta = load_upright(cfg, BATCHES[batch] / "stage_b" / BASE, stem)
            dc = pd.read_parquet(BATCHES[batch] / "stage_b" / BASE / f"{stem}_depth_candidates.parquet")
            fc = pd.read_parquet(BATCHES[batch] / "stage_b" / BASE / f"{stem}_fold_candidates.parquet")
            shape, S = a["w_net"].shape, a["sky_mask"].astype(bool)
            core = np.minimum(depth_w(dc, shape, S, 2, prm), fold_w(fc, shape, 2))
            pre[stem] = {"core": core, "lap": a["laplacian_w_net"].astype(np.float64),
                         "zones": {d: exclusion_zone(None, fc, shape, d) for d in D_EXCL}}
            up[stem] = tuple(meta["geometry"]["upright_WxH"])
    v1 = metrics(Run(V1), lab, draws, strata, totals, masks, edges, b1, b2)
    ref = flat("geom_v1", v1)
    rows, rep = [], {"geom_v1": {k: v for k, v in ref.items()}, "settings": {}}
    for d in D_EXCL:
        name = f"foldonly_d{d}_r2_b2"
        net = {s: (np.minimum(p["core"], apply_fallback(p["lap"], p["zones"][d])), p["core"]) for s, p in pre.items()}
        m = metrics(SettingRun(name, base, net, up), lab, draws, strata, totals, masks, edges, b1, b2)
        pr = paired(m["_U"], v1["_U"], draws, strata, totals)
        f = flat(name, m)
        ok = {"missed": f["missed/px"] <= ref["missed/px"] + TOL,
              "step": f["cov step"] >= ref["cov step"] - TOL, "limb": f["cov limb"] >= ref["cov limb"] - TOL,
              "fold": f["cov fold"] >= ref["cov fold"] - TOL, "rgb depth": f["rgb depth"] >= ref["rgb depth"] - TOL,
              "rgb fold targets": f["rgb fold targets"] >= ref["rgb fold targets"] - TOL,
              "rgb fold cand": f["rgb fold cand"] >= ref["rgb fold cand"] - TOL,
              "band excess reduced (paired CI < 0)": pr["band excess"]["ci_hi"] < 0}
        zone = float(np.mean([p["zones"][d].mean() for p in pre.values()]))
        rep["settings"][name] = {"flat": f, "paired": pr, "constraints": ok, "passes": all(ok.values()),
                                 "zone_share": zone}
        rows.append({"setting": name, "zone": zone, "missed/px": f["missed/px"], "band excess": f["band excess"],
                     "tri_fp": f["tri_fp"],
                     **{f"Δ {k} [CI]": f"{v['diff']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]" for k, v in pr.items()},
                     "rgb depth": f["rgb depth"], "rgb fold targets": f["rgb fold targets"],
                     "rgb fold cand": f["rgb fold cand"], "cov step/limb/fold": f"{f['cov step']:.3f}/{f['cov limb']:.3f}/{f['cov fold']:.3f}",
                     "texture class": f["texture class"], "passes": all(ok.values()),
                     "fails": [k for k, v in ok.items() if not v]})
    (OUT / "fallback_foldonly.json").write_text(json.dumps(rep, indent=1, default=float))
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 80)
    print("geom_v1:", {k: round(ref[k], 4) for k in ("missed/px", "band excess", "tri_fp", "rgb depth",
                                                    "rgb fold targets", "rgb fold cand", "cov step", "cov fold")})
    print(pd.DataFrame(rows).set_index("setting").T.to_string())


if __name__ == "__main__":
    main()
