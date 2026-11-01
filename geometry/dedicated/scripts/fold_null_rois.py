"""F2 follow-up 2: multiplicative vs additive null on the spot-check ROIs, plus per-image real stats (arm A).

ROIs (network px, upright):
  079 grass:   u 5-110 x v 5-110 and u 5-60 x v 360-495 (grass only; no ROI was recorded earlier; confirmed)
  079 grass sunlit: u 110-290 x v 5-85
  083 bollard: u 175-205, v 260-330 (docs/pipeline_tri_veto4.md section 5)
  070 floor:   baseline_output/diag/070/roi_mask.npy (TASK_070_STRIPES T1)
  079 box:     u 115-320, v 85-330 (accepted candidates; rows 225-260 for the lit/shaded fold)

Run from geometry/dedicated:  env -u PYTHONPATH uv run python scripts/fold_null_rois.py [--out stats.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from illum_baseline.config import detector_params, load_config  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.fold import FoldChannel  # noqa: E402

NULLS = ("multiplicative", "additive")


def rect(shape, u0, u1, v0, v1):
    m = np.zeros(shape, dtype=bool)
    m[v0:v1 + 1, u0:u1 + 1] = True
    return m


def rois(stem: str, shape) -> dict[str, np.ndarray]:
    if stem.endswith("079"):
        return {"079 grass": rect(shape, 5, 110, 5, 110) | rect(shape, 5, 60, 360, 495),
                "079 grass sunlit": rect(shape, 110, 290, 5, 85)}
    if stem.endswith("083"):
        return {"083 bollard": rect(shape, 175, 205, 260, 330)}
    if stem.endswith("070"):
        return {"070 floor": np.load(ROOT / "baseline_output/diag/070/roi_mask.npy").astype(bool)}
    return {}


def pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    p = detector_params(load_config(ROOT / "configs/candidate_tri_veto4.yaml"))
    out = {"roi": {}, "images": []}
    for f in sorted((ROOT / "baseline_output/stage_a").glob("taiwei_cui_*.npz")):
        c = np.load(f)
        meta = json.loads(f.with_suffix(".json").read_text())
        ctx = GeometryContext(c["D"], c["sky_prob"].astype(np.float64), Intrinsics.from_stage_a(meta), p)
        res = {nm: FoldChannel().run(ctx, {"null_model": nm}) for nm in NULLS}
        for name, m in rois(f.stem, ctx.shape).items():
            out["roi"][name] = {nm: float((res[nm].w_net[m] < 0.5).mean()) for nm in NULLS} | {"n_px": int(m.sum())}
        df, s = res["multiplicative"].candidates, res["multiplicative"].scalars
        v, g = df["valid"].to_numpy(), df["gate"].to_numpy()
        row = {"stem": f.stem[-3:], "lambda_raw": s["lambda_hat_raw"], "sigma_geo_deg": math.degrees(s["sigma_geo"]),
               "eps_hat": s["eps_hat"], "null_rms_q": s["null_rms_q"], "n_cand": s["n_cand"], "n_valid": s["n_valid"],
               "both_planar": float((df["planar_p"] & df["planar_m"])[v].mean()),
               "no_veto": float(df["no_veto"][g].mean()) if g.any() else float("nan"),
               "valid_normals": float(ctx.normal_valid.mean())}
        for nm in NULLS:
            d, sc = res[nm].candidates, res[nm].scalars
            col = "dtheta_gate" if nm == "multiplicative" else "dtheta_gate_add"
            dg = np.degrees(d[col][d["valid"]])
            row |= {f"{nm[:4]}_gate": sc["n_gate"] / max(sc["n_valid"], 1), f"{nm[:4]}_acc": sc["n_accepted"],
                    f"{nm[:4]}_w05": sc["frac_w_lt_0.5"], f"{nm[:4]}_w09": sc["frac_w_lt_0.9"],
                    f"{nm[:4]}_dg": [pct(dg, 10), pct(dg, 50), pct(dg, 90)]}
            if f.stem.endswith("079"):
                box = d["u"].between(115, 320) & d["v"].between(85, 330)
                row |= {f"{nm[:4]}_box_acc": int((box & d["accepted"]).sum()),
                        f"{nm[:4]}_box_rows_acc": int((box & d["v"].between(225, 260) & d["accepted"]).sum()),
                        f"{nm[:4]}_box_lost_narrow": int((box & d["gate"] & ~d["veto_pass"] & ~d["k2_clean"]).sum())}
        out["images"].append(row)
    text = json.dumps(out, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
