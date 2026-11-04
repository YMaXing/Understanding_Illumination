"""TASK_GEOM_INTEGRATION G6: audit numbers for the geom_v1 limitations (report only).

The Laplacian's band excess and texture in geom_v1 (pooling radius 2), from the G4 labels (audit_split/): the share of
the tri-only area that is labelled tri_band_excess / tri_fp / texture (any bucket) AND relaxed by geom_v1, per pixel:
the indicator of a draw is its unit's fraction of pixels with geom_v1 w < 0.5 (scripts/geom_g5.py units_v1.csv), so the
stratified pps estimator (scripts/geom_g4.py) estimates the relaxed area. 95 % CIs: stratified bootstrap over draws.
Also the geom-only false-positive share and the batch1 texture ROI shares (scripts/geom_g5_radius.py, r = 2).
Writes baseline_output/diag/geom/g6/limits.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from geom_g4 import AUD, Estimator  # noqa: E402

G5 = ROOT / "baseline_output" / "diag" / "geom" / "g5"
OUT = ROOT / "baseline_output" / "diag" / "geom" / "g6"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    U = pd.read_csv(G5 / "units_v1.csv")
    draws = pd.read_csv(AUD / "audit_draws.csv")
    strata = pd.read_csv(AUD / "audit_strata.csv")
    totals = pd.DataFrame(json.loads((AUD / "audit_totals.json").read_text())["per_image"])
    rep = {}
    for direction in ("tri-only", "geom-only"):
        d = draws[draws["direction"] == direction].merge(U, on=["id", "stem", "direction"], how="left").reset_index(
            drop=True)
        t = totals[totals["direction"] == direction]
        E = Estimator(d, strata[strata["direction"] == direction], float(t["area_px"].sum()),
                      float(t["image_px"].sum()), np.random.default_rng(0))
        f = d["frac_v1"].to_numpy()
        if direction == "tri-only":
            ind = {"tri_band_excess": (d["bucket"] == "tri_band_excess").to_numpy(),
                   "tri_band_excess relaxed": (d["bucket"] == "tri_band_excess").to_numpy() * f,
                   "tri_fp": (d["bucket"] == "tri_fp").to_numpy(),
                   "tri_fp relaxed": (d["bucket"] == "tri_fp").to_numpy() * f,
                   "texture (label)": (d["label"] == "texture").to_numpy(),
                   "texture (label) relaxed": (d["label"] == "texture").to_numpy() * f,
                   "missed true geometry": d["bucket"].isin(["gap_fixable", "gap_unresolved"]).to_numpy() * (1 - f)}
        else:
            ind = {"geom_fp": (d["bucket"] == "geom_fp").to_numpy(), "geom_band_excess":
                   (d["bucket"] == "geom_band_excess").to_numpy()}
        df, _ = E.run(ind)
        df.insert(0, "direction", direction)
        rep[direction] = df.to_dict(orient="records")
    rad = json.loads((G5 / "radius.json").read_text())
    rep["texture_rois_r2"] = {r["roi"]: r["share_w<0.5"] for r in rad["rois"] if r["r"] == 2 and r["class"] == "texture"
                              and not r["legacy"]}
    rep["texture_class_r2"] = [c for c in rad["classes"] if c["r"] == 2 and c["class"] == "texture"][0]["share_w<0.5"]
    (OUT / "limits.json").write_text(json.dumps(rep, indent=1, default=float))
    pd.set_option("display.width", 200)
    for direction in ("tri-only", "geom-only"):
        print(pd.DataFrame(rep[direction]).round(4).to_string(index=False))
    print("texture ROIs r=2:", {k: round(v, 4) for k, v in rep["texture_rois_r2"].items()}, "class",
          round(rep["texture_class_r2"], 4))


if __name__ == "__main__":
    main()
