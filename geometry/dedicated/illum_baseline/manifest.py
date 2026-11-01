"""Per-run manifest: one row per image with detector/focal scalars and QA flags (spec M4)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FOCAL_REL_UNC_MAX = 0.15
SKY_FRAC_MAX = 0.5


def manifest_frame(rd: Path) -> pd.DataFrame:
    from .stage_b import image_stems

    rows = []
    for stem in image_stems(rd):
        m = json.loads((rd / f"{stem}.json").read_text())
        d, f, g = m["detector"], m["focal"], m["geometry"]
        rows.append({
            "stem": m["stem"], "file": m["file"],
            "stored_W": g["stored_WxH"][0], "stored_H": g["stored_WxH"][1],
            "W_n": g["W_n"], "H_n": g["H_n"], "s_x": g["s_x"], "s_y": g["s_y"],
            "orientation": g["orientation_tag"],
            **{k: d[k] for k in ("s_hat", "s_hat_raw", "s_hat_clamped", "tau", "eps", "rho_floor", "r", "R",
                                 "sky_frac", "n_bad_depth", "frac_w_lt_0.5", "frac_w_lt_0.9", "frac_w_lt_1")},
            "f_active": f["f_active"], "f_n": f["f_n"], "focal_mode": f["mode"],
            "f_per_image": f["f_per_image"], "f_pooled": f["f_pooled"],
            "pooled_meaningful": f["pooled_meaningful"],
            "focal_uncertainty": f["focal_uncertainty"], "focal_rel_uncertainty": f["focal_rel_uncertainty"],
            "vfov_deg": f["vfov_deg"], "roll_deg": f["roll_deg"], "pitch_deg": f["pitch_deg"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    lo, hi = np.percentile(df["s_hat"], [2, 98])
    w_p98 = np.percentile(df["frac_w_lt_0.5"], 98)
    df["flag_sky_frac_gt_0.5"] = df["sky_frac"] > SKY_FRAC_MAX
    df["flag_focal_rel_unc_gt_0.15"] = df["focal_rel_uncertainty"] > FOCAL_REL_UNC_MAX
    df["flag_s_hat_outside_p2_p98"] = (df["s_hat"] < lo) | (df["s_hat"] > hi)
    df["flag_bad_depth"] = df["n_bad_depth"] > 0
    df["flag_frac_w_lt_0.5_gt_p98"] = df["frac_w_lt_0.5"] > w_p98
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["n_flags"] = df[flag_cols].sum(axis=1)
    df.attrs["thresholds"] = {"s_hat_p2": lo, "s_hat_p98": hi, "frac_w_lt_0.5_p98": w_p98}
    return df


def write_manifest(rd: Path) -> Path:
    df = manifest_frame(rd)
    df.to_csv(rd / "manifest.csv", index=False)
    (rd / "manifest_thresholds.json").write_text(json.dumps(df.attrs.get("thresholds", {}), indent=2))
    try:
        path = rd / "manifest.parquet"
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError) as e:  # pyarrow missing or unusable: CSV is the fallback
        warnings.warn(f"parquet unavailable ({e}); manifest written as CSV only")
        return rd / "manifest.csv"
