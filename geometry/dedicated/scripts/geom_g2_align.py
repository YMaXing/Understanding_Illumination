"""TASK_GEOM_INTEGRATION G2(a): is the DA3-vs-RGB shift stable? (diagnostic only; reads RGB, changes nothing)

Strong accepted depth edges of geom_v0 (batch1 + batch2): accepted, both sides planar at m = 8, r > 0.2, significance
(the run's active null, additive: signif_add) > 2z, not sky. RGB edge: the maximum |dY/dn| within +-8 px along n
(scripts/rgb_edges.py). Shift: least squares s = n . (dx, dy) on |s| <= 2 (RGB edge = DA3 edge + (dx, dy)).

Per image: shift and SE (naive, samples along a contour are correlated). Pooled: the fit on all samples (naive SE) and
the image-level estimate (mean of the per-image shifts, SE = std/sqrt(n_images): the decisive one, it does not
pretend the samples are independent). Stable if the pooled 95 % CI excludes 0 in at least one axis and at least 80 %
of the images lie within 0.3 px (Euclidean) of the pooled shift.
Writes baseline_output/diag/geom/g2/align.json and align_samples.parquet.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rgb_edges import fit_shift, luminance, rgb_offset  # noqa: E402
from illum_baseline.stage_b import image_stems  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
RUN = "geom_v0"
OUT = ROOT / "baseline_output" / "diag" / "geom" / "g2"
Z = 3.29


def strong(cand: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(invalid="ignore"):
        m = (cand["accepted"] & (cand["fit_mode"] == "m8") & (cand["r"] > 0.2) & (cand["signif_add"] > 2 * Z)
             & ~cand["sky_edge"])
    return cand[m]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, samples = [], []
    for batch, root in BATCHES.items():
        rd = root / "stage_b" / RUN
        for stem in image_stems(rd):
            c = strong(pd.read_parquet(rd / f"{stem}_depth_candidates.parquet"))
            Y = luminance(root / "stage_a", stem)
            x0, n = c[["u0", "v0"]].to_numpy(), c[["n_u", "n_v"]].to_numpy()
            s = rgb_offset(Y, x0, n)
            t, se, k = fit_shift(s, n)
            rows.append({"batch": batch, "stem": stem, "n_strong": int(len(c)), "n_fit": k, "dx": t[0], "dy": t[1],
                         "se_dx": se[0], "se_dy": se[1]})
            samples.append(pd.DataFrame({"batch": batch, "stem": stem, "u0": x0[:, 0], "v0": x0[:, 1], "n_u": n[:, 0],
                                         "n_v": n[:, 1], "s": s}))
    df = pd.DataFrame(rows)
    S = pd.concat(samples, ignore_index=True)
    S.to_parquet(OUT / "align_samples.parquet")
    tp, sep, kp = fit_shift(S["s"].to_numpy(), S[["n_u", "n_v"]].to_numpy())
    ok = df["n_fit"] >= 20
    img = df[ok]
    mean = img[["dx", "dy"]].mean().to_numpy()
    se_img = img[["dx", "dy"]].std(ddof=1).to_numpy() / math.sqrt(len(img))
    ci = np.stack([mean - 1.96 * se_img, mean + 1.96 * se_img], -1)
    excl0 = bool(((ci[:, 0] > 0) | (ci[:, 1] < 0)).any())
    dist = np.hypot(img["dx"] - mean[0], img["dy"] - mean[1])
    within = float((dist <= 0.3).mean())
    stable = excl0 and within >= 0.8
    rep = {"per_image": df.to_dict(orient="records"),
           "pooled_samples": {"dx": tp[0], "dy": tp[1], "se_dx": sep[0], "se_dy": sep[1], "n": kp},
           "pooled_images": {"dx": mean[0], "dy": mean[1], "se_dx": se_img[0], "se_dy": se_img[1],
                             "ci95_dx": ci[0].tolist(), "ci95_dy": ci[1].tolist(), "n_images": int(len(img)),
                             "images_excluded_n_fit_lt_20": df.loc[~ok, "stem"].tolist()},
           "spread": {"dist_P50": float(np.median(dist)), "dist_max": float(dist.max()), "share_within_0.3": within,
                      "sd_dx": float(img["dx"].std(ddof=1)), "sd_dy": float(img["dy"].std(ddof=1))},
           "rule": {"ci_excludes_0": excl0, "share_within_0.3_ge_0.8": within >= 0.8, "stable": stable}}
    (OUT / "align.json").write_text(json.dumps(rep, indent=1, default=float))
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print("\npooled (samples):", {k: round(v, 3) for k, v in rep["pooled_samples"].items()})
    print("pooled (images):", json.dumps(rep["pooled_images"], default=float))
    print("spread:", rep["spread"])
    print("rule:", rep["rule"])


if __name__ == "__main__":
    main()
