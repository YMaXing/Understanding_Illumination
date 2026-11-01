"""TASK_FOLD_V2 F6: search the 10 images for moderate creases with a plateau-step normal profile.

Components (8-connected NMS pixels, >= 25 px) of gate-rejected valid fold candidates with median dtheta(k1) in
[5, 25] deg; at the component's middle member, the dense normal-angle profile along e1 must show a step between two plateaus: step >=
4 deg and each plateau's P10-P90 range <= RANGE_FRAC x step, plateaus at max(width/2 + 1, 4) <= |s| <= L_PLATEAU
(arguments: L_PLATEAU, default 24; RANGE_FRAC, default 1/3). Writes baseline_output/diag/fold_v2/f6/crease_search.json and profile figures for the survivors.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fold_v2_f6 import plateau_step, profile, setup  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.geom.fold import FoldParams  # noqa: E402

OUT = ROOT / "baseline_output" / "diag" / "fold_v2" / "f6"


L_PLATEAU = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
RANGE_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 1 / 3


def main():
    cfg = load_config(ROOT / "configs" / "candidate_fold_only.yaml")
    p = FoldParams.from_dict(dataclasses.asdict(cfg.geometry.fold))
    found = []
    for f in sorted((ROOT / "baseline_output/stage_a").glob("taiwei_cui_*.npz")):
        s = f.stem[-3:]
        ctx, res = setup(f.stem, cfg)
        df = res.candidates
        g = df[df.valid & (df.reason == "gate")]
        H, W = ctx.shape
        img = np.zeros((H, W), bool)
        img[g.v, g.u] = True
        lab, n = ndimage.label(img, structure=np.ones((3, 3), bool))
        comp = lab[g.v, g.u]
        for c in range(1, n + 1):
            m = comp == c
            if m.sum() < 25:
                continue
            gc = g[m]
            dth = np.degrees(gc.dtheta.median())
            if not 5 <= dth <= 25:
                continue
            xy = gc[["u0", "v0"]].to_numpy()
            mid = gc.iloc[int(np.argmin(((xy - np.median(xy, 0)) ** 2).sum(1)))]
            x, e = np.array([mid.u0, mid.v0]), np.array([mid.e1_u, mid.e1_v])
            sv, ang = profile(ctx, x, e, L=max(L_PLATEAU, 24.0))
            a = max(mid.width / 2 + 1, 4.0)
            stp, rng = plateau_step(sv, ang, a, L=L_PLATEAU)
            if np.isfinite(stp) and stp >= 4 and rng <= RANGE_FRAC * stp:
                found.append({"stem": s, "n_px": int(m.sum()), "x": [round(mid.u0, 1), round(mid.v0, 1)],
                              "e1": [round(mid.e1_u, 3), round(mid.e1_v, 3)], "dtheta_med_deg": float(dth),
                              "step_deg": stp, "plateau_range_deg": rng, "width": float(mid.width),
                              "T_over_lambda_med": float(gc.T_over_lambda.median()),
                              "C_med": float(gc.C.median()) if gc.C.notna().any() else float("nan"),
                              "u_range": [int(gc.u.min()), int(gc.u.max())], "v_range": [int(gc.v.min()), int(gc.v.max())]})
                fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
                ax.plot(sv, ang, "k-")
                ax.axvspan(-a, a, color="tab:orange", alpha=0.2)
                ax.set_title(f"{s} ({mid.u0:.0f},{mid.v0:.0f}) n={m.sum()} step {stp:.1f} range {rng:.1f}", fontsize=9)
                fig.savefig(OUT / f"crease_{s}_{int(mid.u0)}_{int(mid.v0)}.png", dpi=80)
                plt.close(fig)
    (OUT / f"crease_search_L{L_PLATEAU:g}.json").write_text(json.dumps(found, indent=1))
    for r in found:
        print(r)


if __name__ == "__main__":
    main()
