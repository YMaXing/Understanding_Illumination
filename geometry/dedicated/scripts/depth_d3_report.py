"""TASK_DEPTH_CHANNEL D3 report on the 10 images (DepthChannel with the defaults, context from Stage A, no Stage B):
counts per stage, fit modes, gate diagnostics (r_gate = z sqrt(lambda_hat) SE / rho_hat_-, the share of candidates
where significance rather than tau_low binds, the additive null as a recorded alternative), Delta Z0 sensitivity,
the centring measure for real occlusions vs synthetic ghosts, and components that merge both sides of a thin
structure. Writes baseline_output/diag/depth_v1/d3/d3_report.json.

Run from geometry/dedicated:  env -u PYTHONPATH uv run python scripts/depth_d3_report.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_depth import CFG, STEMS, noisy, real_context  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.detector import Params  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.candidates import depth_candidates  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.depth import DepthChannel, DepthParams  # noqa: E402
from illum_baseline.geom.synthetic import F_PX, step_scene  # noqa: E402

OUT = ROOT / "baseline_output" / "diag" / "depth_v1" / "d3"
BINS = np.round(np.arange(0, 1.0001, 0.05), 2)


def pct(a, qs=(10, 50, 90)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, q)) for q in qs] if a.size else [float("nan")] * len(qs)


def merges(df: pd.DataFrame, radius: float = 8.0) -> list[dict]:
    """Accepted components holding two members that face each other (n_i . n_j < -0.5) within radius px."""
    out = []
    a = df[df["accepted"]]
    for cid, g in a.groupby("component_id"):
        x, n = g[["u0", "v0"]].to_numpy(), g[["n_u", "n_v"]].to_numpy()
        if len(g) < 2:
            continue
        d = np.hypot(*(x[:, None, :] - x[None]).transpose(2, 0, 1))
        opp = (n @ n.T < -0.5) & (d <= radius)
        if opp.any():
            out.append({"component_id": int(cid), "n": int(len(g)), "u": float(x[:, 0].mean()),
                        "v": float(x[:, 1].mean())})
    return out


def ghost_centring() -> np.ndarray:
    """Centring of the ghost crossings next to synthetic steps (prior smear bound, centring off): D-3 passes 1.5-10 px
    from the edge, noise-free and noise 5e-4."""
    TRI = Params(prefilter="trilateral")
    K = Intrinsics.centered(F_PX, F_PX, 504, 378)
    p = dataclasses.replace(DepthParams(), sigma_w_max=3.0, centre_tol=math.inf)
    C = np.array([251.5, 188.5])
    out = []
    for ratio in (1.1, 1.5, 3.0):
        for phi in (30, 75):
            Z, _, _ = step_scene(ratio, 20, -20, phi)
            q = np.array([-math.sin(math.radians(phi)), math.cos(math.radians(phi))])
            for noise in (0.0, 5e-4):
                ctx = GeometryContext(noisy(Z, noise, 1) if noise else Z, None, K, TRI)
                c = depth_candidates(ctx, p)
                d = np.abs((c.x0 - C) @ q)
                g = c.passed & (d >= 1.5) & (d <= 10)
                out.append(np.abs(c.s_p[g] + c.s_m[g]) / c.s_sep[g])
    return np.concatenate(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(CFG)
    p = DepthParams()
    rows, real_cent, rej_cent, merge_rec = [], [], [], {}
    for stem in STEMS:
        ctx, _ = real_context(stem, cfg)
        res = DepthChannel().run(ctx, p)
        df, sc = res.candidates, res.scalars
        a = df[df["accepted"]]
        row = {"stem": stem[-3:]}
        for k in ("n_cand", "n_polarity", "n_height", "n_sharpness", "n_centring", "n_passed", "n_sky_near",
                  "n_sign_ok", "n_signif_ok", "n_rel_ok", "n_accepted"):
            row[k] = sc[k]
        row |= {k: sc[k] for k in ("n_fit_m8", "n_fit_m4", "n_fit_const", "n_fit_doublet", "n_thin_accepted",
                                   "n_sky_edge_accepted", "n_lower_bound_accepted")}
        row |= {"lambda_hat": sc["lambda_hat"], "r_gate_P10": sc["r_gate_P10"], "r_gate_P50": sc["r_gate_P50"],
                "r_gate_P90": sc["r_gate_P90"], "r_gate_P90_over_P10": sc["r_gate_P90"] / sc["r_gate_P10"],
                "share_signif_binding": sc["share_signif_binding"], "sigma_geo_rho": sc["sigma_geo_rho"],
                "n_signif_ok_add": sc["n_signif_ok_add"]}
        fin = a["dZ"].to_numpy()
        row |= {"dZ_P10": pct(fin)[0], "dZ_P50": pct(fin)[1], "dZ_P90": pct(fin)[2],
                "dZ_inf_share": float(np.isinf(fin).mean()) if len(fin) else float("nan"),
                "r_acc_P50": pct(a["r"])[1], "w<0.5": sc["frac_w_lt_0.5"], "w<0.9": sc["frac_w_lt_0.9"],
                "runtime_s": sc["runtime_s"], "context_s": sc["context_s"],
                "normals_built": "_normals" in ctx.__dict__}
        for dz0 in (0.05, 0.20):
            r2 = DepthChannel().run(ctx, dataclasses.replace(p, dZ0=dz0)).scalars
            row[f"w<0.5@{dz0}"], row[f"w<0.9@{dz0}"] = r2["frac_w_lt_0.5"], r2["frac_w_lt_0.9"]
        with np.errstate(invalid="ignore"):
            unamb = df["accepted"] & (df["r"] > 0.2) & (df["signif"] > 2 * p.z) & (df["fit_mode"] == "m8")
        real_cent.append(df.loc[unamb, "centring"].to_numpy())
        rej_cent.append(df.loc[df["reason"] == "centring", "centring"].to_numpy())
        merge_rec[stem[-3:]] = merges(df)
        row["n_merged_components"] = len(merge_rec[stem[-3:]])
        rows.append(row)
        print(f"{stem[-3:]}: accepted {sc['n_accepted']} (m8 {sc['n_fit_m8']}, thin {sc['n_thin_accepted']}, sky "
              f"{sc['n_sky_edge_accepted']}, lb {sc['n_lower_bound_accepted']}); r_gate P10/50/90 "
              f"{sc['r_gate_P10']:.3f}/{sc['r_gate_P50']:.3f}/{sc['r_gate_P90']:.3f}; signif binding "
              f"{sc['share_signif_binding']:.2f}; {sc['runtime_s']:.2f} s")
    df = pd.DataFrame(rows)
    rc, gc = np.concatenate(real_cent), ghost_centring()
    cent = {"real_unambiguous": {"n": int(rc.size), "P50_P90_P95_P99_max": pct(rc, (50, 90, 95, 99, 100)),
                                 "hist": np.histogram(rc, BINS)[0].tolist()},
            "ghosts_synthetic": {"n": int(gc.size), "P1_P5_P10_P50_max": pct(gc, (1, 5, 10, 50, 100)),
                                 "hist": np.histogram(np.minimum(gc, 1.0), BINS)[0].tolist()},
            "bins": BINS.tolist(), "real_rejected_by_centring": int(np.concatenate(rej_cent).size),
            "threshold": p.centre_tol}
    rep = {"per_image": df.to_dict(orient="records"), "centring": cent, "merged_components": merge_rec}
    (OUT / "d3_report.json").write_text(json.dumps(rep, indent=1))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    print("\n" + df.round(4).to_string(index=False))
    print("\ncentring: real", cent["real_unambiguous"]["P50_P90_P95_P99_max"], "ghosts",
          cent["ghosts_synthetic"]["P1_P5_P10_P50_max"])
    print("  hist bins", BINS.tolist())
    print("  real  ", cent["real_unambiguous"]["hist"])
    print("  ghosts", cent["ghosts_synthetic"]["hist"])
    print("merged components:", {k: v for k, v in merge_rec.items() if v})


if __name__ == "__main__":
    main()
