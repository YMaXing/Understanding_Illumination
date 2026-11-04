"""F2 decisions 2-4: compare arms A (k_min = 6) and B (k_min = 11) on synthetic scenes and the 10 real images.

Run from geometry/dedicated:  env -u PYTHONPATH uv run python scripts/fold_arms_report.py [--out report.md]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from illum_baseline.config import detector_params, load_config  # noqa: E402
from illum_baseline.detector import Params  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.fold import FoldChannel  # noqa: E402
from illum_baseline.geom.synthetic import F_PX, fold_scene, scene_objects, step_scene  # noqa: E402

TRI = Params(prefilter="trilateral")
K = Intrinsics.centered(F_PX, F_PX, 504, 378)
NOISE = 5e-4
LAM_N = json.loads((ROOT / "illum_baseline/geom/data/lambda_n.json").read_text())
ARMS = {"A": 6, "B": 11}
FOLD_CASES = [(30, 0, True), (60, 135, True), (90, 30, False), (90, 0, True), (120, 90, False), (60, 90, False)]
STEP_CASES = [(1.1, (20, -20), 30), (1.5, (0, 40), 0), (3.0, (-30, 10), 90)]
BOX_079 = (115, 320, 85, 330)  # u0, u1, v0, v1 (network px, upright): the large box in taiwei_cui_079


def noisy(D, rel, seed):
    return 1.0 / ((1.0 / D) * (1.0 + rel * np.random.default_rng(seed).normal(size=D.shape)))


def at(a, x):
    u = np.clip(np.rint(x[:, 0]).astype(int), 0, a.shape[1] - 1)
    v = np.clip(np.rint(x[:, 1]).astype(int), 0, a.shape[0] - 1)
    return a[v, u]


def pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


def synthetic(k_min: int) -> dict:
    prm = {"k_min": k_min, "lambda_n": LAM_N[f"k{k_min}"]}
    err, both, n_line, cross = [], 0, 0, 0
    for dth, phi, cvx in FOLD_CASES:
        D, sid, ns, sky = fold_scene(dth, phi, cvx)
        df = FoldChannel().run(GeometryContext(noisy(D, NOISE, 3), sky.astype(float), K, TRI), prm).candidates
        ph = math.radians(phi)
        q = np.array([-math.sin(ph), math.cos(ph)])
        dist = (df[["u0", "v0"]].to_numpy() - np.array([251.5, 188.5])) @ q
        line = (np.abs(dist) < 2.5) & df["valid"].to_numpy() & ~sky[df["v"], df["u"]]
        n_line += line.sum()
        both += (line & df["planar_p"] & df["planar_m"]).sum()
        cross += (line & df["planes_cross"]).sum()
        err.append(np.degrees(np.abs(df["dtheta"][line] - math.radians(dth))))
    err = np.concatenate(err)
    # scene with box and cylinders: box folds and box silhouettes
    D, sid, nrm, names = scene_objects(radii=(0.5, 1.0, 2.0))
    df = FoldChannel().run(GeometryContext(noisy(D, NOISE, 0), None, K, TRI), prm).candidates
    x0, e1, k1 = df[["u0", "v0"]].to_numpy(), df[["e1_u", "e1_v"]].to_numpy(), df["k1"].to_numpy()
    sp, sm = at(sid, x0 + (k1 + 4)[:, None] * e1), at(sid, x0 - (k1 + 4)[:, None] * e1)
    Zp, Zm = at(D, x0 + 3 * e1), at(D, x0 - 3 * e1)
    sil = np.abs(Zp - Zm) / np.minimum(Zp, Zm) > 0.05
    boxany = np.isin(sp, [3, 4, 5]) | np.isin(sm, [3, 4, 5])
    solid = np.isin(sp, [1, 3, 4, 5]) & np.isin(sm, [1, 3, 4, 5])
    valid = df["valid"].to_numpy()
    box_fold = ~sil & (sp != sm) & solid & boxany & valid
    box_sil = sil & boxany & valid
    cyl = (sp == sm) & (sp >= 10) & df["gate"].to_numpy()
    # single steps
    st_n, st_dj = 0, 0
    for ratio, tilts, phi in STEP_CASES:
        D, sid2, _ = step_scene(ratio, *tilts, phi)
        d2 = FoldChannel().run(GeometryContext(noisy(D, NOISE, 5), None, K, TRI), prm).candidates
        ph = math.radians(phi)
        q = np.array([-math.sin(ph), math.cos(ph)])
        dist = (d2[["u0", "v0"]].to_numpy() - np.array([251.5, 188.5])) @ q
        line = (np.abs(dist) < 2.5) & d2["valid"].to_numpy()
        st_n += line.sum()
        st_dj += (line & d2["depth_jump"]).sum()
    return {
        "fold_dtheta_err_med": float(np.median(err)), "fold_dtheta_err_p95": pct(err, 95),
        "fold_both_planar": both / max(n_line, 1), "fold_planes_cross": cross / max(n_line, 1), "fold_n": int(n_line),
        "box_fold_both_planar": float((box_fold & df["planar_p"] & df["planar_m"]).sum() / max(box_fold.sum(), 1)),
        "box_fold_depth_jump": int((box_fold & df["depth_jump"]).sum()), "box_fold_n": int(box_fold.sum()),
        "sil_recall_objects": float((box_sil & df["depth_jump"]).sum() / max(box_sil.sum(), 1)),
        "sil_n": int(box_sil.sum()), "step_recall": st_dj / max(st_n, 1), "step_n": int(st_n),
        "cyl_gate": int(cyl.sum()), "cyl_accepted": int((cyl & df["accepted"]).sum()),
    }


def real(k_min: int, cfg) -> list[dict]:
    prm = {"k_min": k_min, "lambda_n": LAM_N[f"k{k_min}"]}
    p = detector_params(cfg)
    rows = []
    for f in sorted((ROOT / "baseline_output/stage_a").glob("taiwei_cui_*.npz")):
        c = np.load(f)
        meta = json.loads(f.with_suffix(".json").read_text())
        ctx = GeometryContext(c["D"], c["sky_prob"].astype(np.float64), Intrinsics.from_stage_a(meta), p)
        ctx.n_hat
        t0 = time.time()
        res = FoldChannel().run(ctx, prm)
        dt = time.time() - t0
        df, s = res.candidates, res.scalars
        v, g = df["valid"].to_numpy(), df["gate"].to_numpy()
        dg = np.degrees(df["dtheta_gate"][v])
        row = {"stem": f.stem[-3:], "t_fold": dt, "lambda_raw": s["lambda_hat_raw"], "lambda": s["lambda_hat"],
               "eps_hat": s["eps_hat"], "null_rms_q": s["null_rms_q"], "sigma_geo_deg": math.degrees(s["sigma_geo"]),
               "n_cand": s["n_cand"], "n_valid": s["n_valid"],
               "both_planar": float((df["planar_p"] & df["planar_m"])[v].mean()),
               "no_veto": float(df["no_veto"][g].mean()) if g.any() else float("nan"),
               "gate_mult": s["n_gate_mult"] / max(s["n_valid"], 1), "gate_add": s["n_gate_add"] / max(s["n_valid"], 1),
               "dg_p10": pct(dg, 10), "dg_p50": pct(dg, 50), "dg_p90": pct(dg, 90),
               "n_accepted": s["n_accepted"], "w_lt_05": s["frac_w_lt_0.5"], "w_lt_09": s["frac_w_lt_0.9"]}
        if f.stem.endswith("079"):
            u0, u1, v0, v1 = BOX_079
            roi = df["u"].between(u0, u1) & df["v"].between(v0, v1)
            rows_fold = roi & df["v"].between(225, 260)
            lost = roi & df["gate"] & ~df["veto_pass"] & ~df["k2_clean"]
            row |= {"box_accepted": int((roi & df["accepted"]).sum()),
                    "box_rows225_260_accepted": int((rows_fold & df["accepted"]).sum()),
                    "box_gate": int((roi & df["gate"]).sum()), "box_vetoed": int((roi & df["gate"] & ~df["veto_pass"]).sum()),
                    "box_lost_narrow_face_veto": int(lost.sum()),
                    "box_accepted_dtheta_med": float(np.degrees(df["dtheta"][roi & df["accepted"]].median()))}
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(ROOT / "configs/candidate_tri_veto4.yaml")
    out = {}
    for arm, k_min in ARMS.items():
        t0 = time.time()
        out[arm] = {"k_min": k_min, "lambda_n": LAM_N[f"k{k_min}"], "synthetic": synthetic(k_min), "real": real(k_min, cfg)}
        print(f"arm {arm} done in {time.time() - t0:.0f} s", flush=True)
    text = json.dumps(out, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
