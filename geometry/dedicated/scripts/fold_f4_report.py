"""TASK_FOLD_CHANNEL F4 report: run tables, spot checks, null arms, 075 breakdown.

Reads the saved runs stage_b/{fold_only, tri_veto4_fold, tri_veto4, baseline_v0}; the null arms and the 075 breakdown
recompute the fold channel per image (multiplicative, additive, additive_local). Writes
baseline_output/diag/fold_f4/{report.json, 075_wfold_classes.png} and prints the tables.

Run from geometry/dedicated:  env -u PYTHONPATH uv run python scripts/fold_f4_report.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.geom.fold import FoldChannel, FoldParams  # noqa: E402
from illum_baseline.stage_b import image_stems, load_upright, run_geometry  # noqa: E402

SB = ROOT / "baseline_output" / "stage_b"
OUT = ROOT / "baseline_output" / "diag" / "fold_f4"
ARMS = ("multiplicative", "additive", "additive_local")
FOLD_U = (180, 215)          # 079 lit/shaded fold columns at rows 230 / 254 (upright network px)
FOLD_ROWS = (230, 254)
BOX = (115, 320, 85, 330)    # 079 box region u0 u1 v0 v1


def rect(shape, u0, u1, v0, v1):
    m = np.zeros(shape, dtype=bool)
    m[v0:v1 + 1, u0:u1 + 1] = True
    return m


def rois(stem, shape):
    if stem.endswith("079"):
        return {"079 grass": rect(shape, 5, 110, 5, 110) | rect(shape, 5, 60, 360, 495),
                "079 grass sunlit": rect(shape, 110, 290, 5, 85)}
    if stem.endswith("083"):
        return {"083 bollard": rect(shape, 175, 205, 260, 330)}
    if stem.endswith("070"):
        return {"070 floor": np.load(ROOT / "baseline_output/diag/070/roi_mask.npy").astype(bool)}
    return {}


def fold_rows_w(w):
    return [float(w[r, FOLD_U[0]:FOLD_U[1] + 1].min()) for r in FOLD_ROWS]


def load(run, stem):
    cfg = load_config(SB / run / "config_resolved.yaml")
    a, meta = load_upright(cfg, SB / run, stem)
    return a, meta


def run_tables(stems):
    tables = {}
    for run in ("fold_only", "tri_veto4_fold"):
        rows = []
        for stem in stems:
            a, meta = load(run, stem)
            t, _ = load("tri_veto4", stem)
            fo = meta["channels"]["fold"]
            w, wt = a["w_net"].astype(float), t["w_net"].astype(float)
            cand = pd.read_parquet(SB / run / f"{stem}_fold_candidates.parquet")
            nv = a["normal_valid"]
            rows.append({
                "stem": stem[-3:], "w<0.5": (w < 0.5).mean(), "w<0.9": (w < 0.9).mean(),
                "d_w<0.5 vs tri_veto4": (w < 0.5).mean() - (wt < 0.5).mean(),
                "d_w<0.9 vs tri_veto4": (w < 0.9).mean() - (wt < 0.9).mean(),
                "lambda_hat": fo["lambda_hat"], "eps_hat": fo["eps_hat"],
                "med_sigma_theta_deg": float(np.median(np.degrees(a["sigma_theta"][nv].astype(float)))),
                "cand": fo["n_cand"], "valid": fo["n_valid"], "gate": fo["n_gate"], "veto": fo["n_veto"],
                "component": fo["n_accepted"],
                "med_J_acc": float(cand.loc[cand["accepted"], "J"].median()) if fo["n_accepted"] else float("nan"),
                "t_context_s": fo["context_s"], "t_fold_s": fo["runtime_s"]})
        tables[run] = pd.DataFrame(rows)
    return tables


def spot_checks():
    out = {}
    for run in ("baseline_v0", "tri_veto4", "tri_veto4_fold", "fold_only"):
        a, _ = load(run, "taiwei_cui_079")
        out[f"079 fold rows w ({run})"] = fold_rows_w(a["w_net"].astype(float))
    a, _ = load("fold_only", "taiwei_cui_083")
    m = rois("taiwei_cui_083", a["fold_w_net"].shape)["083 bollard"]
    out["083 bollard median w_fold"] = float(np.median(a["fold_w_net"][m].astype(float)))
    for run in ("tri_veto4", "tri_veto4_fold"):
        a, _ = load(run, "taiwei_cui_070")
        m = rois("taiwei_cui_070", a["w_net"].shape)["070 floor"]
        out[f"070 floor share w<0.9 ({run})"] = float((a["w_net"][m].astype(float) < 0.9).mean())
        a, _ = load(run, "taiwei_cui_079")
        for name, m in rois("taiwei_cui_079", a["w_net"].shape).items():
            out[f"{name} share w<0.5 ({run})"] = float((a["w_net"][m].astype(float) < 0.5).mean())
    return out


def classify(df, w_fold, D, k_half_extra, thr=0.5):
    """Per pixel with w_fold < thr: the accepted candidate that sets its weight (smallest w_c whose band covers it),
    classed as fold / rim / limb / texture:
      step    = a depth discontinuity (|d log D| > 5% between 0.5-px samples) between the two window centres;
      fold    = no step, both k1 windows planar;     rim  = step, at least one side planar (edge of a flat face);
      limb    = step, neither side planar (curved occluding contour);  texture = no step, not both sides planar."""
    vv, uu = np.nonzero(w_fold < thr)
    acc = df[df["accepted"]].reset_index(drop=True)
    if not len(vv) or not len(acc):
        return np.zeros(w_fold.shape, int), {}
    cls_c = []
    for r in acc.itertuples():
        e = np.array([r.e1_u, r.e1_v])
        half = r.k1 + k_half_extra
        s = np.linspace(-half, half, int(4 * half) + 1)
        pts = np.array([r.u0, r.v0]) + s[:, None] * e
        z = D[np.clip(np.rint(pts[:, 1]).astype(int), 0, D.shape[0] - 1),
              np.clip(np.rint(pts[:, 0]).astype(int), 0, D.shape[1] - 1)]
        step = bool((np.abs(np.diff(np.log(z))) > 0.05).any())
        both = r.planar_p and r.planar_m
        one = r.planar_p or r.planar_m
        cls_c.append((2 if one else 3) if step else (1 if both else 4))
    cls_c = np.array(cls_c)
    resp = np.empty(len(vv), int)
    for i0 in range(0, len(vv), 4000):  # chunked nearest-responsible search
        sl = slice(i0, i0 + 4000)
        d2 = (acc["u"].to_numpy()[None] - uu[sl, None]) ** 2 + (acc["v"].to_numpy()[None] - vv[sl, None]) ** 2
        wc = np.where(d2 <= acc["band"].to_numpy()[None] ** 2, acc["w_c"].to_numpy()[None], np.inf)
        resp[sl] = np.argmin(wc, axis=1)
    lab = np.zeros(w_fold.shape, int)
    lab[vv, uu] = cls_c[resp]
    names = {1: "fold", 2: "rim", 3: "limb", 4: "texture"}
    return lab, {names[k]: int((lab == k).sum()) for k in names}


def null_arms(stems):
    cfg = load_config(ROOT / "configs" / "candidate_fold_only.yaml")
    roi_tab, box, b075 = [], [], {}
    for stem in stems:
        if not any(stem.endswith(s) for s in ("070", "075", "079", "083")):
            continue
        meta_a = json.loads((ROOT / "baseline_output/stage_a" / f"{stem}.json").read_text())
        c = np.load(ROOT / "baseline_output/stage_a" / f"{stem}.npz")
        ctx, _, _ = run_geometry(cfg, c["D"], c["sky_prob"].astype(np.float64), meta_a)
        t, _ = load("tri_veto4", stem)
        wt = t["w_net"].astype(float)
        for arm in ARMS:
            p = FoldParams.from_dict(dataclasses.asdict(cfg.geometry.fold) | {"null_model": arm})
            r = FoldChannel().run(ctx, p)
            wf = r.w_net
            for name, m in rois(stem, wf.shape).items():
                roi_tab.append({"roi": name, "arm": arm, "w_fold<0.5": float((wf[m] < 0.5).mean()),
                                "combined w<0.5": float((np.minimum(wt, wf)[m] < 0.5).mean())})
            if stem.endswith("079"):
                d = r.candidates
                inb = d["u"].between(BOX[0], BOX[1]) & d["v"].between(225, 260)
                box.append({"arm": arm, "accepted rows 225-260": int((inb & d["accepted"]).sum()),
                            "w_fold rows 230/254": fold_rows_w(wf), "combined w rows 230/254":
                            fold_rows_w(np.minimum(wt, wf))})
            if stem.endswith("075"):
                lab, counts = classify(r.candidates, wf, ctx.D.astype(float), p.m / 2)
                b075[arm] = {"labels": lab, "counts": counts, "share": float((wf < 0.5).mean())}
    return pd.DataFrame(roi_tab), pd.DataFrame(box), b075


def plot_075(b075, srgb):
    cmap = matplotlib.colors.ListedColormap([(0, 0, 0, 0), "tab:red", "tab:blue", "tab:green", "gold"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.5), constrained_layout=True)
    for ax, arm in zip(axes, ARMS):
        lab = b075[arm]["labels"]
        ax.imshow(srgb, extent=(-0.5, lab.shape[1] - 0.5, lab.shape[0] - 0.5, -0.5))
        ax.imshow(lab, cmap=cmap, vmin=0, vmax=4, interpolation="nearest", alpha=0.75,
                  extent=(-0.5, lab.shape[1] - 0.5, lab.shape[0] - 0.5, -0.5))
        c = b075[arm]["counts"]
        ax.set_title(f"{arm}: share w_fold<0.5 {b075[arm]['share']:.3f}\nfold {c.get('fold', 0)} (red), rim "
                     f"{c.get('rim', 0)} (blue), limb {c.get('limb', 0)} (green), texture {c.get('texture', 0)} (gold)")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("taiwei_cui_075: pixels with w_fold < 0.5 by the class of the candidate that sets them")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "075_wfold_classes.png", dpi=80)
    plt.close(fig)


def main():
    import cv2

    stems = image_stems(SB / "fold_only")
    pd.set_option("display.width", 250)
    tables = run_tables(stems)
    for run, t in tables.items():
        print(f"\n== {run}\n" + t.round(4).to_string(index=False))
    sc = spot_checks()
    print("\n== spot checks")
    for k, v in sc.items():
        print(f"  {k}: {np.round(v, 3)}")
    roi_tab, box, b075 = null_arms(stems)
    print("\n== null arms: ROIs\n" + roi_tab.round(4).to_string(index=False))
    print("\n== null arms: 079 box\n" + box.to_string(index=False))
    print("\n== 075 w_fold<0.5 breakdown")
    for arm in ARMS:
        print(f"  {arm}: share {b075[arm]['share']:.3f}, {b075[arm]['counts']}")
    meta_a = json.loads((ROOT / "baseline_output/stage_a/taiwei_cui_075.json").read_text())
    srgb = cv2.cvtColor(cv2.imread(str(ROOT / "baseline_output/stage_a/srgb/taiwei_cui_075.png")), cv2.COLOR_BGR2RGB)
    srgb = cv2.resize(srgb, (meta_a["network"]["W_n"], meta_a["network"]["H_n"]), interpolation=cv2.INTER_AREA)
    plot_075(b075, srgb)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps({
        "runs": {k: v.to_dict(orient="records") for k, v in tables.items()}, "spot_checks": sc,
        "null_rois": roi_tab.to_dict(orient="records"), "null_box": box.to_dict(orient="records"),
        "b075": {a: {"counts": b075[a]["counts"], "share": b075[a]["share"]} for a in ARMS}}, indent=1, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
