"""TASK_GEOM_V2 V4: fold recall with rescue_undecidable corroborate (report only; RGB for the overlays and the scorer).

Runs: base geom_v1_hessian (site hessian_peak, rescue_undecidable reject), V4 geom_v1_v4 (+ corroborate), report-only
geom_v1_v4_c6 (+ rescue_min_component_px 6); geom_v1 for the texture constraint.
  - the 21 veto-blocked open fold / crease units (V1: gate failures kept out of the rescue by an undecidable veto,
    baseline_output/diag/geom/v2/fold_gaps_candidates.csv): closed under core w (target >= 7) and per-pixel relaxed
    share (area-weighted);
  - constraints: texture ROIs (combined) <= geom_v1 + 0.01; 083 bollard interior unchanged; texture-labelled units
    relaxed (core) + 0.02 at most; geom_fp units relaxed (core) not increased;
  - fold-candidate RGB-edge coverage (reliable edges), fold targets, ROI coverage incl. crease, the 076 facade ledge
    and 088 rock ridge polylines, units 15, 97, 124;
  - newly rescued candidates (accepted in V4, not in base) split by whether another accepted edge (accepted depth
    candidate, or accepted fold candidate outside the candidate's own 8-connected new component) lies within 10 px
    (narrow-face cases) or not; per group: candidates, fold / crease gap units closed when only that group's bands are
    added to base core w, and the texture ROI shares and texture-labelled relaxed share with only that group added;
  - overlays of the newly rescued candidates: 075 kiosk, 083 bollard (smooth-curvature controls), 072 / 073 boxes.
Writes baseline_output/diag/geom/v2/recall.json and recall_overlays/<stem>.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geom_eval import area_mask, roi_list, roi_metrics  # noqa: E402
from geom_g2_coverage import raster  # noqa: E402
from geom_g4 import GAPS  # noqa: E402
from geom_score import OUT, Run, audit, g1_table, load_labels, rgb_coverage, rgb_edge_set, unit_fractions  # noqa: E402
from illum_baseline.stage_b import load_upright  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
V1, BASE, NEW, C6 = "geom_v1", "geom_v1_hessian", "geom_v1_v4", "geom_v1_v4_c6"
OVERLAYS = ("taiwei_cui_075", "taiwei_cui_083", "acharya_mrudula_072", "acharya_mrudula_073")
NAMED = (15, 97, 124)
NEAR_PX = 10.0


def new_rescued(stem: str, batch: str):
    rd_b, rd_n = BATCHES[batch] / "stage_b" / BASE, BATCHES[batch] / "stage_b" / NEW
    fb = pd.read_parquet(rd_b / f"{stem}_fold_candidates.parquet")
    fn = pd.read_parquet(rd_n / f"{stem}_fold_candidates.parquet")
    new = fn[fn["accepted"] & ~fb["accepted"]].copy()
    dn = pd.read_parquet(rd_n / f"{stem}_depth_candidates.parquet")
    H, W = load_upright(Run(NEW).cfg[batch], rd_n, stem)[0]["w_net"].shape
    if len(new):
        grid = np.zeros((H, W), bool)
        grid[new["v"], new["u"]] = True
        lab, _ = ndimage.label(grid, structure=np.ones((3, 3), bool))
        new["comp"] = lab[new["v"], new["u"]]
        other_f = fn[fn["accepted"]].copy()
        other_f["comp"] = np.where(grid[other_f["v"], other_f["u"]], lab[other_f["v"], other_f["u"]], -1)
        pts = np.concatenate([other_f[["u0", "v0"]].to_numpy(), dn.loc[dn["accepted"], ["u0", "v0"]].to_numpy()])
        comps = np.concatenate([other_f["comp"].to_numpy(), np.full(int(dn["accepted"].sum()), -1)])
        tree = cKDTree(pts)
        near = []
        for x, c in zip(new[["u0", "v0"]].to_numpy(), new["comp"].to_numpy()):
            js = tree.query_ball_point(x, NEAR_PX)
            near.append(bool(np.any(comps[js] != c)))
        new["narrow"] = near
    else:
        new["comp"], new["narrow"] = [], []
    return new, (H, W)


def band_w(df: pd.DataFrame, shape) -> np.ndarray:
    if not len(df):
        return np.ones(shape)
    return raster(shape, df["u_site"].to_numpy(), df["v_site"].to_numpy(), df["w_c"].to_numpy(),
                  df["band"].to_numpy().astype(int))


def overlay(stem, batch, new, out):
    root = BATCHES[batch]
    sa = json.loads((root / "stage_a" / f"{stem}.json").read_text())
    W, H = sa["network"]["W_n"], sa["network"]["H_n"]
    img = (cv2.resize(cv2.imread(str(root / "stage_a" / "srgb" / f"{stem}.png")), (W, H),
                      interpolation=cv2.INTER_AREA) * 0.6).astype(np.uint8)
    fb = pd.read_parquet(root / "stage_b" / BASE / f"{stem}_fold_candidates.parquet")
    a = fb[fb["accepted"]]
    img[a["v_site"], a["u_site"]] = (200, 200, 200)
    for r in new.itertuples():
        cv2.circle(img, (int(r.u_site), int(r.v_site)), 1, (0, 165, 255) if r.narrow else (0, 0, 255), -1)
    big = cv2.resize(img, (2 * W, 2 * H), interpolation=cv2.INTER_NEAREST)
    cv2.putText(big, f"{stem}: +{len(new)} rescued (red isolated, orange narrow-face), base folds gray", (8, 22), 0,
                0.55, (255, 255, 255), 2)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{stem}.png"), big)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab, draws, strata, totals, masks = load_labels()
    C = pd.read_csv(OUT / "fold_gaps_candidates.csv")
    blocked = sorted(C.loc[C["rescue_block"] == "rescue veto (undecidable)", "unit"].unique().tolist())
    edges = rgb_edge_set()
    runs = {n: Run(n) for n in (V1, BASE, NEW, C6)}
    U, A, G, R = {}, {}, {}, {}
    for n, run in runs.items():
        U[n] = unit_fractions(run, lab, masks).set_index("id")
        A[n] = audit(run, U[n].reset_index(), draws, strata, totals)
        G[n] = g1_table(run)
        R[n] = rgb_coverage(run, edges)
    rep = {"blocked_units": blocked, "runs": {}}
    for n in runs:
        u = U[n].loc[blocked]
        rep["runs"][n] = {
            "blocked_closed": int((u["frac_core"] >= 0.5).sum()),
            "blocked_relaxed_share": float((u["frac_core"] * u["area_px"]).sum() / u["area_px"].sum()),
            "blocked_closed_ids": sorted(int(i) for i in u.index[u["frac_core"] >= 0.5]),
            "named": {str(i): round(float(U[n].loc[i, "frac_core"]), 3) for i in NAMED},
            "texture_rois": {w: G[n][w]["texture_rois"] for w in G[n]},
            "roi_coverage": {w: {c: v["coverage"] for c, v in G[n][w]["classes"].items() if v["coverage"] is not None}
                             for w in G[n]},
            "crease_rois": {w: {r["roi"]: r["coverage"] for r in G[n][w]["rois"] if r["roi"] in
                                ("076 facade ledge", "088 rock ridge")} for w in G[n]},
            "rgb": R[n],
            "core_texture_label": A[n]["core"]["tri-only"]["texture label | relaxed per pixel"],
            "core_geom_fp": A[n]["core"]["geom-only"]["geom_fp | relaxed per pixel"],
            "core_missed_px": A[n]["core"]["tri-only"]["missed true geometry | per pixel"]["share"],
            "fold_crease_gaps_closed_core": int(((U[n]["label"].isin(["fold", "crease"])) & U[n]["bucket"].isin(GAPS)
                                                 & (U[n]["direction"] == "tri-only") & (U[n]["frac_core"] >= 0.5)).sum()),
        }
    # constraints for NEW (and C6)
    for n in (NEW, C6):
        r, b, v1 = rep["runs"][n], rep["runs"][BASE], rep["runs"][V1]
        r["constraints"] = {
            "texture_rois_combined_le_v1+0.01": all(r["texture_rois"]["combined"][k] <= v1["texture_rois"]["combined"][k]
                                                    + 0.01 for k in v1["texture_rois"]["combined"]),
            "083_interior_unchanged": (r["texture_rois"]["core"]["083 bollard interior"] == v1["texture_rois"]["core"][
                "083 bollard interior"]) and (r["texture_rois"]["combined"]["083 bollard interior"] ==
                                              v1["texture_rois"]["combined"]["083 bollard interior"]),
            "texture_label_core_rise_le_0.02": r["core_texture_label"]["share"] - b["core_texture_label"]["share"] <= 0.02,
            "geom_fp_core_not_increased": r["core_geom_fp"]["share"] <= b["core_geom_fp"]["share"] + 1e-12,
            "target_ge_7_of_21": r["blocked_closed"] >= 7}
        r["meets"] = all(r["constraints"].values())

    # narrow-face split of the newly rescued candidates
    tex_rois = [x for x in roi_list() if x["class"] == "texture" and not x.get("legacy")]
    groups = {"narrow": {"n": 0, "closed": set(), "tex": {}}, "isolated": {"n": 0, "closed": set(), "tex": {}}}
    tex_lab = {g: [0.0, 0.0] for g in groups}       # sum area*frac, sum area over texture-labelled units
    run_b = runs[BASE]
    fg = U[BASE][(U[BASE]["label"].isin(["fold", "crease"])) & U[BASE]["bucket"].isin(GAPS)
                 & (U[BASE]["direction"] == "tri-only")]
    for batch, stems in run_b.stems.items():
        for stem in stems:
            new, shape = new_rescued(stem, batch)
            if stem in OVERLAYS:
                overlay(stem, batch, new, OUT / "recall_overlays")
            core_b = run_b.maps(stem)["core_net"]
            for gname, gsel in (("narrow", True), ("isolated", False)):
                sub = new[new["narrow"] == gsel] if len(new) else new
                groups[gname]["n"] += int(len(sub))
                w = np.minimum(core_b, band_w(sub, shape))
                for r in lab[lab["stem"] == stem].itertuples():
                    m = np.zeros(shape[0] * shape[1], bool)
                    m[masks[f"id{r.id:03d}"]] = True
                    f = float((w.ravel()[m] < 0.5).mean())
                    if r.id in fg.index and fg.loc[r.id, "frac_core"] < 0.5 and f >= 0.5:
                        groups[gname]["closed"].add(int(r.id))
                    if r.direction == "tri-only" and r.label == "texture":
                        tex_lab[gname][0] += f * r.area_px
                        tex_lab[gname][1] += r.area_px
                if batch == "batch1":
                    for roi in tex_rois:
                        if stem.endswith(roi["stem"]):
                            groups[gname]["tex"][roi["name"]] = float((w[area_mask(roi, shape)] < 0.5).mean())
    rep["split"] = {g: {"n_new_rescued": v["n"], "fold_crease_gaps_closed": sorted(v["closed"]),
                        "texture_rois_core": v["tex"],
                        "texture_units_relaxed_core_px_weighted": tex_lab[g][0] / max(tex_lab[g][1], 1)}
                    for g, v in groups.items()}
    (OUT / "recall.json").write_text(json.dumps(rep, indent=1, default=float))
    rows = []
    for n, r in rep["runs"].items():
        rows.append({"run": n, "blocked closed (of 21)": r["blocked_closed"],
                     "blocked relaxed": round(r["blocked_relaxed_share"], 4),
                     "fold/crease gaps closed core": r["fold_crease_gaps_closed_core"],
                     "core missed/px": round(r["core_missed_px"], 4),
                     "rgb fold cand comb/core": f"{r['rgb']['combined']['fold_candidates']:.4f} / {r['rgb']['core']['fold_candidates']:.4f}",
                     "rgb fold targets comb/core": f"{r['rgb']['combined']['fold_targets']:.4f} / {r['rgb']['core']['fold_targets']:.4f}",
                     "cov fold/crease comb": f"{r['roi_coverage']['combined']['fold']:.4f} / {r['roi_coverage']['combined']['crease']:.4f}",
                     "cov fold/crease core": f"{r['roi_coverage']['core']['fold']:.4f} / {r['roi_coverage']['core']['crease']:.4f}",
                     "ledge/ridge core": "/".join(f"{v:.3f}" for v in r["crease_rois"]["core"].values()),
                     "texture label core": round(r["core_texture_label"]["share"], 4),
                     "geom_fp core": round(r["core_geom_fp"]["share"], 4),
                     **{f"tex comb {k}": round(v, 4) for k, v in r["texture_rois"]["combined"].items()},
                     **{f"tex core {k}": round(v, 4) for k, v in r["texture_rois"]["core"].items()},
                     "named 15/97/124": r["named"]})
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
    print(pd.DataFrame(rows).set_index("run").T.to_string())
    for n in (NEW, C6):
        print(n, "constraints", rep["runs"][n]["constraints"], "meets", rep["runs"][n]["meets"],
              "closed ids", rep["runs"][n]["blocked_closed_ids"])
    print("split:", json.dumps(rep["split"], indent=1, default=float))


if __name__ == "__main__":
    main()
