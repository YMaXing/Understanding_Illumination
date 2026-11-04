"""TASK_GEOM_V2 V1.2: where does the fold channel drop the fold / crease gaps? (diagnostic only)

Units: audit_split/ units labelled fold or crease, bucket gap_fixable or gap_unresolved, open under geom_v1's core w
(min(depth_w_net, fold_w_net) < 0.5 on < 50 % of the unit). Fold candidates: every row of geom_v1's
<stem>_fold_candidates.parquet whose pixel lies within 2 px of the unit. Each unit gets the furthest stage any of its
candidates reached (the V0 ruling's order):
  0 no candidate           (no row; max fold_r_n within 2 px / the run's rate_threshold: below / above the NMS floor)
  1 window validity / contamination   reason invalid_window
  2 planarity              reason gate, a side on the dense-normal fallback (fallback_p | fallback_m)
  3 chi2 gate              reason gate, both sides planar
  4 capped concentration veto         reasons veto, veto_undecidable
  5 component minimum      reason component
  6 accepted, w >= 0.5 (J <= J_min + J0)   accepted or rescued with w_c >= 0.5 (Delta theta below ~20 deg)
  7 accepted, w < 0.5 (band misses the unit)
Gate units (2, 3) are split by why the crease rescue did not take their gate failures (the furthest rescue condition
any of them met: Delta theta < 8 deg | J < J_min | rescue veto | rescue component < 10 px | support < 0.7).
Also, over all candidates of the open units: the candidate-level stage counts, each unit's modal stage, and for every
gate failure the rescue condition that stops it (candidate level; veto split decided / undecidable).
Writes baseline_output/diag/geom/v2/fold_gaps.json and fold_gaps_units.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geom_g4 import GAPS  # noqa: E402
from geom_score import OUT, Run, load_labels, unit_fractions  # noqa: E402
from illum_baseline.detector import disk  # noqa: E402

RUN = "geom_v1"
STAGES = ["no candidate (below the NMS floor)", "no candidate (above the NMS floor)", "window validity / contamination",
          "planarity", "chi2 gate", "capped concentration veto", "component minimum",
          "accepted, w >= 0.5 (J <= J_min + J0)", "accepted, w < 0.5 (band misses the unit)"]
RESCUE = ["dtheta < 8 deg", "J < J_min", "rescue veto", "rescue component < 10 px", "rescue support < 0.7"]
TEXTURED, NARROW = (97, 124, 125), (15, 37)


def cand_stage(c: pd.DataFrame) -> np.ndarray:
    r = c["reason"].to_numpy()
    fb = (c["fallback_p"] | c["fallback_m"]).to_numpy()
    acc = np.isin(r, ["accepted", "rescued"])
    s = np.full(len(c), 2)
    s[(r == "gate") & fb] = 3
    s[(r == "gate") & ~fb] = 4
    s[np.isin(r, ["veto", "veto_undecidable"])] = 5
    s[r == "component"] = 6
    s[acc & (c["w_c"].to_numpy() >= 0.5)] = 7
    s[acc & (c["w_c"].to_numpy() < 0.5)] = 8
    return s


def rescue_stage(g: pd.DataFrame, theta_deg: float, J_min: float) -> int:
    """Furthest rescue condition met by the unit's gate failures (index into RESCUE)."""
    best = 0
    for c in g.itertuples():
        if not c.rescue_eligible:
            k = 0 if np.degrees(c.dtheta) < theta_deg else 1
        elif not c.rescue_veto_pass:
            k = 2
        elif not (c.rescue_comp_px >= 10):
            k = 3
        else:
            k = 4
        best = max(best, k)
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab, _, _, _, masks = load_labels()
    run = Run(RUN)
    U = unit_fractions(run, lab, masks)
    sel = U[(U["direction"] == "tri-only") & U["label"].isin(["fold", "crease"]) & U["bucket"].isin(GAPS)]
    open_ = sel[sel["frac_core"] < 0.5]
    rows, dtheta_acc, cand_rows = [], {}, []
    for stem, g in open_.groupby("stem"):
        b = run.batch_of(stem)
        rd = run.cfg[b].out_dir / "stage_b" / RUN
        fc = pd.read_parquet(rd / f"{stem}_fold_candidates.parquet")
        meta = json.loads((rd / f"{stem}.json").read_text())
        thr = meta["channels"]["fold"]["rate_threshold"]
        pf = meta["params"]["geometry"]["fold"]
        from illum_baseline.stage_b import load_upright

        a, _ = load_upright(run.cfg[b], rd, stem)
        rn = a["fold_r_n"].astype(np.float64)
        shape = rn.shape
        H, W = shape
        for u in g.itertuples():
            m = np.zeros(H * W, bool)
            m[masks[f"id{u.id:03d}"]] = True
            near = ndimage.binary_dilation(m.reshape(shape), structure=disk(2))
            inside = near[np.clip(fc["v"], 0, H - 1), np.clip(fc["u"], 0, W - 1)]
            c = fc[inside]
            row = {"id": u.id, "stem": stem, "label": u.label, "bucket": u.bucket, "tags": u.tags,
                   "area_px": u.area_px, "frac_core": u.frac_core, "frac_combined": u.frac_combined,
                   "n_cand": int(len(c))}
            if not len(c):
                ratio = float(np.nanmax(np.where(near, rn, np.nan)) / thr) if np.isfinite(rn[near]).any() else 0.0
                row["rn_over_floor"] = ratio
                st = 0 if ratio <= 1 else 1
            else:
                cs = cand_stage(c)
                st = int(cs.max())
                row["modal_stage"] = STAGES[int(pd.Series(cs).value_counts().idxmax())]
                for x, sx in zip(c.itertuples(), cs):
                    blk = None
                    if x.reason == "gate":
                        if not x.rescue_eligible:
                            blk = "dtheta < 8 deg" if np.degrees(x.dtheta) < pf["rescue_theta_deg"] else "J < J_min"
                        elif not x.rescue_veto_pass:
                            blk = "rescue veto (" + ("decided" if x.decidable else "undecidable") + ")"
                        elif not x.rescue_comp_px >= 10:
                            blk = "rescue component < 10 px"
                        else:
                            blk = "rescue support < 0.7"
                    cand_rows.append({"unit": u.id, "stage": STAGES[sx], "rescue_block": blk,
                                      "dtheta_deg": float(np.degrees(x.dtheta)), "T_over_lambda": x.T_over_lambda,
                                      "fallback": bool(x.fallback_p or x.fallback_m)})
                row["reasons"] = json.dumps(c["reason"].value_counts().to_dict())
                row["dtheta_deg_P50"] = float(np.degrees(c["dtheta"].median()))
                if st in (3, 4):
                    gf = c[c["reason"] == "gate"]
                    row["rescue"] = RESCUE[rescue_stage(gf, pf["rescue_theta_deg"], pf["J_min"])]
                    row["gate_dtheta_deg_P50"] = float(np.degrees(gf["dtheta"].median()))
                if st == 7:
                    acc = c[c["reason"].isin(["accepted", "rescued"])]
                    dtheta_acc[u.id] = np.degrees(acc["dtheta"].to_numpy()).tolist()
                    row["acc_dtheta_deg_P50"] = float(np.median(dtheta_acc[u.id]))
                    row["acc_w_c_P50"] = float(acc["w_c"].median())
            row["stage"] = STAGES[st]
            rows.append(row)
    D = pd.DataFrame(rows).sort_values("id")
    D.to_csv(OUT / "fold_gaps_units.csv", index=False)
    tab = (D.groupby("stage").agg(units=("id", "size"), px=("area_px", "sum"))
           .reindex(STAGES).fillna(0).astype(int))
    tab["px share"] = tab["px"] / tab["px"].sum()
    resc = D[D["stage"].isin(STAGES[3:5])].groupby("rescue").agg(units=("id", "size"), px=("area_px", "sum"))
    C = pd.DataFrame(cand_rows)
    C.to_csv(OUT / "fold_gaps_candidates.csv", index=False)
    cand_tab = C.groupby("stage").agg(candidates=("unit", "size"), units=("unit", "nunique")).reindex(STAGES).fillna(0)
    modal = D.assign(modal_stage=D["modal_stage"].fillna(D["stage"])).groupby("modal_stage").agg(
        units=("id", "size"), px=("area_px", "sum"))
    g = C[C["rescue_block"].notna()]
    block = g.groupby("rescue_block").agg(candidates=("unit", "size"), units=("unit", "nunique"),
                                          dtheta_deg_P50=("dtheta_deg", "median"), fallback_share=("fallback", "mean"))
    gate_c = C[C["stage"] == "chi2 gate"]
    pooled = np.concatenate([np.asarray(v) for v in dtheta_acc.values()]) if dtheta_acc else np.array([])
    rep = {"n_units_selected": int(len(sel)), "n_open_core": int(len(open_)),
           "n_closed_core": int(len(sel) - len(open_)), "stages": tab.reset_index().to_dict(orient="records"),
           "rescue_split": resc.reset_index().to_dict(orient="records"),
           "accepted_w_ge_0.5_dtheta_deg": {"per_unit_median": {str(k): float(np.median(v)) for k, v in dtheta_acc.items()},
                                           "pooled_P10_P50_P90": np.percentile(pooled, [10, 50, 90]).tolist()
                                           if pooled.size else None, "n_candidates": int(pooled.size)},
           "candidate_stages": cand_tab.reset_index().to_dict(orient="records"),
           "modal_stages": modal.reset_index().to_dict(orient="records"),
           "rescue_block": block.reset_index().to_dict(orient="records"),
           "chi2_gate_candidates": {"dtheta_deg_P10_P50_P90": np.percentile(gate_c["dtheta_deg"].dropna(), [10, 50, 90])
                                    .tolist(), "T_over_lambda_P50": float(gate_c["T_over_lambda"].median())},
           "textured": D[D["id"].isin(TEXTURED)].to_dict(orient="records"),
           "narrow": D[D["id"].isin(NARROW)].to_dict(orient="records"),
           "open_ids_not_in_table": sorted(set(TEXTURED + NARROW) - set(D["id"]))}
    (OUT / "fold_gaps.json").write_text(json.dumps(rep, indent=1, default=float))
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(f"selected {len(sel)} fold/crease gap units; open under core {len(open_)}; closed {len(sel) - len(open_)}")
    print(tab.round(3).to_string())
    print("\nrescue split of the gate units:\n" + resc.to_string())
    print("\ncandidate-level stages:\n" + cand_tab.astype(int).to_string())
    print("\nmodal stage per unit:\n" + modal.to_string())
    print("\nwhat stops the rescue (gate failures, candidate level):\n" + block.round(3).to_string())
    print("chi2-gate candidates:", rep["chi2_gate_candidates"])
    print("\naccepted, w >= 0.5: dtheta (deg) pooled P10/P50/P90", rep["accepted_w_ge_0.5_dtheta_deg"])
    cols = [c for c in ("id", "stem", "label", "bucket", "tags", "area_px", "frac_core", "frac_combined", "n_cand",
                        "stage", "rescue", "dtheta_deg_P50", "acc_dtheta_deg_P50", "acc_w_c_P50", "rn_over_floor")
            if c in D]
    print(D[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
