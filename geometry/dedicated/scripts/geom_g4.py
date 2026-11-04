"""TASK_GEOM_INTEGRATION G4: labelled audit (audit_split/), re-evaluation, Laplacian decision, items 6/7 (diagnostic).

Labels: baseline_output/diag/geom/audit_split/audit_labels.csv, columns id, stem, direction, area_px, bbox, label, tags,
bucket, note. All counts use `bucket` (note is free text, quoted for examples only):
  tri-only:  gap_fixable, gap_unresolved, gap_subscale (geom_v0 misses), tri_band_excess, tri_fp, unsure
  geom-only: geom_gain, geom_offset, geom_band_excess, geom_fp, unsure

1. Estimator (per direction). The sample is pps with replacement within strata h = image x direction, so the
   unweighted fraction of a stratum's draws (with multiplicity, audit_draws.csv) estimates that stratum's area share:
       share_c = sum_h A_h p_hc / sum_h A_h,   A_h = full unsampled stratum area (audit_strata.csv).
   95 % CI: stratified bootstrap over draws (resample each stratum's n_h draws with replacement, B = 10000, seed 0;
   strata with n_h = 1 contribute no variance, so the CI is understated; marked in strata.csv). The pooled draw fraction (sum of indicator / number of draws)
   is reported as a check (self-weighting up to the allocation's rounding and minimum of one draw). Area = share x the
   G1 total of the direction (network px; also as a share of all image px).
2. Re-evaluation run geom_g4_reeval (configs/candidate_geom_g4_reeval.yaml: geom_v0 + sky_band depth_and_mask +
   crease_rescue laplacian_corroborated 8 deg / k_L 3). A unit is closed if the run's w_net < 0.5 on >= 50 % of its
   pixels. Attribution: depth (its depth_w_net alone closes it), fold (fold_w_net alone), or min (only the combination).
3. Laplacian rule: missed true geometry = gap_fixable + gap_unresolved not closed by the re-evaluation, as a share of
   the tri-only area; gap_subscale reported separately and excluded. (a) < 10 % -> retire; else (b) if the missed area is
   mostly in a class the rescue targets (crease), else (c).
4. Open step gaps (label step, bucket gap_fixable | gap_unresolved, not closed): the depth candidates of the run within
   2 px of the unit (all rows of <stem>_depth_candidates.parquet; their reason) -> the stage that drops the edge, taken as
   the most advanced stage any candidate reached: no candidate (split by the gradient floor: max g_n / floor in the
   unit), D-3 tests (polarity, height, sharpness, centring), sky_near, sign (plane fits disagree with the gradient),
   signif (significance gate), rel (tau_low), component (hysteresis / size), accepted but w_c >= 0.5 (small Delta Z),
   accepted nearby but outside the unit (band too narrow). tau_low and the significance gate are separate stages; a
   corroborated depth rescue (proposal (b)) would target the significance gate, so it is proposed if signif dominates.
5. geom_offset: the signed offset s* of the RGB edge along the edge normal (scripts/rgb_edges.py, |s| <= 8) at
   geom_v0's accepted candidates whose own band reaches the unit. Sign: depth edges + toward the near side; folds + toward image up for
   near-horizontal edges (contact folds) and + toward the more frontal face (larger |n_z|) for near-vertical edges
   (vertical corners). Sub-type from the tags (vertical-corner, contact, nosing; contact and nosing use image up) where
   given, else from the orientation. Per unit also the raw offset (dx, dy) = s_raw n in the upright frame and the
   recorded fold convexity (majority).
6. Items 6/7: the narrow-face units (gaps 15, 37, 92; gain 171) after the re-evaluation, and the fold gaps on textured
   surfaces (tag textured-surface), with the reopen rule.

Writes baseline_output/diag/geom/g4/{g4.json, units.csv, estimates.csv}.
Run from geometry/dedicated:  env -u PYTHONPATH uv run python scripts/geom_g4.py [--labels CSV]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rgb_edges import luminance, rgb_offset  # noqa: E402
from illum_baseline.config import dense_fit_params, detector_params, load_config  # noqa: E402
from illum_baseline.detector import disk  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.candidates import depth_field  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.depth import DepthParams  # noqa: E402
from illum_baseline.stage_b import load_upright  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
AUD = ROOT / "baseline_output" / "diag" / "geom" / "audit_split"
OUT = ROOT / "baseline_output" / "diag" / "geom" / "g4"
V0, TRI, NEW = "geom_v0", "tri_veto4", "geom_g4_reeval"
BUCKETS = {"tri-only": ["gap_fixable", "gap_unresolved", "gap_subscale", "tri_band_excess", "tri_fp", "unsure"],
           "geom-only": ["geom_gain", "geom_offset", "geom_band_excess", "geom_fp", "unsure"]}
LABELS = ["step", "limb", "fold", "crease", "texture", "artifact", "clutter", "unsure"]
GAPS = ("gap_fixable", "gap_unresolved")
NARROW = {"gaps": [15, 37, 92], "gain": [171]}
B, SEED, Z = 10000, 0, 3.29
D3 = {"polarity", "height", "sharpness", "centring", "symmetry"}
PROGRESS = ["no candidate (below gradient floor)", "no candidate (above floor)", "D-3 tests", "sky_near", "sign",
            "signif", "rel (tau_low)", "component", "accepted, w_c >= 0.5 (small dZ)", "accepted nearby (band too narrow)"]


# ---------------------------------------------------------------- estimator

class Estimator:
    """Stratified pps-with-replacement area shares for one direction, with a joint stratified bootstrap."""

    def __init__(self, draws: pd.DataFrame, strata: pd.DataFrame, total_px: float, image_px: float, rng):
        self.strata = strata[strata["A_h"] > 0].reset_index(drop=True)
        self.A = self.strata["A_h"].to_numpy(float)
        self.groups = [np.flatnonzero(draws["stem"].to_numpy() == s) for s in self.strata["stem"]]
        assert all(len(g) == n for g, n in zip(self.groups, self.strata["n_h"])), "draws do not match the strata"
        self.idx = [rng.integers(0, len(g), size=(B, len(g))) for g in self.groups]
        self.n = len(draws)
        self.total_px, self.image_px = total_px, image_px
        self.n_single = int(sum(len(g) == 1 for g in self.groups))

    def run(self, ind: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        names = list(ind)
        X = np.stack([np.asarray(ind[k], float) for k in names], 1)  # draws x K
        est = np.zeros(len(names))
        reps = np.zeros((B, len(names)))
        for a, g, ix in zip(self.A, self.groups, self.idx):
            Xg = X[g]
            est += a * Xg.mean(0)
            reps += a * Xg[ix].mean(1)
        est /= self.A.sum()
        reps /= self.A.sum()
        pooled = X.mean(0)
        rows = []
        for j, k in enumerate(names):
            lo, hi = np.percentile(reps[:, j], [2.5, 97.5])
            rows.append({"quantity": k, "n_draws": int(X[:, j].sum()), "share": est[j], "ci_lo": lo, "ci_hi": hi,
                         "pooled_fraction": pooled[j], "area_px": est[j] * self.total_px,
                         "share_of_image_px": est[j] * self.total_px / self.image_px})
        return pd.DataFrame(rows), {k: reps[:, j] for j, k in enumerate(names)} | {f"_est:{k}": est[j]
                                                                                   for j, k in enumerate(names)}


# ---------------------------------------------------------------- per-image data

def stage_a_ctx(root: Path, stem: str, cfg):
    meta_a = json.loads((root / "stage_a" / f"{stem}.json").read_text())
    z = np.load(root / "stage_a" / f"{stem}.npz")
    return GeometryContext(z["D"], z["sky_prob"].astype(np.float64), Intrinsics.from_stage_a(meta_a),
                           detector_params(cfg), dense_fit_params(cfg))


def load_runs(batch: str, stem: str) -> dict:
    root = BATCHES[batch]
    out = {}
    for run in (V0, TRI, NEW):
        rd = root / "stage_b" / run
        cfg = load_config(rd / "config_resolved.yaml")
        a, _ = load_upright(cfg, rd, stem)
        out[run] = {k: a[k].astype(np.float64) for k in ("w_net", "depth_w_net", "fold_w_net") if k in a}
        out[run]["cfg"] = cfg
    for run in (V0, NEW):
        rd = root / "stage_b" / run
        out[run]["depth"] = pd.read_parquet(rd / f"{stem}_depth_candidates.parquet")
        out[run]["fold"] = pd.read_parquet(rd / f"{stem}_fold_candidates.parquet")
    return out


def in_mask(m: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    H, W = m.shape
    u, v = np.clip(df["u"].to_numpy(), 0, W - 1), np.clip(df["v"].to_numpy(), 0, H - 1)
    return m[v, u]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(AUD / "audit_labels.csv"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    lab = pd.read_csv(args.labels, dtype={"tags": str, "note": str, "bucket": str, "label": str})
    for c in ("tags", "note"):
        lab[c] = lab[c].fillna("") if c in lab else ""
    lab["label"] = lab["label"].str.strip()
    lab["bucket"] = lab["bucket"].str.strip()
    bad = [(r.id, r.direction, r.label, r.bucket) for r in lab.itertuples()
           if r.bucket not in BUCKETS[r.direction] or r.label not in LABELS]
    if bad:
        raise SystemExit(f"labels with an unknown bucket/label for their direction: {bad}")
    draws = pd.read_csv(AUD / "audit_draws.csv")
    strata = pd.read_csv(AUD / "audit_strata.csv")
    totals = json.loads((AUD / "audit_totals.json").read_text())
    masks = np.load(AUD / "audit_masks.npz")
    batch_of = dict(zip(strata["stem"], strata["batch"]))

    # ---- per-unit re-evaluation and diagnostics
    units = []
    open_step_cands = []
    offsets = []
    by_stem = lab.groupby("stem")
    for stem, g in by_stem:
        batch = batch_of[stem]
        R = load_runs(batch, stem)
        shape = R[V0]["w_net"].shape
        need_field = ((g["label"] == "step") & g["bucket"].isin(GAPS)).any()
        field, floor = None, None
        Y = None
        for r in g.itertuples():
            m = np.zeros(shape[0] * shape[1], bool)
            m[masks[f"id{r.id:03d}"]] = True
            m = m.reshape(shape)
            assert tuple(masks[f"id{r.id:03d}_shape"]) == shape and m.sum() == r.area_px
            fr = {run: float((R[run]["w_net"][m] < 0.5).mean()) for run in (V0, TRI, NEW)}
            fd = float((R[NEW]["depth_w_net"][m] < 0.5).mean())
            ff = float((R[NEW]["fold_w_net"][m] < 0.5).mean())
            closed = fr[NEW] >= 0.5
            by = "" if not closed else ("depth" if fd >= 0.5 else "fold" if ff >= 0.5 else "min")
            row = {"id": r.id, "batch": batch, "stem": stem, "direction": r.direction, "area_px": r.area_px,
                   "label": r.label, "bucket": r.bucket, "tags": r.tags, "frac_v0": fr[V0], "frac_tri": fr[TRI],
                   "frac_new": fr[NEW], "frac_new_depth": fd, "frac_new_fold": ff, "closed": closed,
                   "closed_by": by, "stage": "", "note": r.note}
            # 4. open step gaps: where does the depth channel drop the edge?
            if r.direction == "tri-only" and r.label == "step" and r.bucket in GAPS and not closed:
                near = ndimage.binary_dilation(m, structure=disk(2))
                dc = R[NEW]["depth"]
                c = dc[in_mask(near, dc)].copy()
                c_in = in_mask(m, c)
                c["id"], c["in_unit"] = r.id, c_in
                open_step_cands.append(c)
                if len(c) == 0:
                    if field is None:
                        cfg = R[NEW]["cfg"]
                        p = DepthParams.from_dict(dataclasses.asdict(cfg.geometry.depth))
                        field = depth_field(stage_a_ctx(BATCHES[batch], stem, cfg), p)
                        floor = p.k_g * field.sigma_g
                    ratio = float(np.nanmax(np.where(near, field.g_n, np.nan)) / floor)
                    row["gn_over_floor"] = ratio
                    stage = PROGRESS[0] if ratio <= 1 else PROGRESS[1]
                else:
                    rs = c["reason"].to_numpy()
                    lvl = []
                    for rr, inside, wc in zip(rs, c_in, c["w_c"].to_numpy()):
                        if rr in D3:
                            lvl.append(2)
                        elif rr == "accepted":
                            lvl.append(8 if inside and wc >= 0.5 else 9)  # inside with w_c < 0.5: partial cover
                        else:
                            lvl.append({"sky_near": 3, "sign": 4, "signif": 5, "rel": 6, "component": 7}[rr])
                    stage = PROGRESS[max(lvl)]
                    with np.errstate(invalid="ignore"):
                        row["n_cand"] = int(len(c))
                        row["n_fail_signif"] = int((c["signif_add"] <= Z).sum())
                        row["n_fail_rel"] = int((c["r"] <= 0.02).sum())
                    row["reasons"] = json.dumps(c["reason"].value_counts().to_dict())
                    row["fit_modes"] = json.dumps(c.loc[c["reason"].isin(["signif", "rel", "component"]), "fit_mode"]
                                                  .value_counts().to_dict())
                row["stage"] = stage
            # 5. geom_offset: signed RGB-edge offset at geom_v0's accepted candidates near the unit
            if r.direction == "geom-only" and r.bucket == "geom_offset":
                if Y is None:
                    Y = luminance(BATCHES[batch] / "stage_a", stem)
                dist = ndimage.distance_transform_edt(~m)   # px to the unit
                tags = set(r.tags.split("|"))
                tag_sub = next((x for x in ("vertical-corner", "contact", "nosing") if x in tags), None)
                for ch in ("depth", "fold"):
                    t = R[V0][ch]
                    H_, W_ = dist.shape
                    dd = dist[np.clip(t["v"].to_numpy(), 0, H_ - 1), np.clip(t["u"].to_numpy(), 0, W_ - 1)]
                    t = t[t["accepted"].to_numpy() & (dd <= t["band"].to_numpy())]   # its own band reaches the unit
                    if not len(t):
                        continue
                    x0 = t[["u0", "v0"]].to_numpy()
                    n = t[["n_u", "n_v"]].to_numpy() if ch == "depth" else t[["e1_u", "e1_v"]].to_numpy()
                    horiz = np.abs(n[:, 1]) > np.abs(n[:, 0])   # normal mostly vertical: near-horizontal edge
                    if tag_sub in ("contact", "nosing"):
                        horiz = np.ones(len(t), bool)
                    elif tag_sub == "vertical-corner":
                        horiz = np.zeros(len(t), bool)
                    if ch == "depth":
                        sgn = np.sign(t["rho_p"].to_numpy() - t["rho_m"].to_numpy())
                        rule = np.full(len(t), "near side")
                        conv = np.zeros(len(t), int)
                    else:
                        sgn_h = -np.sign(n[:, 1])
                        sgn_v = np.sign(np.abs(t["n_p_z"].to_numpy()) - np.abs(t["n_m_z"].to_numpy()))
                        sgn = np.where(horiz, sgn_h, sgn_v)
                        rule = np.where(horiz, "image up", "frontal face")
                        conv = t["convexity"].to_numpy()
                    s_raw = rgb_offset(Y, x0, n)
                    for j in range(len(t)):
                        sub = tag_sub or ("contact" if horiz[j] else "vertical-corner")
                        offsets.append({"id": r.id, "stem": stem, "channel": ch, "subtype": sub,
                                        "subtype_source": "tag" if tag_sub else "orientation", "sign_rule": rule[j],
                                        "s": float(s_raw[j] * sgn[j]), "s_raw": float(s_raw[j]),
                                        "dx": float(s_raw[j] * n[j, 0]), "dy": float(s_raw[j] * n[j, 1]),
                                        "convexity": int(conv[j]), "tags": r.tags})
            units.append(row)
    U = pd.DataFrame(units).sort_values("id").reset_index(drop=True)
    U.to_csv(out / "units.csv", index=False)

    # monotonicity check: the options only lower w
    mono = []
    for stem in lab["stem"].unique():
        R = load_runs(batch_of[stem], stem)
        mono.append(bool((R[NEW]["w_net"] <= R[V0]["w_net"]).all()))
    # new w < 0.5 area added by the re-evaluation, inside / outside tri_veto4's w < 0.5
    added = {"inside_tri": 0, "outside_tri_ge_0.9": 0, "other": 0, "geom_only_v0": 0, "geom_only_new": 0}
    for batch, root in BATCHES.items():
        for stem in sorted({s for s, b in batch_of.items() if b == batch}):
            R = load_runs(batch, stem)
            a = (R[NEW]["w_net"] < 0.5) & (R[V0]["w_net"] >= 0.5)
            tw = R[TRI]["w_net"]
            added["inside_tri"] += int((a & (tw < 0.5)).sum())
            added["outside_tri_ge_0.9"] += int((a & (tw >= 0.9)).sum())
            added["other"] += int((a & (tw >= 0.5) & (tw < 0.9)).sum())
            added["geom_only_v0"] += int(((R[V0]["w_net"] < 0.5) & (tw >= 0.9)).sum())
            added["geom_only_new"] += int(((R[NEW]["w_net"] < 0.5) & (tw >= 0.9)).sum())

    # ---- estimates per direction
    D = draws.merge(U[["id", "label", "bucket", "closed", "closed_by"]], on="id", how="left")
    assert D["bucket"].notna().all()
    tot = pd.DataFrame(totals["per_image"])
    est_rows, reps_all, summary, strata_tables = [], {}, {}, []
    for direction in ("tri-only", "geom-only"):
        d = D[D["direction"] == direction].reset_index(drop=True)
        st = strata[strata["direction"] == direction]
        t = tot[tot["direction"] == direction]
        E = Estimator(d, st, float(t["area_px"].sum()), float(t["image_px"].sum()), rng)
        ind = {}
        for b in BUCKETS[direction]:
            ind[f"bucket:{b}"] = d["bucket"] == b
            ind[f"bucket:{b}|closed"] = (d["bucket"] == b) & d["closed"]
        for lb in LABELS:
            if (d["label"] == lb).any():
                ind[f"label:{lb}"] = d["label"] == lb
        for b in BUCKETS[direction]:
            for lb in LABELS:
                k = (d["bucket"] == b) & (d["label"] == lb)
                if k.any():
                    ind[f"cell:{b}|{lb}"] = k
                    ind[f"cell:{b}|{lb}|closed"] = k & d["closed"]
        if direction == "tri-only":
            gap = d["bucket"].isin(GAPS)
            ind["missed_before"] = gap
            ind["missed_after"] = gap & ~d["closed"]
            ind["subscale_after"] = (d["bucket"] == "gap_subscale") & ~d["closed"]
            for lb in ("step", "limb", "fold", "crease"):
                ind[f"missed_after|{lb}"] = gap & ~d["closed"] & (d["label"] == lb)
            ind["unsure"] = d["bucket"] == "unsure"
        df, reps = E.run(ind)
        df.insert(0, "direction", direction)
        est_rows.append(df)
        reps_all[direction] = reps
        stt = E.strata[["batch", "stem", "direction", "A_h", "N_h", "n_h", "n_unique"]].copy()
        stt["singleton"] = stt["n_h"] == 1
        strata_tables.append(stt)
        summary[direction] = {"singleton_area_share": float(stt.loc[stt["singleton"], "A_h"].sum() / stt["A_h"].sum()),
                              "n_draws": E.n, "n_strata": int(len(E.A)), "n_strata_single_draw": E.n_single,
                              "total_area_px": E.total_px, "image_px": E.image_px,
                              "n_units": int((U["direction"] == direction).sum())}
    EST = pd.concat(est_rows, ignore_index=True)
    EST.to_csv(out / "estimates.csv", index=False)
    STT = pd.concat(strata_tables, ignore_index=True)
    STT.to_csv(out / "strata.csv", index=False)

    # ---- Laplacian rule (ratio excluding unsure with the joint bootstrap)
    r = reps_all["tri-only"]
    missed = r["_est:missed_after"]
    excl = missed / (1 - r["_est:unsure"])
    excl_reps = r["missed_after"] / (1 - r["unsure"])
    m_lab = {lb: r[f"_est:missed_after|{lb}"] for lb in ("step", "limb", "fold", "crease")}
    crease_frac = m_lab["crease"] / missed if missed > 0 else 0.0
    if missed < 0.10:
        option = "a"
    elif crease_frac >= 0.5:
        option = "b"
    else:
        option = "c"
    lap = {"missed_before": r["_est:missed_before"],
           "missed_before_ci": np.percentile(r["missed_before"], [2.5, 97.5]).tolist(),
           "missed_after": missed, "missed_after_ci": np.percentile(r["missed_after"], [2.5, 97.5]).tolist(),
           "missed_after_excl_unsure": excl, "missed_after_excl_unsure_ci": np.percentile(excl_reps, [2.5, 97.5]).tolist(),
           "missed_after_by_label": m_lab, "crease_share_of_missed": crease_frac,
           "subscale_after": r["_est:subscale_after"],
           "subscale_after_ci": np.percentile(r["subscale_after"], [2.5, 97.5]).tolist(),
           "threshold": 0.10, "ci_upper_below_threshold": bool(np.percentile(r["missed_after"], 97.5) < 0.10),
           "option": option}

    # ---- open step gaps
    S = U[(U["direction"] == "tri-only") & (U["label"] == "step") & U["bucket"].isin(GAPS) & ~U["closed"]]
    stage_counts = S["stage"].value_counts().reindex(PROGRESS).fillna(0).astype(int).to_dict()
    stage_area = S.groupby("stage")["area_px"].sum().reindex(PROGRESS).fillna(0).astype(int).to_dict()
    gate = stage_counts["signif"]   # a corroborated depth rescue targets the significance gate
    C = pd.concat(open_step_cands, ignore_index=True) if open_step_cands else pd.DataFrame()
    comp = {}
    if len(C):
        C.to_parquet(out / "open_step_candidates.parquet")
        # component stage: the unit's best candidates passed every gate but their contour component failed
        best = C[C["id"].isin(S.loc[S["stage"] == "component", "id"]) & (C["reason"] == "component")]
        small = best["component_px"] < 5
        comp = {"n_cand": int(len(best)), "share_component_lt_5px": float(small.mean()),
                "share_no_tau_high_seed": float((~small & (best["comp_r_max"] < 0.04)).mean()),
                "component_px_P50": float(best["component_px"].median()), "r_P50": float(best["r"].median()),
                "comp_r_max_P50": float(best["comp_r_max"].median()),
                "fit_modes": best["fit_mode"].value_counts().to_dict(), "thin_share": float(best["thin"].mean())}
    steps = {"n_open_units": int(len(S)), "stage_counts": stage_counts, "stage_area_px": stage_area,
             "signif_units": int(gate), "rel_units": int(stage_counts["rel (tau_low)"]),
             "signif_dominates": bool(gate > len(S) / 2),
             "candidate_reasons": C["reason"].value_counts().to_dict() if len(C) else {}, "component_stage": comp,
             "propose_b_depth_rescue": bool(gate > len(S) / 2)}

    # ---- offsets
    O = pd.DataFrame(offsets)
    off = {}
    if len(O):
        O.to_csv(out / "offsets.csv", index=False)
        for (sub, ch), g in O.groupby(["subtype", "channel"]):
            sv = g["s"].dropna()
            off[f"{sub}|{ch}"] = {"n_cand": int(len(sv)), "n_units": int(g["id"].nunique()),
                                  "median": float(sv.median()), "P25": float(sv.quantile(0.25)),
                                  "P75": float(sv.quantile(0.75)), "mean": float(sv.mean()),
                                  "share_abs_le_1": float((sv.abs() <= 1).mean()),
                                  "share_abs_1_3": float(((sv.abs() > 1) & (sv.abs() <= 3)).mean()),
                                  "share_abs_gt_3": float((sv.abs() > 3).mean()),
                                  "share_at_search_limit": float((sv.abs() >= 7.99).mean())}
        conv_name = {1: "convex", -1: "concave", 0: "none"}
        per_unit = []
        for (uid, ch), g in O.groupby(["id", "channel"]):
            cv = g["convexity"].value_counts()
            per_unit.append({"id": int(uid), "stem": g["stem"].iloc[0], "channel": ch, "subtype": g["subtype"].iloc[0],
                             "source": g["subtype_source"].iloc[0], "sign_rule": g["sign_rule"].mode().iloc[0],
                             "n_cand": int(len(g)), "s_median": float(g["s"].median()),
                             "dx_median": float(g["dx"].median()), "dy_median": float(g["dy"].median()),
                             "convexity": "" if ch == "depth" else
                             f"{conv_name[int(cv.index[0])]} ({int(cv.iloc[0])}/{len(g)})"})
        OU = pd.DataFrame(per_unit)
        OU.to_csv(out / "offsets_per_unit.csv", index=False)
        off["per_unit"] = OU.to_dict(orient="records")

    # ---- items 6/7
    nar = U.set_index("id")
    items = {"narrow_face": {str(i): {k: (nar.loc[i, k].item() if hasattr(nar.loc[i, k], "item") else nar.loc[i, k])
                                      for k in ("stem", "direction", "label", "bucket", "frac_new", "closed",
                                                "closed_by")} for i in NARROW["gaps"] + NARROW["gain"] if i in nar.index}}
    open_narrow = [i for i in NARROW["gaps"] if i in nar.index and not nar.loc[i, "closed"]]
    items["item7_reopen"] = bool(open_narrow)
    items["item7_open_units"] = open_narrow
    fold_gaps = U[(U["direction"] == "tri-only") & U["label"].isin(["fold", "crease"]) & U["bucket"].isin(GAPS)]
    tex = fold_gaps[fold_gaps["tags"].str.split("|").apply(lambda t: "textured-surface" in t) & ~fold_gaps["closed"]]
    items["item6_textured_fold_gaps"] = fold_gaps[fold_gaps["tags"].str.split("|").apply(
        lambda t: "textured-surface" in t)][["id", "label", "bucket", "frac_new", "closed", "closed_by"]].to_dict(
        orient="records")
    items["item6_textured_fold_gaps_open"] = tex["id"].tolist()
    items["item6_reopen"] = bool(len(tex))

    examples = {f"{d}|{b}": [f"{r.id} {r.stem[-3:]} {r.label}: {r.note[:110]}" for r in g.head(3).itertuples()]
                for (d, b), g in U.sort_values("area_px", ascending=False).groupby(["direction", "bucket"])}
    rep = {"examples": examples, "closed_units": U.loc[U["closed"] & (U["direction"] == "tri-only"),
                                                        ["id", "stem", "label", "bucket", "frac_new", "closed_by"]]
           .to_dict(orient="records"), "summary": summary, "monotone_w_new_le_v0": all(mono), "added_w_lt_0.5_px": added, "laplacian": lap,
           "open_step_gaps": steps, "geom_offset": off, "items_6_7": items,
           "closures": U.groupby(["direction", "bucket", "label"]).agg(n=("id", "size"), closed=("closed", "sum"),
                                                                       area_px=("area_px", "sum")).reset_index()
           .to_dict(orient="records"),
           "closed_by": U[U["closed"]].groupby(["direction", "closed_by"]).size().to_dict(),
           "params": {"labels": str(args.labels), "run": NEW, "B": B, "seed": SEED, "closed_if": "w < 0.5 on >= 50 %",
                      "step_gap_dilation_px": 2, "offset_candidates": "accepted, own band reaches the unit"}}
    rep["closed_by"] = {f"{k[0]}|{k[1]}": int(v) for k, v in rep["closed_by"].items()}
    (out / "g4.json").write_text(json.dumps(rep, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))

    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 60)
    print(json.dumps(summary, indent=1))
    print(STT.assign(singleton=np.where(STT["singleton"], "* (n_h = 1: no CI contribution)", "")).to_string(index=False))
    show = EST[~EST["quantity"].str.startswith("cell:")]
    print(show.round(4).to_string(index=False))
    print("\nclosures per bucket x label:")
    print(pd.DataFrame(rep["closures"]).to_string(index=False))
    print("closed_by:", rep["closed_by"], " monotone:", rep["monotone_w_new_le_v0"], " added:", added)
    print("\nLaplacian:", json.dumps(lap, indent=1, default=float))
    print("\nopen step gaps:", json.dumps(steps, indent=1))
    print(S[["id", "stem", "area_px", "stage", "frac_new"] + [c for c in ("reasons", "fit_modes", "gn_over_floor")
                                                                if c in S]].to_string(index=False))
    print("\noffsets:", json.dumps(off, indent=1))
    print("\nclosed tri-only units:", json.dumps(rep["closed_units"], indent=0, default=str))
    print("\nexamples:", json.dumps(examples, indent=1))
    print("\nitems 6/7:", json.dumps(items, indent=1, default=str))


if __name__ == "__main__":
    main()
