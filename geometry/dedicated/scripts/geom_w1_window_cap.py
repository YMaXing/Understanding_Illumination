"""TASK_GEOM_V2 W1 (report and selection, propose only): geometry.fold.window_cap next_ridge (run geom_v1_1_wcap)
against geom_v1_1, both batches, scored with scripts/geom_score.py's functions on combined and core w.

Reports: the 070 middle riser (acceptance per fold, share of the k1 = 8 candidates now accepted; top and bottom risers
for comparison), the narrow-face units 15, 37, 92 (core and combined relaxed share), fold-target and fold-candidate
RGB-edge coverage, the 068 right-wall laps (how many windows the cap reaches), per-image accepted folds and w < 0.5.
Constraints (combined w unless stated):
  texture ROIs each <= geom_v1_1 + 0.01; 083 interior unchanged (|diff| < 0.0005);
  core relaxed share of the texture-labelled and the tri_fp units <= geom_v1_1 + 0.02;
  geom_fp (geom-only audit, relaxed per pixel) not increased, combined and core;
  step / limb / fold ROI coverage each >= geom_v1_1 - 0.01;
  the same verdict with the audit restricted to batch1 units.
Proposal: window_cap next_ridge if every constraint holds and each middle-riser fold reaches >= 90 % accepted.
Writes baseline_output/diag/geom/w1/window_cap.json.
W1b (--run geom_v1_1_wcapf, window_cap next_fold_ridge): also the capped share of candidates, accepted folds lost /
gained per image, and the added constraints: core crease coverage >= geom_v1_1 - 0.02, 088 rock ridge (core coverage)
>= 0.70, geom_gain (combined) >= geom_v1_1 - 0.01, core missed true geometry per pixel <= geom_v1_1 + 0.002, core RGB
fold-candidate coverage >= geom_v1_1 - 0.005. Writes w1/window_cap_<run>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geom_score import Run, g1_table, load_labels, rgb_edge_set  # noqa: E402
from geom_v5_fallback import BATCHES, metrics  # noqa: E402

REF = "geom_v1_1"
NEW = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else "geom_v1_1_wcap"
W1B = NEW != "geom_v1_1_wcap"
OUT = ROOT / "baseline_output" / "diag" / "geom" / "w1"
NARROW = (15, 37, 92)
RISERS = {"top": (344, 382, 362.5), "middle": (386, 414, 400.5), "bottom": (443, 478, 460.5)}
R068 = (284, 372, 32, 316)
ACC = ("accepted", "rescued")


def table(run, stem, batch="batch2"):
    return pd.read_parquet(BATCHES[batch] / "stage_b" / run / f"{stem}_fold_candidates.parquet")


def risers(run):
    fc = table(run, "acharya_mrudula_070")
    out = {}
    for name, (v0, v1, sp) in RISERS.items():
        d = fc[(fc.u0 >= 90) & (fc.u0 <= 290) & (fc.v0 >= v0) & (fc.v0 <= v1) & (fc.e1_v.abs() >= 0.85)]
        for fold, g in d.groupby(np.where(d.v0 < sp, "upper", "lower")):
            acc = g.reason.isin(ACC)
            r = {"n": len(g), "accepted_or_rescued": float(acc.mean()), "gate": float(g.gate.mean()),
                 "k1_8_n": int((g.k1 == 8).sum()),
                 "k1_8_accepted": float(acc[g.k1 == 8].mean()) if (g.k1 == 8).any() else float("nan"),
                 "reason": {k: int(v) for k, v in g.reason.value_counts().items()}}
            if "window_capped_p" in g:
                cap = g.window_capped_p | g.window_capped_m
                r |= {"window_capped": float(cap.mean()),
                      "m_fit_capped_side": {int(k): int(v) for k, v in pd.concat(
                          [g.m_p_fit[g.window_capped_p], g.m_m_fit[g.window_capped_m]]).value_counts().items()},
                      "d_ridge_median": float(np.nanmedian(np.where(np.isfinite(g.d_ridge_p), g.d_ridge_p,
                                                                     g.d_ridge_m)))}
            out[f"{name} {fold}"] = r
    return out


def laps068(run):
    fc = table(run, "acharya_mrudula_068")
    d = fc[(fc.u0 >= R068[0]) & (fc.u0 <= R068[1]) & (fc.v0 >= R068[2]) & (fc.v0 <= R068[3])
           & (fc.e1_v.abs() >= 0.85)]
    r = {"n": len(d), "reason": {k: int(v) for k, v in d.reason.value_counts().items()}}
    if "window_capped_p" in d:
        dr = np.fmin(d.d_ridge_p, d.d_ridge_m)
        reach = d.k1 + 8
        r |= {"window_capped": int((d.window_capped_p | d.window_capped_m).sum()),
              "reach_but_too_short": int(((reach >= d.d_ridge_p - 2) & ~d.window_capped_p).sum()
                                         + ((reach >= d.d_ridge_m - 2) & ~d.window_capped_m).sum()),
              "nearest_ridge_px": {"P10": float(np.nanpercentile(dr[np.isfinite(dr)], 10)),
                                   "P50": float(np.nanpercentile(dr[np.isfinite(dr)], 50)),
                                   "P90": float(np.nanpercentile(dr[np.isfinite(dr)], 90))}}
    return r


def per_image(run):
    rows = {}
    for b, root in BATCHES.items():
        for p in sorted((root / "stage_b" / run).glob("*.json")):
            j = json.loads(p.read_text())
            if "channels" not in j:
                continue
            f = j["channels"]["fold"]
            rows[p.stem] = {"n_accepted": f["n_accepted"], "n_gate": f["n_gate"], "fold_w<0.5": f["frac_w_lt_0.5"],
                            "w<0.5": j["combined"]["frac_w_lt_0.5"] if "frac_w_lt_0.5" in j["combined"] else None,
                            "fold_runtime_s": f["runtime_s"], "n_window_capped": f.get("n_window_capped")}
    return rows


def constraints(ref, new, aud):
    c, k = ref["combined"], new["combined"]
    out = {}
    for roi, v in k["texture_rois"].items():
        if "083" in roi and "interior" in roi:
            out[f"083 interior unchanged ({roi})"] = (abs(v - c["texture_rois"][roi]) < 5e-4, v - c["texture_rois"][roi])
        out[f"texture ROI {roi} <= +0.01"] = (v <= c["texture_rois"][roi] + 0.01, v - c["texture_rois"][roi])
    for key, name in (("texture_label", "texture-labelled"), ("tri_fp", "tri_fp")):
        dv = new["core"][aud][key]["share"] - ref["core"][aud][key]["share"]
        out[f"core {name} relaxed <= +0.02"] = (dv <= 0.02, dv)
    for w in ("combined", "core"):
        dv = new[w][aud]["geom_only geom_fp"]["share"] - ref[w][aud]["geom_only geom_fp"]["share"]
        out[f"geom_fp not increased ({w})"] = (dv <= 0, dv)
    for cl in ("step", "limb", "fold"):
        dv = k["coverage"][cl] - c["coverage"][cl]
        out[f"coverage {cl} >= -0.01"] = (dv >= -0.01, dv)
    return {n: {"ok": bool(ok), "diff": float(d)} for n, (ok, d) in out.items()}


def changes(ref, new):
    """Per image: accepted folds lost / gained, and the capped share of the candidates."""
    rows, tot = {}, {"n": 0, "capped": 0, "lost": 0, "gained": 0, "lost_capped": 0, "gained_capped": 0}
    for b, root in BATCHES.items():
        for p in sorted((root / "stage_b" / ref).glob("*_fold_candidates.parquet")):
            a, n = pd.read_parquet(p), pd.read_parquet(root / "stage_b" / new / p.name)
            assert len(a) == len(n) and np.array_equal(a.u0.to_numpy(), n.u0.to_numpy())
            cap = (n.window_capped_p | n.window_capped_m).to_numpy()
            A, N = a.accepted.to_numpy(), n.accepted.to_numpy()
            r = {"n": len(a), "capped": int(cap.sum()), "lost": int((A & ~N).sum()), "gained": int((~A & N).sum()),
                 "lost_capped": int((A & ~N & cap).sum()), "gained_capped": int((~A & N & cap).sum())}
            rows[p.name.replace("_fold_candidates.parquet", "")] = r
            for k in tot:
                tot[k] += r[k]
    tot["capped_share"] = tot["capped"] / tot["n"]
    return {"total": tot, "per_image": rows}


def roi(run, name, w="core"):
    return next(r for r in g1_table(run)[w]["rois"] if r["roi"] == name)


def headline(m, U):
    out = {}
    for w in ("combined", "core"):
        a = m[w]["audit"]
        out[w] = {"missed/px": a["missed_px"]["share"], "band excess": a["band_excess"]["share"],
                  "tri_fp": a["tri_fp"]["share"], "texture label": a["texture_label"]["share"],
                  "geom_fp": a["geom_only geom_fp"]["share"], "geom_gain": a["geom_only geom_gain"]["share"],
                  **{f"cov {c}": v for c, v in m[w]["coverage"].items()}, "texture class": m[w]["texture_class"],
                  **{f"tex {k}": v for k, v in m[w]["texture_rois"].items()},
                  **{f"rgb {k}": v for k, v in m[w]["rgb"].items()},
                  **{f"w<0.5 {b}": v for b, v in m[w]["wfrac"].items()}}
        for i in NARROW:
            out[w][f"unit {i}"] = float(U.loc[U.id == i, f"frac_{w}"].iloc[0])
    return out


def main():
    lab, draws, strata, totals, masks = load_labels()
    edges = rgb_edge_set()
    runs = {r: Run(r) for r in (REF, NEW)}
    b1, b2 = runs[REF].stems["batch1"], runs[REF].stems["batch2"]
    M = {r: metrics(runs[r], lab, draws, strata, totals, masks, edges, b1, b2) for r in runs}
    rep = {"headline": {r: headline(M[r], M[r]["_U"]) for r in runs},
           "constraints": constraints(M[REF], M[NEW], "audit"),
           "constraints_batch1_audit": constraints(M[REF], M[NEW], "audit_batch1"),
           "risers_070": {r: risers(r) for r in runs}, "laps_068": {r: laps068(r) for r in runs},
           "per_image": {r: per_image(r) for r in runs},
           "narrow_units": {r: M[r]["_U"][M[r]["_U"].id.isin(NARROW)][["id", "stem", "bucket", "tags", "frac_combined",
                                                                     "frac_core"]].to_dict(orient="records")
                            for r in runs}}
    mid = rep["risers_070"][NEW]
    riser_ok = all(mid[f"middle {f}"]["accepted_or_rescued"] >= 0.9 for f in ("upper", "lower"))
    rep["changes"] = changes(REF, NEW)
    rep["rock_ridge_088_core"] = {r: {k: roi(runs[r], "088 rock ridge")[k] for k in ("coverage", "share_w<0.5")}
                                  for r in runs}
    if W1B:
        hr, hn = rep["headline"][REF], rep["headline"][NEW]
        extra = {"core crease coverage >= -0.02": (hn["core"]["cov crease"] - hr["core"]["cov crease"], lambda d: d >= -0.02),
                 "088 rock ridge core coverage >= 0.70": (rep["rock_ridge_088_core"][NEW]["coverage"], lambda v: v >= 0.70),
                 "geom_gain (combined) >= -0.01": (hn["combined"]["geom_gain"] - hr["combined"]["geom_gain"],
                                                   lambda d: d >= -0.01),
                 "core missed/px <= +0.002": (hn["core"]["missed/px"] - hr["core"]["missed/px"], lambda d: d <= 0.002),
                 "core RGB fold candidates >= -0.005": (hn["core"]["rgb fold_candidates"] - hr["core"]["rgb fold_candidates"],
                                                        lambda d: d >= -0.005)}
        rep["constraints_w1b"] = {k: {"ok": bool(f(v)), "diff": float(v)} for k, (v, f) in extra.items()}
        # batch1 audit: geom_gain and core missed from the batch1-unit audit
        a1 = {r: M[r] for r in runs}
        e1 = {"geom_gain (combined) >= -0.01": a1[NEW]["combined"]["audit_batch1"]["geom_only geom_gain"]["share"]
              - a1[REF]["combined"]["audit_batch1"]["geom_only geom_gain"]["share"],
              "core missed/px <= +0.002": a1[NEW]["core"]["audit_batch1"]["missed_px"]["share"]
              - a1[REF]["core"]["audit_batch1"]["missed_px"]["share"]}
        rep["constraints_w1b_batch1_audit"] = {
            k: ({"ok": bool(e1[k] >= -0.01 if "gain" in k else e1[k] <= 0.002), "diff": float(e1[k])} if k in e1
                else v) for k, v in rep["constraints_w1b"].items()}
        rep["constraints"] |= rep["constraints_w1b"]
        rep["constraints_batch1_audit"] |= rep["constraints_w1b_batch1_audit"]
    all_ok = all(v["ok"] for v in rep["constraints"].values())
    all_ok1 = all(v["ok"] for v in rep["constraints_batch1_audit"].values())
    rep["verdict"] = {"constraints": all_ok, "constraints_batch1_audit": all_ok1, "middle_riser_>=0.9": riser_ok,
                      "propose": bool(all_ok and all_ok1 and riser_ok), "same_verdict_batch1": all_ok == all_ok1}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("window_cap.json" if not W1B else f"window_cap_{NEW}.json")).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("headline", "constraints", "constraints_batch1_audit", "risers_070",
                                         "laps_068", "verdict")}, indent=1, default=float))


if __name__ == "__main__":
    main()
