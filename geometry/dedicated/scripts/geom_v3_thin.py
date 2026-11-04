"""TASK_GEOM_V2 V3: short thin depth components (report only; reads RGB only for the overlays).

Variants of geometry.depth.short_thin: off, strong, linked with gap 4 and 6 px. The depth channel is rerun from the
Stage A cache for each variant (geom_v1's settings otherwise; `off` reproduces the saved depth_w_net, checked); the
fold and Laplacian weights are geom_v1's saved arrays. core w = min(depth, fold), combined w = min(depth, fold,
laplacian) (full resolution: the deliverable's upsampling). Scored with scripts/geom_score.py's functions:
  - the 25 component-stage step units of G4 (baseline_output/diag/geom/g4/units.csv) and all step gaps (label step,
    gap_fixable | gap_unresolved), core w: closed units (>= 50 %), closed px, relaxed px (per pixel);
  - per-pixel relaxed shares (estimator) of tri_fp, geom_fp and texture-labelled units, core w;
  - texture ROI shares, core and combined; w < 0.5 per batch;
  - overlays of the newly accepted pieces on 080, 075, 082, 067 (foliage) and 090, 073 (fences).
Rule (propose, don't switch): the variant that closes the most step-gap px (core w, units closed at >= 50 %), provided that
every texture ROI share (combined) is <= geom_v1 + 0.01 and the texture-labelled relaxed share (core) rises by <= 0.02.
Writes baseline_output/diag/geom/v2/thin.json and thin_overlays/<stem>.png.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geom_g4 import GAPS  # noqa: E402
from geom_score import OUT, W_MIN, Run, audit, g1_table, load_labels, unit_fractions, wfrac  # noqa: E402
from illum_baseline.config import dense_fit_params, detector_params  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.depth import DepthChannel, DepthParams  # noqa: E402
from illum_baseline.outputs import upsample_weights  # noqa: E402
from illum_baseline.stage_b import load_upright  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
V1 = "geom_v1"
VARIANTS = {"off": {"short_thin": "off"}, "strong": {"short_thin": "strong"},
            "linked4": {"short_thin": "linked", "short_thin_gap_px": 4.0},
            "linked6": {"short_thin": "linked", "short_thin_gap_px": 6.0}}
OVERLAYS = ("taiwei_cui_080", "taiwei_cui_075", "taiwei_cui_082", "acharya_mrudula_067", "taiwei_cui_090",
            "acharya_mrudula_073")


class VirtualRun(Run):
    """geom_v1 with the depth channel's w replaced: maps from in-memory arrays."""

    def __init__(self, name: str, depth: dict, base: Run):
        self.name, self.cfg, self.stems = name, base.cfg, base.stems
        self.depth, self.base = depth, base
        self._cache = {}

    def maps(self, stem: str) -> dict:
        if stem not in self._cache:
            b = self.base.batch_of(stem)
            a, meta = load_upright(self.cfg[b], BATCHES[b] / "stage_b" / V1, stem)
            up = tuple(meta["geometry"]["upright_WxH"])
            wd = self.depth[stem]
            core = np.minimum(wd, a["fold_w_net"].astype(np.float64))
            comb = np.minimum(core, a["laplacian_w_net"].astype(np.float64))
            self._cache = {stem: {"combined_net": comb, "combined_full": upsample_weights(comb, up, W_MIN),
                                  "core_net": core, "core_full": upsample_weights(core, up, W_MIN), "up_wh": up}}
        return self._cache[stem]


def overlay(stem: str, root: Path, base_tab: pd.DataFrame, new: dict, out: Path):
    sa = json.loads((root / "stage_a" / f"{stem}.json").read_text())
    W, H = sa["network"]["W_n"], sa["network"]["H_n"]
    rgb = cv2.resize(cv2.imread(str(root / "stage_a" / "srgb" / f"{stem}.png")), (W, H), interpolation=cv2.INTER_AREA)
    panels = []
    for name, tab in new.items():
        img = (rgb * 0.6).astype(np.uint8)
        acc = base_tab[base_tab["accepted"]]
        img[acc["v"].to_numpy(), acc["u"].to_numpy()] = (200, 200, 200)
        nw = tab[tab["short_thin"]]
        for u_, v_ in zip(nw["u"].to_numpy(), nw["v"].to_numpy()):
            cv2.circle(img, (int(u_), int(v_)), 1, (0, 0, 255), -1)
        big = cv2.resize(img, (2 * W, 2 * H), interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, f"{stem} {name}: +{len(nw)} new (red), accepted geom_v1 gray", (8, 22), 0, 0.6, (255, 255, 255), 2)
        panels.append(big)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{stem}.png"), np.concatenate(panels, 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = Run(V1)
    depth = {k: {} for k in VARIANTS}
    tabs = {k: {} for k in VARIANTS}
    counts = {k: {"batch1": 0, "batch2": 0} for k in VARIANTS}
    check = []
    for batch, root in BATCHES.items():
        cfg = base.cfg[batch]
        dp0 = DepthParams.from_dict(dataclasses.asdict(cfg.geometry.depth))
        for stem in base.stems[batch]:
            meta_a = json.loads((root / "stage_a" / f"{stem}.json").read_text())
            z = np.load(root / "stage_a" / f"{stem}.npz")
            ctx = GeometryContext(z["D"], z["sky_prob"].astype(np.float64), Intrinsics.from_stage_a(meta_a),
                                  detector_params(cfg), dense_fit_params(cfg))
            for name, kw in VARIANTS.items():
                res = DepthChannel().run(ctx, dataclasses.replace(dp0, **kw))
                depth[name][stem] = res.w_net
                if name == "off":
                    saved = load_upright(cfg, root / "stage_b" / V1, stem)[0]["depth_w_net"]
                    check.append(bool(np.array_equal(res.w_net.astype(np.float16), saved)))
                else:
                    counts[name][batch] += int(res.candidates["short_thin"].sum())
                    if stem in OVERLAYS:
                        tabs[name][stem] = res.candidates
            if stem in OVERLAYS:
                overlay(stem, root, pd.read_parquet(root / "stage_b" / V1 / f"{stem}_depth_candidates.parquet"),
                        {n: tabs[n][stem] for n in VARIANTS if n != "off"}, OUT / "thin_overlays")
    lab, draws, strata, totals, masks = load_labels()
    g4 = pd.read_csv(ROOT / "baseline_output" / "diag" / "geom" / "g4" / "units.csv")
    comp_ids = set(g4.loc[g4["stage"] == "component", "id"])
    rep = {"off_reproduces_saved": all(check), "n_new_accepted": counts, "variants": {}}
    for name in VARIANTS:
        vr = VirtualRun(name, depth[name], base)
        U = unit_fractions(vr, lab, masks)
        A = audit(vr, U, draws, strata, totals)
        G = g1_table(vr)
        steps = U[(U["direction"] == "tri-only") & (U["label"] == "step") & U["bucket"].isin(GAPS)]
        cs = steps[steps["id"].isin(comp_ids)]
        r = {}
        for key, g in (("component_stage_units", cs), ("step_gaps", steps)):
            f = g["frac_core"]
            r[key] = {"n": int(len(g)), "closed": int((f >= 0.5).sum()), "closed_px": int(g.loc[f >= 0.5, "area_px"].sum()),
                      "relaxed_px": float((g["area_px"] * f).sum()), "area_px": int(g["area_px"].sum())}
        a_t, a_g = A["core"]["tri-only"], A["core"]["geom-only"]
        r["core_relaxed"] = {"tri_fp": a_t["tri_fp | relaxed per pixel"], "geom_fp": a_g["geom_fp | relaxed per pixel"],
                             "texture label": a_t["texture label | relaxed per pixel"],
                             "missed true geometry per pixel": a_t["missed true geometry | per pixel"],
                             "missed true geometry per unit": a_t["missed true geometry | per unit"]}
        r["texture_rois"] = {w: G[w]["texture_rois"] for w in G}
        r["roi_coverage"] = {w: {c: v["coverage"] for c, v in G[w]["classes"].items() if v["coverage"] is not None}
                             for w in G}
        r["wfrac"] = wfrac(vr)
        r["closed_component_ids"] = sorted(int(i) for i in cs.loc[cs["frac_core"] >= 0.5, "id"])
        rep["variants"][name] = r
    # rule
    off = rep["variants"]["off"]
    ok = {}
    for name, r in rep["variants"].items():
        tex_ok = all(r["texture_rois"]["combined"][k] <= off["texture_rois"]["combined"][k] + 0.01
                     for k in off["texture_rois"]["combined"])
        lab_ok = r["core_relaxed"]["texture label"]["share"] <= off["core_relaxed"]["texture label"]["share"] + 0.02
        ok[name] = {"texture_rois_ok": tex_ok, "texture_label_ok": lab_ok, "passes": tex_ok and lab_ok,
                    "closed_px": r["step_gaps"]["closed_px"], "relaxed_px": r["step_gaps"]["relaxed_px"]}
    cand = [n for n in VARIANTS if n != "off" and ok[n]["passes"]]
    best = max(cand, key=lambda n: ok[n]["closed_px"]) if cand else None
    closes = best is not None and ok[best]["closed_px"] > ok["off"]["closed_px"]
    by_relaxed = max(cand, key=lambda n: ok[n]["relaxed_px"]) if cand else None
    rep["rule"] = {"per_variant": ok, "proposed": best if closes else None,
                   "note": "the rule ranks by closed step-gap px; relaxed px is reported, not a rule term",
                   "most_relaxed_px": by_relaxed}
    (OUT / "thin.json").write_text(json.dumps(rep, indent=1, default=float))
    print("off reproduces the saved depth w:", rep["off_reproduces_saved"], "| new accepted:", counts)
    rows = []
    for name, r in rep["variants"].items():
        rows.append({"variant": name,
                     "comp-stage closed": f"{r['component_stage_units']['closed']}/{r['component_stage_units']['n']}",
                     "comp-stage relaxed px": round(r["component_stage_units"]["relaxed_px"]),
                     "step gaps closed": f"{r['step_gaps']['closed']}/{r['step_gaps']['n']}",
                     "step closed px": r["step_gaps"]["closed_px"], "step relaxed px": round(r["step_gaps"]["relaxed_px"]),
                     "tri_fp": round(r["core_relaxed"]["tri_fp"]["share"], 4),
                     "geom_fp": round(r["core_relaxed"]["geom_fp"]["share"], 4),
                     "texture label": round(r["core_relaxed"]["texture label"]["share"], 4),
                     "missed/px core": round(r["core_relaxed"]["missed true geometry per pixel"]["share"], 4),
                     **{f"tex {w[:4]} {k}": round(v, 4) for w in ("core", "combined")
                        for k, v in r["texture_rois"][w].items()},
                     **{f"w<.5 {b}": round(v["deliverable"], 4) for b, v in r["wfrac"]["combined"].items()}})
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(pd.DataFrame(rows).set_index("variant").T.to_string())
    print("\nrule:", json.dumps(rep["rule"], indent=1, default=float))
    print({n: r["closed_component_ids"] for n, r in rep["variants"].items()})


if __name__ == "__main__":
    main()
