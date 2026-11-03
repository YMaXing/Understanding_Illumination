"""TASK_GEOM_V2 V1: one scorer for any list of Stage B runs (report only; reads RGB for the edge-coverage measure).

For each run and for two weights:
  combined  the run's deliverable (`w`, original resolution; `w_net` at network resolution);
  core      min(depth_w_net, fold_w_net) from the saved per-channel arrays (geometry without the Laplacian; runs
            without a depth or fold channel have no core).
Metrics (imported from the existing scripts, not duplicated):
  g1        the G1 ROI table, batch1 pinned ROIs (scripts/geom_eval.py roi_list / roi_metrics, pooled per class as
            geom_eval.evaluate): share of w < 0.5, median w, polyline coverage; texture ROIs listed one by one;
  audit     the labelled audit (audit_split/, scripts/geom_g4.py Estimator: stratified pps, stratified bootstrap
            B = 10000, a fresh seed-0 generator per run, weight and direction): per bucket the closed share (per
            unit: w < 0.5 on >= 50 % of the unit) and the relaxed share (per pixel: the unit's fraction with w < 0.5);
            missed true geometry (gap_fixable + gap_unresolved) per unit and per pixel; band excess, tri_fp and
            texture-labelled area still relaxed (per pixel);
  rgb       RGB-edge coverage (the G2(c) measure; scripts/geom_g2_coverage.py, geom_g2_align.py, rgb_edges.py) of a
            fixed edge set: the strong accepted depth edges of geom_v1 (both batches; identical to geom_v0's) and the
            batch1 fold targets; an edge is covered if the full-resolution weight is < 0.5 at its RGB edge;
  wfrac     share of w < 0.5 per batch, deliverable and network resolution (core: upsampled as the deliverable is);
  runtime   (--timing) Stage B process_one per image, median per batch (scripts/geom_g6_timing.py IOTimer protocol,
            no viz, into a temporary folder).
Writes <out>/score.json (default baseline_output/diag/geom/v2/) and prints the tables.
Run: env -u PYTHONPATH uv run python scripts/geom_score.py [--runs tri_veto4 geom_v1_r3 geom_v1] [--timing] [--no-rgb]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from depth_rois import POLYLINES  # noqa: E402
from fold_null_rois import TARGETS  # noqa: E402
from geom_eval import CLASSES, roi_list, roi_metrics  # noqa: E402
from geom_g2_align import strong  # noqa: E402
from geom_g2_coverage import covered, poly_normals, to_full  # noqa: E402
from geom_g4 import AUD, BUCKETS, GAPS, Estimator  # noqa: E402
from rgb_edges import luminance, rgb_offset  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.outputs import upsample_weights  # noqa: E402
from illum_baseline.stage_b import image_stems, load_upright, process_one  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
DEFAULT_RUNS = ("tri_veto4", "geom_v1_r3", "geom_v1")
EDGE_RUN = "geom_v1"     # the fixed RGB-edge set comes from its depth tables
OUT = ROOT / "baseline_output" / "diag" / "geom" / "v2"
W_MIN = 0.05
WEIGHTS = ("combined", "core")


# ---------------------------------------------------------------- loading

class Run:
    """Upright arrays of one run, cached per stem."""

    def __init__(self, name: str):
        self.name = name
        self.cfg = {b: load_config(r / "stage_b" / name / "config_resolved.yaml") for b, r in BATCHES.items()}
        self.stems = {b: image_stems(r / "stage_b" / name) for b, r in BATCHES.items()}
        self._cache: dict[str, dict] = {}

    def batch_of(self, stem: str) -> str:
        return next(b for b, s in self.stems.items() if stem in s)

    def maps(self, stem: str) -> dict:
        if stem not in self._cache:
            b = self.batch_of(stem)
            rd = BATCHES[b] / "stage_b" / self.name
            a, meta = load_upright(self.cfg[b], rd, stem)
            out = {"combined_net": a["w_net"].astype(np.float64), "combined_full": a["w"].astype(np.float64),
                   "up_wh": tuple(meta["geometry"]["upright_WxH"])}
            if "depth_w_net" in a and "fold_w_net" in a:
                core = np.minimum(a["depth_w_net"].astype(np.float64), a["fold_w_net"].astype(np.float64))
                out["core_net"] = core
                out["core_full"] = upsample_weights(core, out["up_wh"], W_MIN)
            self._cache = {stem: out}   # one image at a time (memory)
        return self._cache[stem]

    def has_core(self) -> bool:
        s = self.stems["batch1"][0]
        return "core_net" in self.maps(s)


# ---------------------------------------------------------------- metrics

def g1_table(run: Run) -> dict:
    rois = roi_list()
    out = {}
    for weight in WEIGHTS:
        if weight == "core" and not run.has_core():
            continue
        pooled, per_roi = {}, []
        for roi in rois:
            w = run.maps(f"taiwei_cui_{roi['stem']}")[f"{weight}_net"]
            m = roi_metrics(w, roi)
            per_roi.append({"roi": roi["name"], "class": roi["class"], "legacy": bool(roi.get("legacy", False)),
                            **{k: v for k, v in m.items() if not k.startswith("_")}})
            if roi.get("legacy"):
                continue
            p = pooled.setdefault(roi["class"], {"vals": [], "cov": []})
            p["vals"].append(m["_vals"])
            if m["_cov"] is not None:
                p["cov"].append(m["_cov"])
        classes = {}
        for cls in CLASSES:
            v = np.concatenate(pooled[cls]["vals"])
            c = pooled[cls]["cov"]
            classes[cls] = {"share_w<0.5": float((v < 0.5).mean()), "median_w": float(np.median(v)),
                            "coverage": float(np.concatenate(c).mean()) if c else None}
        out[weight] = {"classes": classes, "texture_rois": {r["roi"]: r["share_w<0.5"] for r in per_roi
                                                           if r["class"] == "texture" and not r["legacy"]},
                       "rois": per_roi}
    return out


def unit_fractions(run: Run, lab: pd.DataFrame, masks) -> pd.DataFrame:
    rows = []
    for stem, g in lab.groupby("stem"):
        mp = run.maps(stem)
        shape = mp["combined_net"].shape
        for r in g.itertuples():
            m = np.zeros(shape[0] * shape[1], bool)
            m[masks[f"id{r.id:03d}"]] = True
            row = {"id": r.id, "stem": stem, "direction": r.direction, "label": r.label, "bucket": r.bucket,
                   "tags": r.tags, "area_px": r.area_px}
            for weight in WEIGHTS:
                if f"{weight}_net" in mp:
                    row[f"frac_{weight}"] = float((mp[f"{weight}_net"].ravel()[m] < 0.5).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def audit(run: Run, U: pd.DataFrame, draws: pd.DataFrame, strata: pd.DataFrame, totals: pd.DataFrame) -> dict:
    out = {}
    for weight in WEIGHTS:
        col = f"frac_{weight}"
        if col not in U:
            continue
        res = {}
        for direction in ("tri-only", "geom-only"):
            d = draws[draws["direction"] == direction].merge(U, on=["id", "stem", "direction"], how="left")
            d = d.reset_index(drop=True)
            t = totals[totals["direction"] == direction]
            E = Estimator(d, strata[strata["direction"] == direction], float(t["area_px"].sum()),
                          float(t["image_px"].sum()), np.random.default_rng(0))
            f = d[col].to_numpy()
            closed = f >= 0.5
            ind = {}
            for b in BUCKETS[direction]:
                isb = (d["bucket"] == b).to_numpy()
                ind[f"{b} | share"] = isb
                ind[f"{b} | closed per unit"] = isb & closed
                ind[f"{b} | relaxed per pixel"] = isb * f
            if direction == "tri-only":
                gap = d["bucket"].isin(GAPS).to_numpy()
                ind["missed true geometry | per unit"] = gap & ~closed
                ind["missed true geometry | per pixel"] = gap * (1 - f)
                ind["gap_unresolved missed | per pixel"] = (d["bucket"] == "gap_unresolved").to_numpy() * (1 - f)
                ind["texture label | relaxed per pixel"] = (d["label"] == "texture").to_numpy() * f
            df, _ = E.run(ind)
            res[direction] = {r["quantity"]: {k: r[k] for k in ("share", "ci_lo", "ci_hi", "area_px")}
                              for r in df.to_dict(orient="records")}
        out[weight] = res
    return out


def rgb_edge_set() -> dict:
    """{stem: {'depth': full-res RGB-edge points of the strong depth edges, 'fold': [... per target polyline]}}."""
    cache = OUT / "rgb_edge_set.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["edges"].item()
    edges = {}
    for batch, root in BATCHES.items():
        rd = root / "stage_b" / EDGE_RUN
        cfg = load_config(rd / "config_resolved.yaml")
        for stem in image_stems(rd):
            a, meta = load_upright(cfg, rd, stem)
            net_wh, up_wh = (a["w_net"].shape[1], a["w_net"].shape[0]), tuple(meta["geometry"]["upright_WxH"])
            Y = luminance(root / "stage_a", stem)
            st = strong(pd.read_parquet(rd / f"{stem}_depth_candidates.parquet"))
            x0, n = st[["u0", "v0"]].to_numpy(), st[["n_u", "n_v"]].to_numpy()
            s = rgb_offset(Y, x0, n)
            ok = np.isfinite(s)
            e = {"depth": to_full(x0[ok] + s[ok, None] * n[ok], net_wh, up_wh), "fold": []}
            if batch == "batch1":
                polys = [poly for (st_, _, poly) in TARGETS.values() if stem.endswith(st_)]
                polys += [poly for (st_, cls, poly) in POLYLINES.values() if stem.endswith(st_) and cls == "fold"]
                for poly in polys:
                    pts, nn = poly_normals(poly)
                    o = rgb_offset(Y, pts, nn)
                    k = np.isfinite(o)
                    e["fold"].append(to_full(pts[k] + o[k, None] * nn[k], net_wh, up_wh))
            edges[stem] = e
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, edges=np.array(edges, dtype=object))
    return edges


def rgb_coverage(run: Run, edges: dict) -> dict:
    out = {}
    for weight in WEIGHTS:
        if weight == "core" and not run.has_core():
            continue
        dep, fold = [], []
        for stem, e in edges.items():
            wf = run.maps(stem)[f"{weight}_full"]
            dep.append(covered(wf, e["depth"]))
            fold += [covered(wf, p) for p in e["fold"]]
        d, f = np.concatenate(dep), np.concatenate(fold)
        out[weight] = {"depth_edges": float(d.mean()), "n_depth_edges": int(d.size), "fold_targets": float(f.mean()),
                       "n_fold_samples": int(f.size)}
    return out


def wfrac(run: Run) -> dict:
    out = {}
    for weight in WEIGHTS:
        if weight == "core" and not run.has_core():
            continue
        res = {}
        for batch, stems in run.stems.items():
            c = np.zeros(4)
            for stem in stems:
                mp = run.maps(stem)
                c += [(mp[f"{weight}_full"] < 0.5).sum(), mp[f"{weight}_full"].size,
                      (mp[f"{weight}_net"] < 0.5).sum(), mp[f"{weight}_net"].size]
            res[batch] = {"deliverable": c[0] / c[1], "network": c[2] / c[3]}
        out[weight] = res
    return out


def runtime(run: Run) -> dict:
    from geom_g6_timing import IOTimer

    out = {}
    for batch, stems in run.stems.items():
        t, io = [], []
        with tempfile.TemporaryDirectory() as td:
            for stem in stems:
                with IOTimer() as tm:
                    t0 = time.perf_counter()
                    process_one(run.cfg[batch], stem, Path(td))
                    t.append(time.perf_counter() - t0)
                io.append(tm.t)
        out[batch] = {"median_s": float(np.median(t)), "median_io_s": float(np.median(io))}
    return out


def load_labels():
    lab = pd.read_csv(AUD / "audit_labels.csv", dtype={"tags": str, "note": str}).fillna({"tags": "", "note": ""})
    return (lab, pd.read_csv(AUD / "audit_draws.csv"), pd.read_csv(AUD / "audit_strata.csv"),
            pd.DataFrame(json.loads((AUD / "audit_totals.json").read_text())["per_image"]),
            np.load(AUD / "audit_masks.npz"))


def score(runs, timing: bool = False, rgb: bool = True) -> dict:
    lab, draws, strata, totals, masks = load_labels()
    edges = rgb_edge_set() if rgb else None
    rep = {}
    for name in runs:
        run = Run(name)
        U = unit_fractions(run, lab, masks)
        r = {"g1": g1_table(run), "audit": audit(run, U, draws, strata, totals), "wfrac": wfrac(run),
             "units": U.to_dict(orient="records")}
        if rgb:
            r["rgb"] = rgb_coverage(run, edges)
        if timing:
            r["runtime"] = runtime(run)
        rep[name] = r
    return rep


def summary(rep: dict) -> pd.DataFrame:
    """One row per run x weight with the headline numbers (the task file's comparison table)."""
    rows = []
    for name, r in rep.items():
        for weight in WEIGHTS:
            if weight not in r["audit"]:
                continue
            a, g = r["audit"][weight]["tri-only"], r["g1"][weight]
            row = {"run": name, "w": weight,
                   "missed/px": a["missed true geometry | per pixel"]["share"],
                   "missed/unit": a["missed true geometry | per unit"]["share"],
                   "unresolved missed/px": a["gap_unresolved missed | per pixel"]["share"],
                   "band excess relaxed": a["tri_band_excess | relaxed per pixel"]["share"],
                   "tri_fp relaxed": a["tri_fp | relaxed per pixel"]["share"],
                   "texture label relaxed": a["texture label | relaxed per pixel"]["share"],
                   **{f"cov {c}": g["classes"][c]["coverage"] for c in ("step", "limb", "fold", "crease")},
                   "texture class": g["classes"]["texture"]["share_w<0.5"],
                   **{f"tex {k}": v for k, v in g["texture_rois"].items()},
                   **{f"w<0.5 {b}": v["deliverable"] for b, v in r["wfrac"][weight].items()}}
            if "rgb" in r:
                row |= {"rgb depth edges": r["rgb"][weight]["depth_edges"], "rgb fold targets": r["rgb"][weight][
                    "fold_targets"]}
            if "runtime" in r:
                row |= {f"runtime {b}": v["median_s"] for b, v in r["runtime"].items()}
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--no-rgb", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = score(args.runs, timing=args.timing, rgb=not args.no_rgb)
    (out / "score.json").write_text(json.dumps(rep, indent=1, default=float))
    S = summary(rep)
    S.to_csv(out / "score_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print(S.set_index(["run", "w"]).T.round(4).to_string())
    print(f"\n-> {out / 'score.json'}")


if __name__ == "__main__":
    main()
