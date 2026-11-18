#!/usr/bin/env python3
"""Checkpoint 2: the three §7 baselines on the real annotations, plain tau vs the widened (Delta Q1) tolerances.

    metrics/run.sh tools/ck2_report.py [--out metrics/runs/ck2]

Writes per_pair_<baseline>.csv (one row per pair, annotator column included) and ck2.md. Groups: headline sets
(dev_v2 + heldout, C5) and each glossy set separately. Read-only apart from --out.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import baselines as BL  # noqa: E402
from core import config as K  # noqa: E402
from core import io as IO  # noqa: E402
from core import pair_metrics as PM  # noqa: E402
from core import summary as SM  # noqa: E402
import migrate_v0 as MG  # noqa: E402

COLS = ["id", "image", "set", "annotator", "type", "kind", "confidence", "length_px", "ar", "br", "nI", "nR", "nL",
        "alpha", "H_I", "H_R", "H_L", "E", "H", "se_noise", "se_tex", "textured", "invalid", "geom_cross",
        "detector_split", "length_out_of_range", "low_snr", "input_excursion", "alpha_defined", "tolE_wide",
        "tolH_wide", "pass_plain", "pass_wide", "bin_plain", "bin_wide", "both_step", "exclusions", "dI", "dR", "dL"]


def group_of(s, cfg):
    return K.group_of(cfg, s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(K.METRICS / "runs" / "ck2"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = K.load_config()
    tau = cfg["thresholds"]["tau"]
    files = MG.read_files(cfg)
    R = {b: [] for b in cfg["baselines"]["names"]}
    for stem in K.eval_stems(cfg):
        f = files[stem]
        img = IO.load_image(cfg, stem)
        for b in R:
            ctx = IO.context(cfg, stem, BL.predict(b, img["lI"], cfg), img, f["set"])
            for rec, J, _ in PM.score_pairs(f["pairs"], ctx, cfg):
                R[b].append((rec, J, None))
        print(f"{f['set']:15s} {stem}", flush=True)
    for b, res in R.items():
        with open(out / f"per_pair_{b}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            for rec, J, _ in res:
                row = {**rec, **J[tau]}
                row["exclusions"] = ";".join(row["exclusions"])
                for k in ("dI", "dR", "dL"):
                    row[k] = " ".join(f"{v:.5f}" for v in row[k])
                w.writerow(row)
    groups = sorted({group_of(rec["set"], cfg) for rec, _, _ in R["all_R"]}, key=lambda g: (not g.startswith("head"), g))
    L = [f"# Checkpoint 2: baselines on the real annotations (tau {tau})", "",
         "Error rate = failing / headline-eligible pairs (eligible = no exclusion: low_snr [noise model], invalid, "
         "unsure, b, targeted s, d_L, unscored d). plain = tau for E and H; wide = Delta Q1 (R-flat kinds: E <= "
         "max(tau, 2 se_tex), H_R <= max(tau, H_I); m plain).", ""]
    for g in groups:
        L += [f"## {g}", ""]
        sub = {b: [x for x in res if group_of(x[0]["set"], cfg) == g] for b, res in R.items()}
        any_ = sub["all_R"]
        cnt = Counter(rec["kind"] for rec, _, _ in any_)
        ex = Counter(e for rec, J, _ in any_ for e in J[tau]["exclusions"])
        fl = {k: sum(1 for rec, _, _ in any_ if rec[k]) for k in ("geom_cross", "textured", "length_out_of_range", "invalid")}
        L.append("pairs by kind: " + ", ".join(f"{k} {cnt[k]}" for k in SM.KINDS) + "  ")
        L.append("exclusion reasons (a pair can have several): " + ", ".join(f"{k} {v}" for k, v in sorted(ex.items())) + "  ")
        L.append("flags: " + ", ".join(f"{k} {v}" for k, v in fl.items()) +
                 f", input_excursion (H_I > tau) {sum(1 for _, J, _ in any_ if J[tau]['input_excursion'])}  ")
        L += ["", "| baseline | kind | eligible | error plain | error wide |", "|---|---|---|---|---|"]
        for b, res in sub.items():
            sp, sw = SM.by_kind(res, tau, "plain"), SM.by_kind(res, tau, "wide")
            for k in ("i", "m", "s_random", "d_R"):
                L.append(f"| {b} | {k} | {sp[k]['n']} | {sp[k]['text']} | {sw[k]['text']} |")
            split = [sp[k]["rate"] for k in ("i", "m")], [sw[k]["rate"] for k in ("i", "m")]
            fmt = lambda v: "n/a" if None in v else f"{np.mean(v):.3f}"
            L.append(f"| {b} | **Split** = mean(ER_i, ER_m) | | {fmt(split[0])} | {fmt(split[1])} |")
        L += ["", "Split bins of failing eligible pairs (wide; plain in brackets):", "",
              "| baseline | kind | reversed | full_leak | partial | overshoot | n/a |", "|---|---|---|---|---|---|---|"]
        for b, res in sub.items():
            for k in ("i", "m"):
                el = [J[tau] for rec, J, _ in res if rec["kind"] == k and not J[tau]["exclusions"]]
                cw = Counter(j["bin_wide"] for j in el if j["bin_wide"])
                cp = Counter(j["bin_plain"] for j in el if j["bin_plain"])
                L.append(f"| {b} | {k} | " + " | ".join(f"{cw[x]} ({cp[x]})" for x in PM.BINS) + " |")
        L += ["", "b pairs (diagnostic): both-step rate (||ΔℓR|| > tau and ||ΔℓL|| > tau), alpha quartiles:", ""]
        for b, res in sub.items():
            bb = [(rec, J[tau]) for rec, J, _ in res if rec["kind"] == "b" and not rec["invalid"]]
            q = np.nanpercentile([r["alpha"] for r, _ in bb], [25, 50, 75]) if bb else []
            L.append(f"- {b}: {SM.rate(not j['both_step'] for _, j in bb)['n']} pairs, both-step "
                     f"{sum(j['both_step'] for _, j in bb)}; alpha Q1/Q2/Q3 {np.round(q, 3).tolist()}")
        # blur-L specifics (§7): H_R > tau on most i pairs; E_i small where patches lie beyond 2 sigma of the edge
        bl = [(rec, J[tau]) for rec, J, _ in sub["blur_L"] if rec["kind"] == "i" and not J[tau]["exclusions"]]
        if bl:
            s2 = 2 * cfg["baselines"]["blur_sigma_px"]
            far = [(r, j) for r, j in bl if r["length_px"] / 2 - max(r["ar"], r["br"]) >= s2]
            near = [(r, j) for r, j in bl if (r, j) not in far]
            med = lambda xs: f"{np.median(xs):.3f}" if xs else "n/a"
            L += ["", f"blur_L, eligible i pairs: H_R > tau on {sum(r['H_R'] > tau for r, _ in bl)}/{len(bl)}; "
                  f"H_R > tolH_wide on {sum(not r['H_R'] <= j['tolH_wide'] for r, j in bl)}/{len(bl)}.  ",
                  f"E_i median: patches >= {s2:g} px from the segment midpoint (proxy for 'beyond 2 sigma from the "
                  f"edge'; {len(far)} pairs) {med([r['E'] for r, _ in far])}, nearer ({len(near)} pairs) "
                  f"{med([r['E'] for r, _ in near])}.", ""]
        # all-R: i pairs that pass
        ar = [(rec, J[tau]) for rec, J, _ in sub["all_R"] if rec["kind"] == "i" and not J[tau]["exclusions"]]
        ok = [r for r, j in ar if j["pass_wide"]]
        okp = [r for r, j in ar if j["pass_plain"]]
        L += [f"all_R, eligible i pairs passing: wide {len(ok)}, plain {len(okp)}; their ||ΔℓI|| max: wide "
              f"{max([r['nI'] for r in ok], default=float('nan')):.3f}, plain {max([r['nI'] for r in okp], default=float('nan')):.3f}.", ""]
    (out / "ck2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(dict(config=cfg["_path"], tau=tau, n_pairs={b: len(r) for b, r in R.items()},
                                                   annotation_sha256=json.loads((K.repo_path(cfg["paths"]["migration"]) / "manifest.json").read_text())["annotation_files"]), indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
