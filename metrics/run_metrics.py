#!/usr/bin/env python3
"""Run the v1 metrics (DESIGN.md + deltas) on a prediction folder and/or the three baselines.

    metrics/run.sh run_metrics.py --out runs/NAME [--pred DIR [--name model]] [--images LIST] [--config FILE]

  --pred DIR     <DIR>/<stem>.npz with log_R float32 (H, W, 3) (optional log_L); omit to run the baselines only
  --images LIST  a file of stems (one per line, # comments), a comma-separated list, or set names (dev_v2, heldout,
                 glossy_dev, glossy_confirm); default: every eval image in config.yaml
  --out          relative paths resolve against metrics/ (runs/ is git-ignored)
The three §7 baselines (all_R, all_L, blur_L) always run alongside. Outputs in --out:
  per_pair.csv   one row per (method, pair): measurements, flags, judge at the config tau, pass/excluded per sweep tau
  profiles.npz   p_I, p_R, p_L per (method, pair): concatenated arrays t, I, R, L + index (method, image, id, start, stop)
  per_image.csv  per (method, image): eligible counts and ER per kind (tau 0.10), dense diagnostics D1-D4, A1, A2
  summary.json   everything below + config, git state, detector/map versions, annotation and prediction hashes
  summary.md     headline numbers (widened rule; plain alongside) with image-bootstrap CIs, tau sweep, exclusions,
                 strata, bins, b diagnostics, dense diagnostics, textured stratum + synthetic error floor, lit_end list
Report blocks (Checkpoint 5): one per data split (the per-image field split: train / val / test), never pooled; test
only with --test (test images are not even loaded otherwise). Pairs with used_in_loss = true are scored in a separate
"supervised fit" block per split. Each run reports the train - val gap per headline number. Provenance (the old set
and group names), the annotator group and glossy-pool membership are strata.
"""

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from core import aggregate as AG
from core import baselines as BL
from core import config as K
from core import dense as DN
from core import io as IO
from core import pair_metrics as PM
from core import summary as SM

VERSION = "1.0"
REC_COLS = ["id", "image", "data_split", "used_in_loss", "group", "set", "annotator_group", "glossy_pool", "reflective_advisory",
            "annotator", "annotator_id", "type", "kind", "confidence", "pool", "lighting", "material",
            "illumination", "lit_end", "ax", "ay", "ar", "bx", "by", "br", "length_px", "nI", "nR", "nL", "alpha",
            "H_I", "H_R", "H_L", "E_kind", "E", "H_kind", "H", "leaked_frac", "se_noise", "se_tex", "textured",
            "sigma_t", "invalid", "geom_cross", "detector_split", "length_out_of_range", "fracA", "fracB"]
J_COLS = ["low_snr", "low_contrast", "input_excursion", "alpha_defined", "tolE_plain", "tolH_plain", "tolE_wide",
          "tolH_wide", "pass_plain", "pass_wide", "bin_plain", "bin_wide", "both_step", "headline", "exclusions"]


def resolve_images(cfg, spec):
    allev = K.eval_stems(cfg)
    if not spec:
        return allev
    p = Path(spec)
    if p.exists():
        items = [ln.split("#")[0].strip() for ln in p.read_text().splitlines()]
    else:
        items = [x.strip() for x in spec.split(",")]
    out = []
    for x in filter(None, items):
        out += cfg["image_sets"][x] if x in cfg["image_sets"] else [x]
    bad = [s for s in out if s not in allev]
    if bad:
        raise SystemExit(f"not eval images in config.yaml: {bad}")
    return list(dict.fromkeys(out))


def git(*a):
    r = subprocess.run(["git", "-C", str(K.REPO), *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def git_state():
    """Read-only: HEAD, metrics-v1, and whether metrics/ in the working tree equals the metrics-v1 tree."""
    st = dict(head=git("rev-parse", "HEAD"), head_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
              metrics_v1=git("rev-parse", "--verify", "-q", "refs/heads/metrics-v1"))
    if st["metrics_v1"]:
        tree = {}
        for ln in (git("ls-tree", "-r", "metrics-v1", "--", "metrics") or "").splitlines():
            meta, path = ln.split("\t", 1)
            tree[path] = meta.split()[2]
        work = set((git("ls-files", "--others", "--exclude-standard", "--", "metrics") or "").splitlines())
        work |= set((git("ls-files", "--", "metrics") or "").splitlines())
        changed = sorted(p for p in work if p not in tree or git("hash-object", p) != tree[p])
        removed = sorted(p for p in tree if p not in work)
        st.update(metrics_matches_metrics_v1=not changed and not removed, metrics_changed=changed[:50],
                  metrics_removed=removed[:50])
    return st


def map_meta(cfg):
    """Detector / map provenance: the arm, and each map dir's meta / run file (SHA-256 + key fields)."""
    out = dict(detector_arm=cfg["maps"]["detector_arm"], geom_v1_1=cfg["maps"]["geom_v1_1"]["dirs"], dirs={})
    for name in ("smoothness_weight", "illum_edges", "material_edges"):
        for s, e in cfg["maps"][name].items():
            d = K.repo_path(e["dir"])
            for cand in (d / "meta.json", d.parent / "meta.json", d / "run.json", d.parent / "run.json"):
                if cand.exists():
                    m = json.loads(cand.read_text())
                    out["dirs"][f"{name}/{s}"] = dict(dir=e["dir"], key=e["key"], meta=str(cand.relative_to(K.REPO)),
                                                      meta_sha256=K.sha256(cand),
                                                      **{k: m.get(k) for k in ("default_arm", "default_status",
                                                                               "exporter_sha256", "arm_resolved", "at")
                                                         if k in m})
                    break
            else:
                out["dirs"][f"{name}/{s}"] = dict(dir=e["dir"], key=e["key"], meta=None)
    return out


def lit_end_list(files):
    """Q2: migrated i pairs whose inferred lit end is B (the guides say A lit); marginal = |b_B - b_A| < 0.5."""
    out = []
    for stem, f in sorted(files.items()):
        for p in f["pairs"]:
            le = p.get("legacy", {}).get("lit_end")
            if p["type"] == "i" and le and not le["agrees_guide_A_lit"]:
                d = le["b_B"] - le["b_A"]
                out.append(dict(set=f["set"], image=stem, id=p["id"], b_A=le["b_A"], b_B=le["b_B"], diff=round(d, 3),
                                marginal=abs(d) < 0.5))
    return out


def floor_lines(cfg):
    X = cfg["texture"]
    p = K.repo_path(X["floor_reference"])
    if not p.exists():
        return ["n/a: reference file missing"]
    F = json.loads(p.read_text())["floor"]
    out = []
    for g, lab in (("0.1", "moderate texture (grey SD 0.10)"), ("0.2", "strong texture (grey SD 0.20)")):
        for kind in ("i", "m"):
            for tex in ("textured", "not_textured"):
                v = F.get(g, {}).get(f"{X['floor_option']}|{kind}|{tex}")
                if v and v["n"]:
                    out.append(f"{lab} {kind} {tex.replace('_', ' ')} {100 * v['fail_wide'] / v['n']:.1f}% (n {int(v['n'])}, "
                               f"mean se_tex {v['se_tex_sum'] / v['n']:.3f})")
    return out + ["smooth materials 0% (synthetic suite)"]


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"


def ci_txt(ci, k):
    c = ci["ci"].get(k)
    if not c or c["lo"] is None:
        return "n/a"
    und = f", {c['n_undefined']} undefined" if c["n_undefined"] else ""
    return f"[{c['lo']:.3f}, {c['hi']:.3f}]{und}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred")
    ap.add_argument("--name", default="model")
    ap.add_argument("--images")
    ap.add_argument("--config")
    ap.add_argument("--test", action="store_true", help="also score and report the test split (never pooled)")
    a = ap.parse_args()
    cfg = K.load_config(a.config)
    out = Path(a.out)
    out = out if out.is_absolute() else K.METRICS / out
    out.mkdir(parents=True, exist_ok=True)
    T = cfg["thresholds"]
    tau0, taus = T["tau"], list(dict.fromkeys(T["tau_sweep"] + [T["tau"]]))
    stems = resolve_images(cfg, a.images)
    splits_on = list(cfg["aggregate"]["report_splits"]) + (["test"] if a.test else [])
    skipped_test = [s for s in stems if K.split_of(cfg, s) not in splits_on]
    stems = [s for s in stems if s not in skipped_test]
    glossy = K.glossy_pool(cfg)
    from core import annotators as AN
    name_of = AN.resolve(K.repo_path(cfg["paths"]["annotations"]) / "annotators.json")    # id -> current name
    methods = ([a.name] if a.pred else []) + list(cfg["baselines"]["names"])
    if a.pred and a.name in BL.BASELINES:
        raise SystemExit(f"--name {a.name} clashes with a baseline name")
    ann_dir = K.repo_path(cfg["paths"]["annotations"])
    files, ann_sha, pred_sha, geom_sha, map_src = {}, {}, {}, {}, {}
    rows = {m: [] for m in methods}
    dense_rows = []
    prof = dict(t=[], I=[], R=[], L=[], index=[])
    pos = 0
    t_start = dt.datetime.now(dt.timezone.utc)
    for stem in stems:
        p = ann_dir / f"{stem}.json"
        f = json.loads(p.read_text(encoding="utf-8"))
        files[stem], ann_sha[stem] = f, K.sha256(p)
        img = IO.load_image(cfg, stem)
        set_name = f["set"]
        maps = None
        for mi, m in enumerate(methods):
            if m in BL.BASELINES:
                lR = BL.predict(m, img["lI"], cfg)
                lL = IO.derive_log_L(img["lI"], lR)
            else:
                lR, lL = IO.load_prediction(cfg, a.pred, stem, img)
                pred_sha[stem] = K.sha256(Path(a.pred) / f"{stem}.npz")
            ctx = IO.context(cfg, stem, lR, img, set_name)
            if maps is None:
                maps = DN.ImageMaps(cfg, stem, set_name, img)
                map_src[stem] = maps.sources
                gd = cfg["maps"]["geom_v1_1"]["dirs"].get(set_name)
                geom_sha[stem] = K.sha256(K.repo_path(gd) / f"{stem}.npz") if gd else None
            for rec, J, pr in PM.score_pairs(f["pairs"], ctx, cfg, taus):
                rec.update(group=K.group_of(cfg, set_name), annotator_group=K.annotator_group(cfg, set_name),
                           glossy_pool=stem in glossy, data_split=f["split"], reflective_advisory=bool(f.get("reflective_advisory")),
                           annotator_id=rec["annotator"], annotator=name_of(rec["annotator"]))
                rows[m].append(dict(rec=rec, J=J))
                n = len(pr["t"])
                for k in ("t", "I", "R", "L"):
                    prof[k].append(pr[k].astype(np.float32))
                prof["index"].append((mi, stem, rec["id"], pos, pos + n))
                pos += n
            dense_rows.append(dict(method=m, image=stem, set=set_name, **DN.diagnostics(cfg, img, maps, lR, lL)))
        print(f"{set_name:15s} {stem:22s} {len(f['pairs']):4d} pairs", flush=True)

    # ---- per_pair.csv
    with open(out / "per_pair.csv", "w", newline="", encoding="utf-8") as fh:
        sweep_cols = [f"{c}@{t:g}" for t in taus for c in ("pass", "excluded")]
        w = csv.writer(fh)
        w.writerow(["method"] + REC_COLS + J_COLS + ["dI", "dR", "dL", "PA_I", "PB_I", "PA_R", "PB_R"] + sweep_cols)
        for m, rr in rows.items():
            for r in rr:
                rec, j = r["rec"], r["J"][tau0]
                w.writerow([m] + [rec.get(c) for c in REC_COLS] +
                           [";".join(j[c]) if c == "exclusions" else j.get(c) for c in J_COLS] +
                           [" ".join(f"{v:.6f}" for v in rec[k]) for k in ("dI", "dR", "dL", "PA_I", "PB_I", "PA_R", "PB_R")] +
                           [x for t in taus for x in (r["J"][t]["pass"], ";".join(r["J"][t]["exclusions"]))])
    # ---- profiles.npz
    idx = np.array(prof["index"], dtype=[("method", "i4"), ("image", "U64"), ("id", "U32"), ("start", "i8"), ("stop", "i8")])
    np.savez_compressed(out / "profiles.npz", methods=np.array(methods), index=idx,
                        t=np.concatenate(prof["t"]) if prof["t"] else np.zeros(0, np.float32),
                        **{k: (np.concatenate(prof[k]) if prof[k] else np.zeros((0, 3), np.float32)) for k in ("I", "R", "L")})
    # ---- report blocks: one per data split (unused pairs), plus "supervised fit" blocks (used_in_loss pairs)
    groups, sel = {}, {}
    for sp in splits_on:
        imgs = [x for x in stems if files[x]["split"] == sp]
        if not imgs:
            continue
        groups[sp], sel[sp] = imgs, (sp, False)
        if any(p.get("used_in_loss") for x in imgs for p in files[x]["pairs"]):
            groups[f"supervised fit · {sp}"], sel[f"supervised fit · {sp}"] = imgs, (sp, True)
    summ = {g: {m: AG.summarize([r for r in rows[m] if r["rec"]["data_split"] == sel[g][0] and r["rec"]["used_in_loss"] == sel[g][1]],
                                taus, tau0, imgs, cfg) for m in methods} for g, imgs in groups.items()}
    gaps = {}
    if "train" in summ and "val" in summ:
        for m in methods:
            tr, va = summ["train"][m]["headline"][tau0]["wide"], summ["val"][m]["headline"][tau0]["wide"]
            gaps[m] = {k: (None if tr[k] is None or va[k] is None else va[k] - tr[k]) for k in ("split", "leak", "long_range", "composite")}
    X = cfg["texture"]
    fref = K.repo_path(X["floor_reference"])
    floors = {}
    if fref.exists():
        ref = json.loads(fref.read_text())
        opt = ref["options"].get(X["floor_option"])
        matches = opt is not None and list(opt) == [cfg["profile"].get("sigma_t_textured_px"), X.get("profile_ref", "H_I")] \
            and ref.get("k_tex") == X["k_tex"] and ref.get("tau") == tau0
        for g in groups:
            cnt = next(iter(summ[g].values()))["texture_strata"]
            floors[g] = dict(counts=cnt, matches_config=matches,
                             **(AG.gt_floor(cnt, ref, X["floor_option"]) if matches else {}))
    # ---- per_image.csv
    dkeys = ["D1", "D1_n", "D2", "D2_n", "D3", "D3_n", "D4"] + [f"A{k}_q{q:.2f}" for k in (1, 2) for q in cfg["dense"]["anchor_q"]]
    empty = {k: SM.rate([]) for k in SM.RATED}                         # images with no pairs yet: explicit n/a rows
    by_img = {(m, s): AG.per_image([r for r in rows[m] if r["rec"]["image"] == s], tau0).get(s, empty) for m in methods for s in stems}
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "image", "set"] + [f"{k}_{c}" for k in SM.RATED for c in ("n_eligible", "n_fail", "ER")] + dkeys)
        for d in dense_rows:
            pi = by_img[(d["method"], d["image"])]
            w.writerow([d["method"], d["image"], d["set"]] +
                       [x for k in SM.RATED for x in (pi[k]["n"], pi[k]["k"], pi[k]["rate"] if pi[k]["rate"] is not None else "n/a: 0 pairs")] +
                       [d[k] for k in dkeys])
    # ---- dense summary per group and method (mean over images where available)
    dense_summ = {}
    for g, imgs in groups.items():
        for m in methods:
            ds = [d for d in dense_rows if d["method"] == m and d["image"] in imgs]
            row = {}
            for k in ("D1", "D2", "D3"):
                v = [d[k] for d in ds if not isinstance(d[k], str)]
                row[k] = dict(mean=float(np.mean(v)) if v else None, n_images=len(v), n_na=len(ds) - len(v))
            row["D4"] = DN.NA_REGION
            dense_summ[f"{g}/{m}"] = row
    le = lit_end_list({s: files[s] for s in stems})
    meta = dict(tool="metrics/run_metrics.py", version=VERSION, started=t_start.isoformat(timespec="seconds"),
                finished=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), methods=methods,
                pred_dir=str(Path(a.pred).resolve()) if a.pred else None, pred_sha256=pred_sha, images=stems,
                groups=groups, taus=taus, tau=tau0, config={k: v for k, v in cfg.items() if not k.startswith("_")},
                config_path=cfg["_path"], config_sha256=K.sha256(cfg["_path"]), git=git_state(),
                detector_versions=map_meta(cfg), map_sources=map_src, geom_sha256=geom_sha,
                annotation_sha256=ann_sha, gt_floor=floors, report_splits=splits_on, skipped_test_images=len(skipped_test),
                train_val_gap=gaps,
                gt_floor_reference=dict(path=cfg["texture"]["floor_reference"], sha256=K.sha256(fref) if fref.exists() else None))
    (out / "summary.json").write_text(json.dumps(dict(meta=meta, summary=summ, dense=dense_summ, lit_end=le),
                                                 indent=1, default=str), encoding="utf-8")
    (out / "summary.md").write_text(render_md(meta, summ, dense_summ, le, cfg), encoding="utf-8")
    print(out / "summary.md")


def render_md(meta, summ, dense, le, cfg):
    t0 = meta["tau"]
    L = [f"# Metrics v1 run: {', '.join(meta['methods'])}", "",
         f"{meta['finished']} · {len(meta['images'])} images · tau {t0} (sweep {', '.join(f'{t:g}' for t in meta['taus'])}) · "
         f"config SHA-256 `{meta['config_sha256'][:12]}` · git HEAD `{(meta['git']['head'] or '')[:10]}` "
         f"({meta['git']['head_branch']}), metrics-v1 `{(meta['git']['metrics_v1'] or '')[:10]}`, metrics/ matches "
         f"metrics-v1: {meta['git'].get('metrics_matches_metrics_v1')} · detector arm `{meta['detector_versions']['detector_arm']}`", "",
         "Headline = the widened rule (R-flat kinds: E <= max(tau, 2.5 se_tex), H_R <= max(tau, H_I), sigma_t 3 px on "
         "textured pairs; m plain tau). Plain-tau values in brackets. Error rates are micro over eligible pairs; CIs: "
         f"cluster bootstrap over images ({cfg['aggregate']['bootstrap_n']} resamples, seed {cfg['aggregate']['bootstrap_seed']}). "
         "Never read the composite alone.", ""]
    if not meta["pred_dir"]:
        L += ["**Baselines only:** no prediction folder was given (no network exists yet, Delta B1).", ""]
    if meta.get("skipped_test_images"):
        L += [f"Test split: {meta['skipped_test_images']} images not loaded or reported (run with --test to report them).", ""]
    if meta.get("train_val_gap"):
        L += ["## Train - val gap (val minus train, widened rule; positive = worse on val)", "",
              "| method | Split error | Leak | Long-range | composite |", "|---|---|---|---|---|"]
        for m, gp in meta["train_val_gap"].items():
            L.append(f"| {m} | " + " | ".join("n/a" if gp[k] is None else f"{gp[k]:+.3f}" for k in ("split", "leak", "long_range", "composite")) + " |")
        L.append("")
    for g, by_m in summ.items():
        sup = g.startswith("supervised fit")
        L += [f"## {'Supervised fit (used_in_loss pairs; never pooled with unused pairs)' if sup else 'Data split'}: "
              f"{g} ({len(meta['groups'][g])} images; never pooled with another split)", ""]
        fl = meta.get("gt_floor", {}).get(g)
        if fl and fl.get("matches_config") and fl["lo"]["split"] is not None:
            ftxt = f"≈ {fl['lo']['split']:.3f}–{fl['hi']['split']:.3f}"
        else:
            ftxt = "n/a" if not fl else "n/a: Monte Carlo option differs from this config"
        L += [f"Expected Split error of a perfect decomposition (approximate; synthetic texture): **{ftxt}**. From "
              f"`{cfg['texture']['floor_reference']}` ({cfg['texture']['floor_option']}), weighted by this group's eligible "
              f"stratum counts" + (f" (i smooth / moderate / strong {fl['counts']['i']['smooth']} / {fl['counts']['i']['moderate']} / "
                                   f"{fl['counts']['i']['strong']}; m {fl['counts']['m']['smooth']} / {fl['counts']['m']['moderate']} / "
                                   f"{fl['counts']['m']['strong']})" if fl else "") +
              ". Range: smooth-stratum i pairs counted at 0 (smooth material) to the rate of not-textured pairs on textured "
              "material; m pairs have a 0 floor in the Monte Carlo.", ""]
        L += ["| method | Split | GT floor (approx.) | 95% CI | Leak | Long-range | composite (never alone) | ER_i | ER_m |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m, S in by_m.items():
            h, hp, ci = S["headline"][t0]["wide"], S["headline"][t0]["plain"], S["ci"]
            L.append(f"| {m} | {fmt(h['split'] if h['split'] is not None else h['text']['split'])} ({fmt(hp['split'])}) | {ftxt} | "
                     f"{ci_txt(ci, 'split')} | {h['text']['leak']} | {h['text']['long_range']} | {h['text']['composite']} | "
                     f"{h['ER']['i']['text']} {ci_txt(ci, 'i')} | {h['ER']['m']['text']} {ci_txt(ci, 'm')} |")
        L += ["", f"tau sweep, Split (widened). Per-tau population: pairs eligible at that tau (low_contrast and low_snr move with "
              f"tau). Fixed population: the pairs eligible at tau {t0:g}, judged at each tau, which isolates the threshold effect:", "",
              "| method | population | " + " | ".join(f"tau {t:g}" for t in meta["taus"]) + " |", "|---|---|" + "---|" * len(meta["taus"])]
        for m, S in by_m.items():
            for nm, key in (("per-tau", "headline"), (f"fixed (tau {t0:g})", "headline_fixed_pop")):
                L.append(f"| {m} | {nm} | " + " | ".join(
                    f"{fmt(S[key][t]['wide']['split'])} (i {S[key][t]['wide']['ER']['i']['n']}, m {S[key][t]['wide']['ER']['m']['n']})"
                    for t in meta["taus"]) + " |")
        S0 = next(iter(by_m.values()))
        L += ["", f"Exclusions at tau {t0} (they depend only on ℓI and the annotations, so they are the same for every "
              "method; a pair can have several reasons):", "",
              "| kind | total | eligible | reasons |", "|---|---|---|---|"]
        for k, c in sorted(S0["exclusions"][t0].items()):
            rs = ", ".join(f"{r} {n}" for r, n in sorted(c.items()) if not r.startswith("_"))
            L.append(f"| {k} | {c['_total']} | {c['_eligible']} | {rs or '-'} |")
        L += ["", "low_contrast (i, m with ||ΔℓI|| <= tau; excluded from Split) by tau: " +
              ", ".join(f"tau {t:g}: {sum(S0['exclusions'][t].get(k, {}).get('low_contrast', 0) for k in ('i', 'm'))}" for t in meta["taus"]), ""]
        L += ["Split bins of failing eligible pairs (widened; plain in brackets):", "",
              "| method | kind | " + " | ".join(PM.BINS) + " |", "|---|---|" + "---|" * len(PM.BINS)]
        for m, S in by_m.items():
            for k in ("i", "m"):
                L.append(f"| {m} | {k} | " + " | ".join(f"{S['bins'][k][b]} ({S['bins_plain'][k][b]})" for b in PM.BINS) + " |")
        L += ["", "Continuous companions (eligible pairs; E and H mean / median; leaked fraction mean):", "",
              "| method | kind | n | E mean | E median | H mean | H median | leaked frac |", "|---|---|---|---|---|---|---|---|"]
        for m, S in by_m.items():
            for k, c in S["companions"].items():
                if c["n"]:
                    e, hh = c["E"] or {}, c["H"] or {}
                    L.append(f"| {m} | {k} | {c['n']} | {fmt(e.get('mean'))} | {fmt(e.get('median'))} | {fmt(hh.get('mean'))} | "
                             f"{fmt(hh.get('median'))} | {fmt(c.get('leaked_frac_mean'))} |")
        L += ["", "b pairs (diagnostic): " + "; ".join(f"{m}: both-step {S['b']['both_step']['text']}, alpha quartiles "
                                                       f"{S['b']['alpha_quartiles']}" for m, S in by_m.items()), ""]
        L += ["Strata (widened ER; eligible pairs):", ""]
        for name in S0["strata"]:
            L.append(f"- **{name}:** " + "; ".join(f"{m}: " + ", ".join(f"{k} {v['text']}" for k, v in S["strata"][name].items())
                                                   if S["strata"][name] else f"{m}: n/a: 0 pairs" for m, S in by_m.items()))
        L += ["", "m error with / without input_excursion pairs: " +
              "; ".join(f"{m}: {S['m_excursion']['all']['text']} / {S['m_excursion']['without_input_excursion']['text']}"
                        for m, S in by_m.items()), ""]
        cm = next((m for m in by_m if m not in ("all_R", "all_L")), next(iter(by_m)))
        L += [f"input_excursion x textured x geom_cross (pair counts; widened ER of eligible i / m pairs for {cm}, the "
              "first method that is not a degenerate all-R / all-L baseline):", ""]
        ct = by_m[cm]["crosstab"]
        L += ["| cell | i n | i ER | m n | m ER | b n |", "|---|---|---|---|---|---|"]
        cells = sorted(set(ct["i"]) | set(ct["m"]) | set(ct["b"]))
        for c in cells:
            gi, gm, gb = ct["i"].get(c), ct["m"].get(c), ct["b"].get(c)
            L.append(f"| {c} | {gi['n'] if gi else 0} | {gi['ER']['text'] if gi else '-'} | {gm['n'] if gm else 0} | "
                     f"{gm['ER']['text'] if gm else '-'} | {gb['n'] if gb else 0} |")
        L += ["", "Dense diagnostics (mean over images with the map; never headline):", "",
              "| method | D1 off-edge L roughness | D2 shadow leak into R | D3 material leak into L | D4 | A1 / A2 |", "|---|---|---|---|---|---|"]
        for m in by_m:
            d = dense[f"{g}/{m}"]
            cell = lambda k: (f"{d[k]['mean']:.4f} ({d[k]['n_images']} img" + (f", {d[k]['n_na']} n/a" if d[k]["n_na"] else "") + ")"
                              if d[k]["mean"] is not None else DN.NA_MAP)
            L.append(f"| {m} | {cell('D1')} | {cell('D2')} | {cell('D3')} | {DN.NA_REGION} | {DN.NA_REGION} |")
        L.append("")
    L += ["## Textured stratum and the perfect-decomposition floor", "",
          "Textured pairs (se_tex > 0.0333) stay in the headline; their ER is in the strata above ('by textured'). On the synthetic "
          "scene (30 seeds; `" + cfg["texture"]["floor_reference"] + "`), ground-truth ℓR fails the widened rule on: "
          + "; ".join(f"{k}" for k in floor_lines(cfg)) + ". The Split floors in the headline tables combine these with each "
          "group's stratum counts (approximate; synthetic iid texture, real texture is spatially correlated).", ""]
    L += ["## lit_end: migrated i pairs inferred B-lit (guides: A lit)", "",
          f"{len(le)} pairs ({sum(x['marginal'] for x in le)} marginal, |b_B − b_A| < 0.5). lit_end only feeds A1/A2 (n/a).", "",
          "| set | image | id | b_A | b_B | b_B − b_A | marginal |", "|---|---|---|---|---|---|---|"]
    for x in sorted(le, key=lambda x: (not x["marginal"], x["set"], x["image"])):
        L.append(f"| {x['set']} | {x['image']} | {x['id']} | {x['b_A']:.3f} | {x['b_B']:.3f} | {x['diff']:+.3f} | {'yes' if x['marginal'] else ''} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
