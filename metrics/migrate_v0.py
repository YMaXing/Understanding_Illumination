#!/usr/bin/env python3
"""Lossless migration of the existing i/m/b pairs (annotate_pairs.html JSON version 2) to schema v1.

    metrics/run.sh migrate_v0.py run [--force]     # write metrics/annotations/<stem>.json + metrics/migration_v0/
    metrics/run.sh migrate_v0.py verify            # rebuild every source's pairs / deleted lists exactly; schema check
    metrics/run.sh migrate_v0.py counts            # print the per-image, per-type table from the written files

Sources (config.yaml sets.*.source) are read-only and SHA-256 checked first. Per pair (DESIGN.md §3, Deltas C3–C6):
  id = "t=<t>"; A = {x: a_col, y: a_row, r: half_a or half}; B likewise; created = ISO time of t (ms epoch, the old
  tool's Date.now() id); annotator = migration.annotator; confidence / split / overlay_seen / note = §3 defaults;
  i pairs: lit_end = the end with the brighter b(P(ℓI)) (§3), cross-checked against the guides' "A lit" rule;
  config migration.relabels applied (070 t=1794165897383 m -> b).
  legacy = {v0: the original record verbatim, v0_index: its index in the source's pairs list, [lit_end], [relabel]}.
Per image: set, split (paths.split_assignment), seen_in_training (seen_in_training_default), size, image SHA-256, and legacy = source set /
file / SHA-256 and the source's deleted records on this image, verbatim with their v0_index (never scored).
File-level source fields (header, candidates, verdict_codes, image_log, ...) go to migration_v0/manifest.json.
Existing annotation files are never overwritten without --force (after Phase 4 they hold new work).
"""

import argparse
import copy
import csv
import datetime as dt
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from core import config as K
from core import io as IO
from core import patches as PT
from core import annotators as AN
from core import schema as SC

VERSION = "1.0"
TYPES = ("i", "m", "b", "s", "d")


def created_from_t(t):
    return dt.datetime.fromtimestamp(int(t) / 1000, dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_sources(cfg):
    out = {}
    for name, s in cfg["sets"].items():
        if not s.get("source"):
            continue                                   # sets with new annotations only (confirmation)
        p = K.repo_path(s["source"])
        got = K.sha256(p)
        if got != s["sha256"]:
            raise SystemExit(f"{name}: SHA-256 of {p} is {got}, config says {s['sha256']}")
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("version") != 2:
            raise SystemExit(f"{name}: expected annotate_pairs JSON version 2, got {obj.get('version')}")
        out[name] = dict(path=p, rel=s["source"], sha=got, obj=obj)
    return out


def end(q, which):
    h = q.get(f"half_{which}", q["half"])
    return dict(x=int(q[f"{which}_col"]), y=int(q[f"{which}_row"]), r=int(h))


def infer_lit_end(img, A, B, P):
    """§3: the end with the brighter b(P(ℓI)); falls back to the guides' A-lit rule if a patch has no valid pixel."""
    sa = PT.patch_stats(img["lI"], img["valid"], A["x"], A["y"], A["r"], P)
    sb = PT.patch_stats(img["lI"], img["valid"], B["x"], B["y"], B["r"], P)
    ba, bb = PT.brightness(sa["median"]), PT.brightness(sb["median"])
    if sa["n_valid"] == 0 or sb["n_valid"] == 0:
        return "A", dict(rule="guide A-lit (a patch has no valid pixel)", b_A=None, b_B=None, agrees_guide_A_lit=True)
    le = "A" if ba >= bb else "B"
    return le, dict(rule="brighter b(P(lI))", b_A=round(ba, 6), b_B=round(bb, 6), agrees_guide_A_lit=le == "A")


def build(cfg, log=print):
    src = load_sources(cfg)
    M = cfg["migration"]
    P = cfg["patch"]
    relabels = {r["id"]: r for r in M["relabels"]}
    split_of = {x["stem"]: x["split"] for x in json.loads(K.repo_path(cfg["paths"]["split_assignment"]).read_text())["images"]}
    applied = []
    files, manifest_sources = {}, {}
    for name, s in src.items():
        obj = s["obj"]
        listed = set(cfg["image_sets"][name])
        seen = {q["image"] for q in obj["pairs"]} | {q["image"] for q in obj["deleted"]}
        if seen - listed:
            raise SystemExit(f"{name}: images with pairs not in config splits.eval.{name}: {sorted(seen - listed)}")
        manifest_sources[name] = dict(file=s["rel"], sha256=s["sha"],
                                      fields={k: v for k, v in obj.items() if k not in ("pairs", "deleted")})
        by_img = defaultdict(lambda: dict(pairs=[], deleted=[]))
        for k, q in enumerate(obj["pairs"]):
            by_img[q["image"]]["pairs"].append((k, q))
        for k, q in enumerate(obj["deleted"]):
            by_img[q["image"]]["deleted"].append((k, q))
        for stem in sorted(listed):
            g = by_img[stem]
            ipath = K.image_path(cfg, stem)
            img = IO.load_image(cfg, stem, ipath)
            pairs = []
            for k, q in g["pairs"]:
                pid = f"t={int(q['t'])}"
                A, B = end(q, "a"), end(q, "b")
                p = dict(id=pid, image=stem, type=q["type"], A=A, B=B, confidence=M["defaults"]["confidence"],
                         used_in_loss=M["defaults"]["used_in_loss"], overlay_seen=M["defaults"]["overlay_seen"],
                         annotator=M["annotator"], created=created_from_t(q["t"]), note=M["defaults"]["note"],
                         schema_version=cfg["schema_version"])
                leg = dict(v0=copy.deepcopy(q), v0_index=k)
                if pid in relabels:
                    r = relabels[pid]
                    if r["image"] != stem or r["from"] != q["type"]:
                        raise SystemExit(f"relabel {pid}: expected {r['image']} type {r['from']}, found {stem} {q['type']}")
                    p["type"] = r["to"]
                    leg["relabel"] = {k2: r[k2] for k2 in ("from", "to", "by", "at", "reason")}
                    applied.append(dict(set=name, image=stem, id=pid, **leg["relabel"]))
                if p["type"] == "i":
                    p["lit_end"], leg["lit_end"] = infer_lit_end(img, A, B, P)
                p["legacy"] = leg
                pairs.append(p)
            deleted = [dict(v0=copy.deepcopy(q), v0_index=k) for k, q in g["deleted"]]
            files[stem] = dict(schema_version=cfg["schema_version"], image=stem, set=name, split=split_of[stem],
                               seen_in_training=cfg["seen_in_training_default"], width=img["W"],
                               height=img["H"], image_sha256=K.sha256(ipath), pairs=pairs, s_seed_log=[],
                               legacy=dict(source_set=name, source_file=s["rel"], source_sha256=s["sha"],
                                           migrated_by=f"metrics/migrate_v0.py {VERSION}", deleted=deleted))
            log(f"  {name:15s} {stem:22s} {len(pairs):4d} pairs, {len(deleted):3d} deleted")
    missing = set(relabels) - {a["id"] for a in applied}
    if missing:
        raise SystemExit(f"relabels not applied (pair not found): {sorted(missing)}")
    manifest = dict(tool="metrics/migrate_v0.py", version=VERSION, schema_version=cfg["schema_version"],
                    code_sha256={f: K.sha256(K.METRICS / f) for f in ("migrate_v0.py", "core/io.py", "core/patches.py",
                                                                       "core/config.py", "core/schema.py")},
                    config_sha256=K.sha256(cfg["_path"]), loader=cfg["loader"], patch_shape=P["shape"],
                    annotator=M["annotator"], defaults=M["defaults"], relabels_applied=applied,
                    sources=manifest_sources)
    return files, manifest


def count_rows(files, name_of=lambda i: i):
    """Per image and type: v0 count (before), v1 count (after), deleted v0 records; with the annotator column."""
    rows = []
    for stem, f in sorted(files.items(), key=lambda kv: (kv[1]["set"], kv[0])):
        dele = Counter(d["v0"]["type"] for d in f["legacy"].get("deleted", []))
        mig = [p for p in f["pairs"] if "v0" in p.get("legacy", {})]
        dele_ann = mig[0]["annotator"] if mig else ""        # deleted v0 records belong to the migrated annotator
        ann = sorted({p["annotator"] for p in f["pairs"]} | ({dele_ann} if dele else set())) or [""]
        for a in ann:
            mine = [p for p in f["pairs"] if p["annotator"] == a]
            before = Counter(p["legacy"]["v0"]["type"] for p in mine if "v0" in p.get("legacy", {}))
            after = Counter(p["type"] for p in mine)
            for t in TYPES:
                b, n, d = before[t], after[t], dele[t] if a == dele_ann else 0
                if b or n or d:
                    rows.append(dict(set=f["set"], image=stem, annotator=name_of(a), annotator_id=a, type=t, before=b, after=n, deleted_v0=d))
    return rows


def print_counts(rows, out=sys.stdout):
    tab = defaultdict(lambda: {t: [0, 0, 0] for t in TYPES})
    for r in rows:
        tab[(r["set"], r["image"], r["annotator"])][r["type"]] = [r["before"], r["after"], r["deleted_v0"]]
    hdr = f"{'set':15s} {'image':22s} {'annotator':9s} " + " ".join(f"{t + ' b>a (del)':>15s}" for t in "imb") + f" {'total b>a':>11s}"
    print(hdr, file=out)
    tot = defaultdict(lambda: [0, 0, 0])
    sets = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for (s, im, a), d in sorted(tab.items()):
        cells = []
        for t in "imb":
            b, n, dl = d[t]
            cells.append(f"{b:5d}>{n:<4d}({dl:3d})")
            for k in range(3):
                tot[t][k] += d[t][k]
                sets[s][t][k] += d[t][k]
        tb, ta = sum(d[t][0] for t in TYPES), sum(d[t][1] for t in TYPES)
        print(f"{s:15s} {im:22s} {a:9s} " + " ".join(f"{c:>15s}" for c in cells) + f" {tb:5d}>{ta:<5d}", file=out)
    print("-" * len(hdr), file=out)
    for s, d in sorted(sets.items()) + [("ALL", tot)]:
        cells = [f"{d[t][0]:5d}>{d[t][1]:<4d}({d[t][2]:3d})" for t in "imb"]
        tb, ta = sum(d[t][0] for t in TYPES), sum(d[t][1] for t in TYPES)
        print(f"{s:15s} {'':22s} {'':9s} " + " ".join(f"{c:>15s}" for c in cells) + f" {tb:5d}>{ta:<5d}", file=out)
    print("cells: v0 count > v1 count (deleted v0 records, kept in legacy, never scored); s and d: none in v0", file=out)


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_files(cfg):
    d = K.repo_path(cfg["paths"]["annotations"])
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json")) if p.name != "annotators.json"}


def cmd_run(cfg, force):
    ann = K.repo_path(cfg["paths"]["annotations"])
    mig = K.repo_path(cfg["paths"]["migration"])
    print("building (lit_end needs ℓI, so every image is loaded):")
    files, manifest = build(cfg)
    exist = [s for s in files if (ann / f"{s}.json").exists()]
    if exist and not force:
        raise SystemExit(f"{len(exist)} annotation files exist (e.g. {exist[0]}); refusing to overwrite without --force")
    for stem, f in files.items():
        errs = SC.validate_file(f)
        if errs:
            raise SystemExit(f"{stem}: schema errors: {errs[:5]}")
    for stem, f in files.items():
        atomic_write_json(ann / f"{stem}.json", f)
    ap = ann / "annotators.json"
    if not ap.exists():
        reg = dict(schema_version="1.2", annotators=[])
        AN.add(reg, "Yueren", id_=cfg["migration"]["annotator"])
        AN.save(ap, reg)
    rows = count_rows(files, AN.resolve(ap))
    manifest["migrated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    manifest["annotation_files"] = {s: K.sha256(ann / f"{s}.json") for s in sorted(files)}
    le = [p["legacy"]["lit_end"] for f in files.values() for p in f["pairs"] if p["type"] == "i"]
    manifest["lit_end_summary"] = dict(n_i=len(le), inferred_B=sum(x["rule"].startswith("brighter") and not x["agrees_guide_A_lit"] for x in le),
                                       fallback=sum(not x["rule"].startswith("brighter") for x in le))
    atomic_write_json(mig / "manifest.json", manifest)
    with open(mig / "counts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "image", "annotator", "annotator_id", "type", "before", "after", "deleted_v0"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(files)} files to {ann}, manifest + counts.csv to {mig}\n")
    print_counts(rows)
    print(f"\nrelabels applied: {manifest['relabels_applied']}")
    print(f"lit_end (i pairs): {manifest['lit_end_summary']}")
    problems = verify(cfg)
    print("verify:", "OK" if not problems else problems)
    return 0 if not problems else 1


def verify(cfg):
    """Every source's pairs and deleted lists rebuilt from the v1 files must equal the source exactly."""
    problems = []
    src = load_sources(cfg)
    files = read_files(cfg)
    mig = K.repo_path(cfg["paths"]["migration"])
    man = json.loads((mig / "manifest.json").read_text(encoding="utf-8")) if (mig / "manifest.json").exists() else None
    relabels = {r["id"]: r for r in cfg["migration"]["relabels"]}
    for stem, f in files.items():
        for e in SC.validate_file(f):
            problems.append(f"{stem}: {e}")
    for name, s in src.items():
        mine = {st: f for st, f in files.items() if f["set"] == name}
        if set(mine) != set(cfg["image_sets"][name]):
            problems.append(f"{name}: files {sorted(mine)} != config list")
        P = sorted(((p["legacy"]["v0_index"], p) for f in mine.values() for p in f["pairs"]), key=lambda kv: kv[0])
        D = sorted(((d["v0_index"], d["v0"]) for f in mine.values() for d in f["legacy"].get("deleted", [])), key=lambda kv: kv[0])
        if [p["legacy"]["v0"] for _, p in P] != s["obj"]["pairs"] or [k for k, _ in P] != list(range(len(s["obj"]["pairs"]))):
            problems.append(f"{name}: rebuilt pairs list differs from the source")
        if [d for _, d in D] != s["obj"]["deleted"] or [k for k, _ in D] != list(range(len(s["obj"]["deleted"]))):
            problems.append(f"{name}: rebuilt deleted list differs from the source")
        if man is not None:
            hdr = {k: v for k, v in s["obj"].items() if k not in ("pairs", "deleted")}
            if man["sources"][name]["fields"] != hdr or man["sources"][name]["sha256"] != s["sha"]:
                problems.append(f"{name}: manifest source fields differ from the source")
        for _, p in P:
            q = p["legacy"]["v0"]
            want_type = relabels[p["id"]]["to"] if p["id"] in relabels else q["type"]
            if (p["id"] != f"t={int(q['t'])}" or p["type"] != want_type or p["A"] != end(q, "a") or p["B"] != end(q, "b")
                    or p["image"] != q["image"]):
                problems.append(f"{name}: {p['id']} v1 fields inconsistent with v0")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "verify", "counts"])
    ap.add_argument("--config")
    ap.add_argument("--force", action="store_true", help="overwrite existing annotation files (before Phase 4 only)")
    a = ap.parse_args()
    cfg = K.load_config(a.config)
    if a.cmd == "run":
        return cmd_run(cfg, a.force)
    if a.cmd == "verify":
        pr = verify(cfg)
        print("verify:", "OK" if not pr else "\n".join(pr))
        return 0 if not pr else 1
    print_counts(count_rows(read_files(cfg), AN.resolve(K.repo_path(cfg["paths"]["annotations"]) / "annotators.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
