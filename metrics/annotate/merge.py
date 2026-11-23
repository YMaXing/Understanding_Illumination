#!/usr/bin/env python3
"""Merge another machine's annotation folder into this one (stdlib only). Dry run unless --apply.

    python3 metrics/annotate/merge.py --from OTHER_DIR [--into metrics/annotations] [--apply]

Per image file present in OTHER_DIR:
  - absent here: copied (after schema validation);
  - present: pairs are unioned by id. Same id + identical content: kept once. Same id + different content: a CONFLICT,
    reported; this folder's version is kept (never overwritten). An id deleted on either side (deleted_pairs tombstone)
    is not resurrected; a pair edited on one side and deleted on the other is a conflict.
  - exhausted marks: union by type; the same type with a different mark is a conflict (this folder's kept). A mark
    removed on one side comes back from the other (there are no tombstones for marks).
  - s_seed_log: union by seed (the seeded RNG gives both machines the same seed for the same index). Same seed with a
    different outcome: a conflict, this folder's entry kept. Seeds only in OTHER_DIR are appended in index order.
annotators.json: union by id; a rename recorded only in OTHER_DIR's history is taken over; the same id with a different
name otherwise (renamed on both sides, or no record) is a conflict (reported, never overwritten). relabels/<stem>.json: union by (pair_id, relabeler); differing entries are conflicts. Every written file is schema-validated; the old one is backed up to
<into>/.backups/<stem>/ first. Exit code 1 if there were conflicts (after writing the non-conflicting union).
"""

import argparse
import copy
import datetime as dt
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core import annotators as AN  # noqa: E402
from core import schema as SC  # noqa: E402


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(path, obj, backups):
    if path.exists():
        bdir = backups / path.stem
        bdir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        (bdir / f"{path.stem}.{stamp}.merge.json").write_bytes(path.read_bytes())
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def merge_file(mine, theirs):
    """Returns (merged, report dict)."""
    out = copy.deepcopy(mine)
    rep = dict(added=[], conflicts=[], kept_deleted=[])
    if mine["image"] != theirs["image"] or mine["set"] != theirs["set"]:
        rep["conflicts"].append(f"file header differs (image/set): {mine['image']}/{mine['set']} vs {theirs['image']}/{theirs['set']}")
        return out, rep
    tomb = {t["id"]: t for t in mine.get("deleted_pairs", [])}
    for t in theirs.get("deleted_pairs", []):
        if t["id"] not in tomb:
            tomb[t["id"]] = t
    mp = {p["id"]: p for p in mine["pairs"]}
    for p in theirs["pairs"]:
        if p["id"] in mp:
            if p != mp[p["id"]]:
                rep["conflicts"].append(f"{p['id']}: edited differently on the two machines (kept this folder's)")
        elif p["id"] in tomb:
            if p != tomb[p["id"]]["pair"]:
                rep["conflicts"].append(f"{p['id']}: deleted on one side, edited on the other (kept deleted)")
            else:
                rep["kept_deleted"].append(p["id"])
        else:
            out["pairs"].append(copy.deepcopy(p))
            rep["added"].append(p["id"])
    for pid in list(mp):
        if pid in tomb and pid not in {t["id"] for t in mine.get("deleted_pairs", [])}:
            if mp[pid] == tomb[pid]["pair"]:
                out["pairs"] = [q for q in out["pairs"] if q["id"] != pid]
                rep["kept_deleted"].append(pid)
            else:
                rep["conflicts"].append(f"{pid}: deleted on the other machine, edited here (kept this folder's)")
    out["deleted_pairs"] = [tomb[k] for k in sorted(tomb) if not any(p["id"] == k for p in out["pairs"])]
    seeds = {e["seed"]: e for e in mine.get("s_seed_log", [])}
    for e in theirs.get("s_seed_log", []):
        if e["seed"] in seeds:
            if e != seeds[e["seed"]]:
                rep["conflicts"].append(f"seed {e['seed']}: different outcome on the two machines (kept this folder's)")
        else:
            out["s_seed_log"].append(copy.deepcopy(e))
            rep["added"].append(f"seed {e['seed']}")
    out["s_seed_log"].sort(key=lambda e: e["index"])
    ex = dict(mine.get("exhausted") or {})                     # exhausted marks: union by type; a type marked
    for k, v in (theirs.get("exhausted") or {}).items():      # differently on the two sides is a conflict (ours kept)
        if k not in ex:
            ex[k] = copy.deepcopy(v)
            rep["added"].append(f"exhausted {k}")
        elif ex[k] != v:
            rep["conflicts"].append(f"exhausted {k}: marked differently on the two machines (kept this folder's)")
    if ex or "exhausted" in out:
        out["exhausted"] = ex or out.get("exhausted")
    return out, rep


def merge_annotators(mine, theirs):
    """Union by id. A tag only on their side is added; the same id with a different name is a conflict (reported,
    this folder's name kept); rename histories are unioned."""
    have = {e["id"]: e for e in mine["annotators"]}
    add, conflicts = [], []
    for e in theirs["annotators"]:
        h = have.get(e["id"])
        if h is None:
            if any(x["name"].casefold() == e["name"].casefold() for x in mine["annotators"]):
                conflicts.append(f"{e['id']}: name {e['name']!r} is already used here by another id (not added)")
                continue
            add.append(copy.deepcopy(e))
            continue
        seen = {json.dumps(x, sort_keys=True) for x in h["history"]}
        new = [x for x in e["history"] if json.dumps(x, sort_keys=True) not in seen]
        theirs_seen = {json.dumps(x, sort_keys=True) for x in e["history"]}
        ours_only = [x for x in h["history"] if json.dumps(x, sort_keys=True) not in theirs_seen]
        if h["name"] != e["name"]:
            # A rename recorded only on their side (their history extends ours and ends in their name) is taken over;
            # a rename on both sides, or an unrecorded difference, is a conflict and this folder's name is kept.
            renamed_there = new and not ours_only and e["history"][-1]["to"] == e["name"]
            clash = any(x["name"].casefold() == e["name"].casefold() for x in mine["annotators"] if x["id"] != e["id"])
            if renamed_there and not clash:
                h["name"] = e["name"]
                mine.setdefault("_renamed", []).append(f"{e['id']}: {e['history'][-1]['from']!r} -> {e['name']!r}")
            else:
                conflicts.append(f"{e['id']}: named {h['name']!r} here and {e['name']!r} there (kept {h['name']!r})")
        if new:
            h["history"] = sorted(h["history"] + new, key=lambda x: x["at"])
            mine.setdefault("_changed", True)
    return add, conflicts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--into", default=str(HERE.parent / "annotations"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.into)
    backups = dst / ".backups"
    n_conf = 0
    for p in sorted(src.glob("*.json")):
        if p.name == "annotators.json":
            continue
        theirs = load(p)
        errs = SC.validate_file(theirs)
        if errs:
            print(f"{p.name}: SKIPPED, invalid in --from: {errs[:3]}")
            n_conf += 1
            continue
        q = dst / p.name
        if not q.exists():
            print(f"{p.stem}: new file ({len(theirs['pairs'])} pairs)")
            if a.apply:
                write(q, theirs, backups)
            continue
        merged, rep = merge_file(load(q), theirs)
        errs = SC.validate_file(merged)
        if errs:
            print(f"{p.stem}: merge result invalid, not written: {errs[:3]}")
            n_conf += 1
            continue
        n_conf += len(rep["conflicts"])
        print(f"{p.stem}: +{len(rep['added'])}, deleted kept {len(rep['kept_deleted'])}, conflicts {len(rep['conflicts'])}")
        for c in rep["conflicts"]:
            print(f"    CONFLICT {c}")
        if a.apply and (rep["added"] or rep["kept_deleted"]):
            write(q, merged, backups)
    rs, rd = src / "relabels", dst / "relabels"                  # blind relabels: union by (pair_id, relabeler)
    for p in sorted(rs.glob("*.json")) if rs.exists() else []:
        theirs = load(p)
        q = rd / p.name
        mine = load(q) if q.exists() else dict(schema_version=theirs["schema_version"], image=theirs["image"], relabels=[])
        have = {(e["pair_id"], e["relabeler"]): e for e in mine["relabels"]}
        add, conf = [], []
        for e in theirs["relabels"]:
            k = (e["pair_id"], e["relabeler"])
            if k not in have:
                add.append(e)
            elif have[k] != e:
                conf.append(k)
        n_conf += len(conf)
        if add or conf:
            print(f"relabels {p.stem}: +{len(add)}, conflicts {len(conf)}" + "".join(f"\n    CONFLICT relabel {k}" for k in conf))
        if a.apply and add:
            rd.mkdir(parents=True, exist_ok=True)
            write(q, dict(mine, relabels=mine["relabels"] + add), backups)
    sa, da = src / "annotators.json", dst / "annotators.json"
    if sa.exists():
        mine = AN.load(da)
        theirs = AN.load(sa)
        add, conflicts = merge_annotators(mine, theirs)
        n_conf += len(conflicts)
        for c in conflicts:
            print(f"    CONFLICT name tag {c}")
        if add:
            print(f"annotators: + {[(e['id'], e['name']) for e in add]}")
        changed = mine.pop("_changed", False)
        for r in mine.pop("_renamed", []):
            print(f"annotators: renamed there, taken over: {r}")
        if a.apply and (add or changed):
            mine["annotators"] += add
            write(da, mine, backups)
    print(("applied" if a.apply else "dry run (nothing written; add --apply)") + f"; conflicts: {n_conf}")
    return 1 if n_conf else 0


if __name__ == "__main__":
    sys.exit(main())
