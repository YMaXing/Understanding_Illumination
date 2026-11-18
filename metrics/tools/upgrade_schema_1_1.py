#!/usr/bin/env python3
"""Upgrade annotation files from schema 1.0 to 1.1 (owner, Checkpoint 5). Idempotent; validates before writing.

    metrics/run.sh tools/upgrade_schema_1_1.py [--dry-run]

Per file: schema_version 1.1; per-image `split` from paths.split_assignment (train / val / test). Per pair: the old
pair-level `split` (always "eval" so far) is replaced by used_in_loss = false; schema_version 1.1. Tombstoned pairs
(deleted_pairs[].pair) are kept verbatim as history. Files already at 1.1 are skipped.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config as K  # noqa: E402
from core import schema as SC  # noqa: E402
import migrate_v0 as MG  # noqa: E402


def upgrade(obj, split):
    if obj["schema_version"] == "1.1":
        return None
    assert obj["schema_version"] == "1.0", obj["schema_version"]
    out = {}
    for k, v in obj.items():                      # keep key order; insert split after set
        out[k] = v
        if k == "set":
            out["split"] = split
    out["schema_version"] = "1.1"
    pairs = []
    for p in obj["pairs"]:
        if p.get("split") != "eval":
            raise SystemExit(f"{obj['image']} {p['id']}: unexpected pair split {p.get('split')!r}")
        q = {}
        for k, v in p.items():
            if k == "split":
                q["used_in_loss"] = False
            elif k == "schema_version":
                q[k] = "1.1"
            else:
                q[k] = v
        pairs.append(q)
    out["pairs"] = pairs
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = K.load_config()
    split = {x["stem"]: x["split"] for x in json.loads(K.repo_path(cfg["paths"]["split_assignment"]).read_text())["images"]}
    ann = K.repo_path(cfg["paths"]["annotations"])
    n = 0
    for p in sorted(ann.glob("*.json")):
        if p.name == "annotators.json":
            continue
        obj = json.loads(p.read_text())
        new = upgrade(obj, split[obj["image"]])
        if new is None:
            continue
        errs = SC.validate_file(new)
        if errs:
            raise SystemExit(f"{p.name}: {errs[:5]}")
        if not a.dry_run:
            MG.atomic_write_json(p, new)
        n += 1
    print(f"{'would upgrade' if a.dry_run else 'upgraded'} {n} files")


if __name__ == "__main__":
    main()
