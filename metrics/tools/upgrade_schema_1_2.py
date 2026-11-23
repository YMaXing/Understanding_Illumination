#!/usr/bin/env python3
"""Upgrade annotation files 1.1 -> 1.2 and annotators.json to stable ids (owner, Checkpoint 6). Idempotent.

    metrics/run.sh tools/upgrade_schema_1_2.py [--dry-run]

annotators.json: each name becomes {id, name, created, history} (id = slug fixed now; "Yueren" -> "yueren"), and the
second annotator's tag is pre-created (--teammate-name, default "Annotator B"; id annotator_b; she may rename it).
Per image file: schema_version 1.2; assigned_to = the annotator-group owner (A -> yueren, B -> the teammate's id);
image_pixel_sha256 (core/io.pixel_sha256 of loader.read); reflective_advisory = true for the draw54 contact-sheet ranks
in REFLECTIVE_RANKS (advisory; nothing is re-split). Every annotator name becomes its id: pairs (annotator,
last_edited_by, history[].by), tombstones (deleted_by, pair.annotator, pair.last_edited_by), s_seed_log
(proposed_by, resolved_by), exhausted marks (by). legacy v0 records stay verbatim.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import annotators as AN  # noqa: E402
from core import config as K  # noqa: E402
from core import io as IO  # noqa: E402
from core import schema as SC  # noqa: E402
import migrate_v0 as MG  # noqa: E402

REFLECTIVE_RANKS = (1, 5, 10, 12, 16, 20, 28, 31, 33, 39, 45)          # draw54 contact sheet (owner, Checkpoint 6)
GROUP_OWNER = {"A": "yueren", "B": "annotator_b"}


def upgrade_annotators(path, teammate):
    obj = json.loads(Path(path).read_text())
    if obj.get("schema_version") == "1.2":
        return obj, False
    created = json.loads((K.METRICS / "migration_v0" / "manifest.json").read_text()).get("migrated_at", AN.now_iso())
    created = created if "." in created else created.replace("Z", ".000Z")
    new = dict(schema_version="1.2", annotators=[])
    for n in obj["annotators"]:
        e, _ = AN.add(new, n)
        e["created"] = created
    AN.add(new, teammate, id_="annotator_b")
    return new, True


def upgrade_file(obj, ids, pixel, assigned, reflective):
    if obj["schema_version"] == "1.2":
        return None
    assert obj["schema_version"] == "1.1", obj["schema_version"]
    to_id = lambda n: ids.get(n.casefold(), n) if isinstance(n, str) else n

    def pair(p):
        q = dict(p)
        for k in ("annotator", "last_edited_by"):
            if k in q:
                q[k] = to_id(q[k])
        if "schema_version" in q:
            q["schema_version"] = "1.2"
        if "history" in q:
            q["history"] = [dict(h, by=to_id(h["by"])) if "by" in h else h for h in q["history"]]
        return q
    out = {}
    for k, v in obj.items():
        out[k] = v
        if k == "split" and reflective:
            out["reflective_advisory"] = True
        if k == "image_sha256":
            out["image_pixel_sha256"] = pixel
            out["assigned_to"] = assigned
    out["schema_version"] = "1.2"
    out["pairs"] = [pair(p) for p in obj["pairs"]]
    if "deleted_pairs" in out:
        out["deleted_pairs"] = [dict(t, deleted_by=to_id(t["deleted_by"]), pair=pair(t["pair"])) for t in obj["deleted_pairs"]]
    out["s_seed_log"] = [{k: (to_id(v) if k in ("proposed_by", "resolved_by") else v) for k, v in e.items()} for e in obj["s_seed_log"]]
    if "exhausted" in out:
        out["exhausted"] = {k: dict(m, by=to_id(m["by"])) for k, m in obj["exhausted"].items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--teammate-name", default="Annotator B")
    a = ap.parse_args()
    cfg = K.load_config()
    ann = K.repo_path(cfg["paths"]["annotations"])
    ap_ = ann / "annotators.json"
    annot, changed = upgrade_annotators(ap_, a.teammate_name)
    errs = SC.validate(annot, SC.load_schema(K.METRICS / "schema" / "annotators.schema.json"))
    if errs:
        raise SystemExit(f"annotators.json: {errs}")
    ids = {e["name"].casefold(): e["id"] for e in annot["annotators"]}
    draw = json.loads((K.METRICS / "splits" / "draw54.json").read_text())["images"]
    reflective = {r["stem"] for r in draw if r["rank"] in REFLECTIVE_RANKS}
    n = 0
    for p in sorted(ann.glob("*.json")):
        if p.name == "annotators.json":
            continue
        obj = json.loads(p.read_text())
        if obj["schema_version"] == "1.2":
            continue
        img = IO.read_raw(cfg, K.image_path(cfg, obj["image"]))
        if K.sha256(K.image_path(cfg, obj["image"])) != obj["image_sha256"]:
            raise SystemExit(f"{obj['image']}: the local file differs from the one annotated; run on the owner's machine")
        new = upgrade_file(obj, ids, IO.pixel_sha256(img), GROUP_OWNER[K.annotator_group(cfg, obj["set"])], obj["image"] in reflective)
        e2 = SC.validate_file(new)
        if e2:
            raise SystemExit(f"{p.name}: {e2[:5]}")
        if not a.dry_run:
            MG.atomic_write_json(p, new)
        n += 1
    if changed and not a.dry_run:
        AN.save(ap_, annot)
    print(f"{'would upgrade' if a.dry_run else 'upgraded'} {n} files; annotators: {[(e['id'], e['name']) for e in annot['annotators']]}; "
          f"reflective_advisory on {sorted(reflective)}")


if __name__ == "__main__":
    main()
