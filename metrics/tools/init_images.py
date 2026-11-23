#!/usr/bin/env python3
"""Create empty annotation files (current schema) for listed images that have none (e.g. the 54 drawn images).

    metrics/run.sh tools/init_images.py --assign ID [--images LIST]   # default: every config image_sets image without a file

The data split comes from paths.split_assignment (tools/assign_splits.py); assigned_to is --assign (an id in
annotators.json). image_pixel_sha256 is computed from the decoded image.

Never touches an existing file. Then run metrics/run.sh annotate/prepare.py --images LIST for the tool's cache.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images")
    ap.add_argument("--assign", required=True, help="annotator id the new images are assigned to")
    a = ap.parse_args()
    cfg = K.load_config()
    ann = K.repo_path(cfg["paths"]["annotations"])
    if a.assign not in AN.names(ann / "annotators.json"):
        raise SystemExit(f"--assign {a.assign!r}: not an id in annotators.json")
    stems = K.all_stems(cfg) if not a.images else [x.strip() for x in a.images.split(",")]
    split = {x["stem"]: x["split"] for x in json.loads(K.repo_path(cfg["paths"]["split_assignment"]).read_text())["images"]}
    for s in stems:
        p = ann / f"{s}.json"
        if p.exists():
            continue
        img = IO.load_image(cfg, s)
        obj = dict(schema_version=cfg["schema_version"], image=s, set=K.image_set(cfg, s), split=split[s],
                   assigned_to=a.assign, seen_in_training=cfg["seen_in_training_default"], width=img["W"], height=img["H"],
                   image_sha256=K.sha256(img["path"]), image_pixel_sha256=IO.pixel_sha256(img["I"]),
                   pairs=[], s_seed_log=[], legacy={}, deleted_pairs=[])
        errs = SC.validate_file(obj)
        if errs:
            raise SystemExit(f"{s}: {errs}")
        MG.atomic_write_json(p, obj)
        print(f"created {p}")


if __name__ == "__main__":
    main()
