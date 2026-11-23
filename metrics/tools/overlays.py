#!/usr/bin/env python3
"""Export / import the annotation tool's overlay maps, for annotators who do not run the detector pipelines.

    metrics/run.sh tools/overlays.py export --out overlays.tar.gz [--images LIST]   # owner: from the built cache
    metrics/run.sh tools/overlays.py import overlays.tar.gz                         # annotator: into the local cache
    metrics/run.sh annotate/prepare.py                                              # then: picks the imported overlays up

Archive: manifest.json + <stem>/<name>.png (name in illum, material, geom). The manifest lists, per image, the image's
SHA-256 and pixel SHA-256 (from its annotation file) and each PNG's SHA-256 and source. Import reads only the expected
members (no extraction of arbitrary paths), verifies every PNG hash, refuses an image whose annotation file has a
different pixel SHA-256, and writes <cache>/<stem>/imported/<name>.png + imported.json. Nothing goes into git.
"""

import argparse
import datetime as dt
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config as K  # noqa: E402

NAMES = ("illum", "material", "geom")
FORMAT = "metrics-overlay-archive/1"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def stems_of(cfg, spec):
    if not spec:
        return K.all_stems(cfg)
    out = []
    for x in spec.split(","):
        out += cfg["image_sets"].get(x.strip(), [x.strip()])
    return out


def export(cfg, out, spec):
    cache = K.repo_path(cfg["annotate"]["cache"])
    man = dict(format=FORMAT, created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), images={})
    blobs = {}
    for s in stems_of(cfg, spec):
        mp = cache / s / "meta.json"
        if not mp.exists():
            print(f"  skip {s}: not prepared")
            continue
        meta = json.loads(mp.read_text())
        files, sources = {}, {}
        for o in meta["overlays"]:
            if o["name"] not in NAMES or "imported" in o["source"]:
                continue                                   # export only overlays built from local maps
            b = (cache / s / f"{o['name']}.png").read_bytes()
            files[o["name"]], sources[o["name"]] = sha(b), o["source"]
            blobs[f"{s}/{o['name']}.png"] = b
        if files:
            man["images"][s] = dict(image_sha256=meta["image_sha256"], image_pixel_sha256=meta.get("image_pixel_sha256"),
                                    files=files, sources=sources)
    mb = json.dumps(man, indent=1).encode()
    out = Path(out)
    with tarfile.open(out, "w:gz") as t:
        for name, b in [("manifest.json", mb)] + sorted(blobs.items()):
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime = len(b), 0
            t.addfile(ti, io.BytesIO(b))
    print(f"wrote {out}: {len(man['images'])} images, {len(blobs)} overlays; archive SHA-256 {sha(out.read_bytes())}")


def do_import(cfg, archive):
    cache = K.repo_path(cfg["annotate"]["cache"])
    ann = K.repo_path(cfg["paths"]["annotations"])
    arc_sha = sha(Path(archive).read_bytes())
    with tarfile.open(archive, "r:gz") as t:
        man = json.loads(t.extractfile("manifest.json").read())
        if man.get("format") != FORMAT:
            raise SystemExit(f"not an overlay archive ({man.get('format')!r})")
        ok = bad = 0
        known = set(K.all_stems(cfg))
        for s, e in sorted(man["images"].items()):
            if s not in known:
                print(f"  skip {s}: not an image of this project")
                bad += 1
                continue
            af = json.loads((ann / f"{s}.json").read_text()) if (ann / f"{s}.json").exists() else {}
            local, theirs = af.get("image_pixel_sha256"), e.get("image_pixel_sha256") or e.get("image_sha256")
            if local is None or theirs != local:
                print(f"  REFUSED {s}: the archive was built for image (pixels) {str(theirs)[:12]}..., your annotation file says {str(local)[:12]}...")
                bad += 1
                continue
            got = {}
            for name, h in e["files"].items():
                if name not in NAMES:
                    raise SystemExit(f"{s}: unexpected overlay name {name!r}")
                b = t.extractfile(f"{s}/{name}.png").read()
                if sha(b) != h:
                    raise SystemExit(f"{s}/{name}.png: hash mismatch (archive damaged?)")
                got[name] = b
            d = cache / s / "imported"
            d.mkdir(parents=True, exist_ok=True)
            for name, b in got.items():
                (d / f"{name}.png").write_bytes(b)
            (d / "imported.json").write_text(json.dumps(dict(archive=f"{Path(archive).name} ({arc_sha[:16]})", image_pixel_sha256=local,
                                                             files=e["files"], sources=e.get("sources", {})), indent=1))
            ok += 1
    print(f"imported overlays for {ok} images ({bad} skipped or refused). Next: metrics/run.sh annotate/prepare.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    e.add_argument("--images")
    i = sub.add_parser("import")
    i.add_argument("archive")
    a = ap.parse_args()
    cfg = K.load_config()
    export(cfg, a.out, a.images) if a.cmd == "export" else do_import(cfg, a.archive)


if __name__ == "__main__":
    main()
