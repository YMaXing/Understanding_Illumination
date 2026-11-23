#!/usr/bin/env python3
"""Build the annotation tool's cache (repo deps; the server itself is stdlib-only and reads only this cache).

    metrics/run.sh annotate/prepare.py [--images LIST] [--force]

Per image, in <cache>/<stem>/ (config annotate.cache, git-ignored):
  raw.u16.gz    the linear RGB from the loader hook (loader.read), uint16 little-endian, H x W x 3 (display only)
  lI.f32.gz     ℓI from the loader hook (loader.log_i), float32 H x W x 3 (patch medians, lit-end inference, auto-size)
  valid.u8.gz   the metrics validity mask (core/io.validity_mask), uint8 H x W
  illum.png / material.png / geom.png   RGBA overlays (default detector illumination edges, its material edges,
                geom_v1_1 w < 0.5), only where the map exists for the image's set (config maps.*); no region overlay
                exists (Delta B2)
  meta.json     size, set, split, sensor-noise model [(g, r2)] per channel (Delta Q1: the low-SNR warning), tone
                reference (99.5th percentile of I, as illumination_edge/common.tone), ℓI display range, image SHA-256
<cache>/manifest.json: images (in annotation order: val, test, then train; annotate.image_order), annotation paths,
and every constant the tool uses (validity, patch, thresholds, lengths, s-pool workflow, auto-sizer, targets). Nothing
about predictions is written: the tool never loads them.
Checks: every TIFF must match the image_sha256 in its annotation file (refuses on a mismatch: a different file with the
same name would silently misplace every pair). Overlays: a local detector map (config maps.*) wins; otherwise an
overlay imported from an archive (tools/overlays.py import -> <cache>/<stem>/imported/) is used; otherwise none
(the tool then shows the overlay as n/a and targeted s pairs are disabled).
"""

import argparse
import datetime as dt
import gzip
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config as K  # noqa: E402
from core import dense as DN  # noqa: E402
from core import io as IO  # noqa: E402

OVERLAYS = {"illum": ("illum_edges", (255, 150, 0)), "material": ("material_edges", (0, 220, 255))}
GEOM_RGBA = (60, 120, 255, 110)


def write_gz(path, arr):
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(np.ascontiguousarray(arr).tobytes())


def rgba_png(path, mask, rgb, alpha=255):
    out = np.zeros(mask.shape + (4,), np.uint8)
    out[mask] = (rgb[2], rgb[1], rgb[0], alpha)          # cv2 writes BGRA
    cv2.imwrite(str(path), out)


class HashMismatch(SystemExit):
    pass


def check_hash(cfg, stem):
    """Image identity (Checkpoint 6): the decoded pixels must match the annotation file's image_pixel_sha256. A file
    whose bytes differ but whose pixels match (metadata edited, e.g. rated in Windows Explorer) is accepted with a
    warning; different pixels are refused."""
    ann = json.loads((K.repo_path(cfg["paths"]["annotations"]) / f"{stem}.json").read_text())
    path = K.image_path(cfg, stem)
    if K.sha256(path) == ann["image_sha256"]:
        return path, ann
    px = IO.pixel_sha256(IO.read_raw(cfg, path))
    if px != ann["image_pixel_sha256"]:
        raise HashMismatch(f"{stem}: the pixels of {path} differ from the image the annotations were made on (pixel SHA-256 "
                           f"{px[:16]}... vs {ann['image_pixel_sha256'][:16]}...). Refusing: point dataset.val_set "
                           "(config.local.yaml) at the project's copy of the image set.")
    print(f"  warning {stem}: file bytes differ from the annotated copy, pixels identical (metadata edited?): accepted", flush=True)
    return path, ann


def prepare_image(cfg, stem, cache, force=False):
    d = cache / stem
    path, ann = check_hash(cfg, stem)
    if (d / "meta.json").exists() and not force:
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("image_pixel_sha256") == ann["image_pixel_sha256"] and meta.get("overlay_inputs") == overlay_inputs(cfg, stem, d):
            return meta
    d.mkdir(parents=True, exist_ok=True)
    set_name = K.image_set(cfg, stem)
    img = IO.load_image(cfg, stem, path)
    write_gz(d / "raw.u16.gz", img["I"].astype("<u2"))
    write_gz(d / "lI.f32.gz", img["lI"].astype("<f4"))
    write_gz(d / "valid.u8.gz", img["valid"].astype(np.uint8))
    overlays = []
    imported = json.loads((d / "imported" / "imported.json").read_text()) if (d / "imported" / "imported.json").exists() else {}
    for name in ("illum", "material", "geom"):
        (d / f"{name}.png").unlink(missing_ok=True)
        if name == "geom":
            ge = IO.geom_edge(cfg, stem, img["lI"].shape, set_name)
            if ge is not None:
                rgba_png(d / "geom.png", ge, GEOM_RGBA[:3], GEOM_RGBA[3])
                overlays.append(dict(name="geom", source=dict(dir=cfg["maps"]["geom_v1_1"]["dirs"][set_name], key="w < 0.5")))
                continue
        else:
            mapname, rgb = OVERLAYS[name]
            m, src = DN.load_map(cfg, mapname, stem, set_name, img["lI"].shape)
            if m is not None:
                rgba_png(d / f"{name}.png", m.astype(bool), rgb)
                overlays.append(dict(name=name, source=src))
                continue
        if name in imported.get("files", {}):
            shutil.copyfile(d / "imported" / f"{name}.png", d / f"{name}.png")
            overlays.append(dict(name=name, source=dict(imported=imported["archive"], sha256=imported["files"][name])))
    lv = img["lI"][img["valid"]]
    meta = dict(stem=stem, set=set_name, split=ann["split"], overlay_inputs=overlay_inputs(cfg, stem, d),
                H=img["H"], W=img["W"], noise=[list(p) for p in IO.noise_params(cfg, img)],
                tone_ref=float(max(np.percentile(img["I"].astype(np.float32), 99.5), 1.0)),
                lI_range=[float(np.percentile(lv, 0.5)), float(np.percentile(lv, 99.5))] if lv.size else [0.0, 11.1],
                image_sha256=ann["image_sha256"], image_pixel_sha256=ann["image_pixel_sha256"], overlays=overlays, regions=None,
                built_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def overlay_inputs(cfg, stem, d):
    """What the overlays were built from (so a changed local map or a new import rebuilds the image's cache)."""
    s = K.image_set(cfg, stem)
    imp = d / "imported" / "imported.json"
    return dict(maps={k: cfg["maps"][k].get(s) for k in ("illum_edges", "material_edges")},
                geom=cfg["maps"]["geom_v1_1"]["dirs"].get(s), imported=K.sha256(imp) if imp.exists() else None)


def _ann(cfg, stem):
    p = K.repo_path(cfg["paths"]["annotations"]) / f"{stem}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def manifest(cfg, metas, cache):
    A, P, T, PR = cfg["annotate"], cfg["patch"], cfg["thresholds"], cfg["pairs"]
    return dict(
        schema_version=cfg["schema_version"], built_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        annotations=str(K.repo_path(cfg["paths"]["annotations"])), backups=str(K.repo_path(A["backups"])),
        relabels=str(K.repo_path(A["relabels"])), targets=A["targets"],
        cache=str(cache), host=A["host"], port=A["port"], seed_base=str(A["seed_base"]),
        images=sorted([dict(stem=m["stem"], set=m["set"], group=K.group_of(cfg, m["set"]),
                            annotator_group=K.annotator_group(cfg, m["set"]), split=K.split_of(cfg, m["stem"]),
                            assigned_to=_ann(cfg, m["stem"]).get("assigned_to"),
                            reflective_advisory=bool(_ann(cfg, m["stem"]).get("reflective_advisory")), H=m["H"], W=m["W"],
                            overlays=[o["name"] for o in m["overlays"]]) for m in metas],
                      key=lambda x: (A["image_order"].index(x["split"]) if x["split"] in A["image_order"] else 9, x["stem"])),
        constants=dict(validity=cfg["validity"], patch=P, tau=T["tau"], snr_divisor=T["snr_divisor"],
                       near_types=PR["near_types"], distant_types=PR["distant_types"], near_max_px=PR["near_max_px"],
                       distant_min_px=PR["distant_min_px"], nudge_max_px=A["nudge_max_px"], s_partner_px=A["s_partner_px"],
                       autosize=A["autosize"], regions=None))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="comma list of stems or set names (default: every eval image)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = K.load_config(a.config)
    cache = K.repo_path(cfg["annotate"]["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    stems = K.all_stems(cfg)
    if a.images:
        want = []
        for x in a.images.split(","):
            want += cfg["image_sets"].get(x.strip(), [x.strip()])
        stems = [s for s in stems if s in want]
    metas = []
    for s in K.all_stems(cfg):
        if s in stems:
            metas.append(prepare_image(cfg, s, cache, a.force))
            print(f"{metas[-1]['set']:15s} {s:22s} overlays: {', '.join(o['name'] for o in metas[-1]['overlays']) or '-'}", flush=True)
        elif (cache / s / "meta.json").exists():
            metas.append(json.loads((cache / s / "meta.json").read_text()))
    (cache / "manifest.json").write_text(json.dumps(manifest(cfg, metas, cache), indent=1))
    print(f"{len(metas)} images in {cache / 'manifest.json'}")


if __name__ == "__main__":
    main()
