#!/usr/bin/env python3
"""Check that metrics/ and the annotation tool can run on this machine; print what is missing and how to fix it.

    metrics/run.sh doctor.py [--quick]          # --quick: skip hashing the 100 TIFFs

Required (exit code 1 if any fails): Python 3.12 + pinned packages, the dataset folder, every annotation file and its
TIFF with a matching SHA-256, the ℓI loader hook, the annotation cache (prepare.py). Optional (reported only): overlay
maps (local detector maps or an imported archive; without them overlays are n/a and targeted s pairs are disabled),
dense-diagnostic maps, the parity CSV, Chrome + PowerShell for the browser self-test.
"""

import argparse
import importlib.metadata as md
import json
import shutil
import sys
from pathlib import Path

M = Path(__file__).resolve().parent
sys.path.insert(0, str(M))

RESULTS = []


def say(level, what, detail="", fix=""):
    RESULTS.append(level)
    mark = {"ok": "ok     ", "fail": "MISSING", "warn": "note   "}[level]
    print(f"{mark} {what}" + (f": {detail}" if detail else ""))
    if fix and level != "ok":
        print(f"        fix: {fix}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    print("== environment")
    v = sys.version_info
    say("ok" if (v.major, v.minor) == (3, 12) else "fail", "Python 3.12", f"{v.major}.{v.minor}.{v.micro}",
        "cd metrics && env -u PYTHONPATH uv sync   (uv installs 3.12 from metrics/.python-version)")
    pins = {"numpy": "2.5.4", "scipy": "1.18.1", "opencv-python-headless": "5.0.0.93", "tifffile": "2026.10.15",
            "imagecodecs": "2026.10.10", "pyyaml": "6.0.3"}
    for pkg, want in pins.items():
        try:
            got = md.version(pkg)
            say("ok" if got == want else "warn", f"{pkg} {want}", got if got != want else "", "cd metrics && uv sync")
        except md.PackageNotFoundError:
            say("fail", f"{pkg} {want}", "not installed", "cd metrics && env -u PYTHONPATH uv sync")
    try:
        from core import config as K
        cfg = K.load_config()
    except Exception as e:                      # noqa: BLE001
        say("fail", "config", str(e), "check metrics/config.yaml and metrics/config.local.yaml")
        return 1
    print("== config")
    say("ok" if cfg["_local"] else "warn", "metrics/config.local.yaml", cfg["_local"] or "absent (using defaults)",
        "cp metrics/config.local.example.yaml metrics/config.local.yaml, then set dataset.val_set")
    ds = K.repo_path(cfg["dataset"]["val_set"])
    n_tif = len(list(ds.glob("*.tif"))) if ds.is_dir() else 0
    say("ok" if n_tif else "fail", "dataset.val_set", f"{ds} ({n_tif} TIFFs)",
        "set dataset.val_set in metrics/config.local.yaml to the folder holding <stem>.tif for every image")
    print("== loader hook (ℓI)")
    try:
        K.import_hook(cfg["loader"]["log_i"], cfg["loader"].get("optional_imports"))
        K.import_hook(cfg["noise"]["model"], cfg["loader"].get("optional_imports"))
        say("ok", "illumination_edge/common.py hooks import")
        if K.STUBBED:
            say("warn", "optional import replaced by an inert placeholder", ", ".join(K.STUBBED) +
                " (log_chromaticity/ is git-ignored; the hooks do not use it)")
    except Exception as e:                      # noqa: BLE001
        say("fail", "loader hook", f"{type(e).__name__}: {e}", "the clone must contain illumination_edge/common.py")
    print("== images and annotation files")
    ann = K.repo_path(cfg["paths"]["annotations"])
    stems = K.all_stems(cfg)
    miss_ann, miss_tif, bad_hash, meta_only, splits = [], [], [], [], {}
    for s in stems:
        p = ann / f"{s}.json"
        if not p.exists():
            miss_ann.append(s)
            continue
        o = json.loads(p.read_text())
        splits[o.get("split")] = splits.get(o.get("split"), 0) + 1
        t = ds / f"{s}.tif"
        if not t.exists():
            miss_tif.append(s)
        elif not a.quick and K.sha256(t) != o["image_sha256"]:
            from core import io as IO
            if IO.pixel_sha256(IO.read_raw(cfg, t)) == o["image_pixel_sha256"]:
                meta_only.append(s)
            else:
                bad_hash.append(s)
    say("fail" if miss_ann else "ok", f"annotation files ({len(stems)} images)", f"missing {miss_ann[:5]}" if miss_ann else
        f"splits {splits}", "git pull (annotation files are tracked)")
    say("fail" if miss_tif else "ok", "TIFFs in dataset.val_set", f"missing {len(miss_tif)}: {miss_tif[:5]}" if miss_tif else "all present",
        "copy the missing images into the dataset folder")
    if not a.quick:
        say("fail" if bad_hash else "ok", "image pixels = the annotated image (pixel SHA-256)", f"{len(bad_hash)} differ: {bad_hash[:5]}" if bad_hash
            else "all match", "use the project's copy of the image set (different pixels = a different image)")
        if meta_only:
            say("warn", "file bytes differ but pixels match (metadata edited, e.g. a Windows Explorer rating)",
                ", ".join(meta_only), "accepted; never rate, tag or edit dataset images in Explorer or Photos")
    print("== annotation cache (prepare.py)")
    cache = K.repo_path(cfg["annotate"]["cache"])
    man = cache / "manifest.json"
    if not man.exists():
        say("fail", "cache manifest", str(man), "metrics/run.sh annotate/prepare.py")
    else:
        mimgs = {x["stem"]: x for x in json.loads(man.read_text())["images"]}
        need = ("raw.u16.gz", "lI.f32.gz", "valid.u8.gz", "meta.json")
        missing = [s for s in stems if s not in mimgs or not all((cache / s / f).exists() for f in need)]
        say("fail" if missing else "ok", "cache files per image", f"{len(missing)} images not prepared: {missing[:5]}" if missing
            else f"{len(mimgs)} images", "metrics/run.sh annotate/prepare.py")
        with_ovl = [s for s in stems if s in mimgs and mimgs[s]["overlays"]]
        imported = [s for s in stems if (cache / s / "imported" / "imported.json").exists()]
        say("ok" if len(with_ovl) == len(stems) else "warn", "overlays (illumination / material / geometric edges)",
            f"{len(with_ovl)} of {len(stems)} images ({len(imported)} from an imported archive)",
            "metrics/run.sh tools/overlays.py import <archive from the project lead>; then metrics/run.sh annotate/prepare.py "
            "(without overlays, targeted s pairs are disabled on that image)")
    print("== optional")
    for mp in ("geom_v1_1", "smoothness_weight", "illum_edges", "material_edges"):
        entries = cfg["maps"][mp]["dirs"] if mp == "geom_v1_1" else {k: v for k, v in cfg["maps"][mp].items() if isinstance(v, dict)}
        say("ok" if entries else "warn", f"detector maps: {mp}", f"sets {sorted(entries)}" if entries else "none configured",
            "only needed for geom_cross and dense diagnostics in run_metrics.py; see docs/SETUP.md")
    pc = cfg["parity"]["csv"]
    say("ok" if pc and K.repo_path(pc).exists() else "warn", "parity CSV", pc or "not configured", "only the parity test needs it (it skips)")
    T = cfg.get("tools", {})
    for k, std in (("chrome", "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"),
                   ("powershell", "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")):
        p = T.get(k) or (std if Path(std).exists() else shutil.which(k))
        say("ok" if p and Path(p).exists() else "warn", f"{k} (browser self-test)", p or "not found", f"set tools.{k} in config.local.yaml")
    nf = RESULTS.count("fail")
    print(f"\n{'READY' if not nf else 'NOT READY'}: {nf} required item(s) missing, {RESULTS.count('warn')} note(s).")
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
