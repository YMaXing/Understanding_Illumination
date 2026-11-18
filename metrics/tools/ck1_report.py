#!/usr/bin/env python3
"""Checkpoint 1 read-only reports (owner rulings B1, C1, C3). Changes no convention; writes only under --out.

    metrics/run.sh tools/ck1_report.py [--out metrics/runs/ck1]

1. Black-level pedestal, per annotated image: per-channel minimum and 0.1th percentile (raw DN), zero-pixel count,
   and every TIFF tag (looking for BlackLevel / WhiteLevel / DNG / EXIF / maker metadata).
2. Per-image pixel counts with max channel in [64224, 65000) (valid at sat 65000, invalid under §2's 2% rule),
   and >= 65000 for context.
3. Patch QC per set and type, square vs disc (radius = the annotated half-size per end): invalid patches (< 70% valid),
   and low-SNR pairs (§2 noise guard on ℓI, tau = 0.10) among pairs with both patches valid.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config as K  # noqa: E402
from core import io as IO  # noqa: E402
from core import patches as PT  # noqa: E402
import migrate_v0 as MG  # noqa: E402


def tiff_meta(path):
    with tifffile.TiffFile(path) as t:
        pg = t.pages[0]
        tags = {}
        for tag in pg.tags.values():
            v = tag.value
            if isinstance(v, (bytes, bytearray)):
                v = f"<{len(v)} bytes>"
            elif isinstance(v, (tuple, list)) and len(v) > 8:
                v = f"<{len(v)} values>"
            elif hasattr(v, "name"):
                v = v.name
            tags[tag.name] = v if isinstance(v, (int, float, str)) else str(v)
        return dict(tags=tags, is_dng=bool(getattr(t, "is_dng", False)), n_pages=len(t.pages),
                    extra=[k for k in ("is_imagej", "is_ome", "is_micromanager") if getattr(t, k, False)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(K.METRICS / "runs" / "ck1"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = K.load_config()
    files = MG.read_files(cfg)
    tau, div = cfg["thresholds"]["tau"], cfg["thresholds"]["snr_divisor"]
    lo, sat = cfg["validity"]["sat_report_lo_dn"], cfg["validity"]["sat_dn"]
    img_rows, tag_union = [], Counter()
    qc = defaultdict(lambda: defaultdict(Counter))      # shape -> (set, type) -> counts
    for stem in K.eval_stems(cfg):
        f = files[stem]
        img = IO.load_image(cfg, stem)
        I = img["I"]
        meta = tiff_meta(img["path"])
        tag_union.update(meta["tags"].keys())
        mx = I.max(-1)
        img_rows.append(dict(set=f["set"], image=stem, H=img["H"], W=img["W"],
                             min=[int(I[..., c].min()) for c in range(3)],
                             p0_1=[float(np.percentile(I[..., c], 0.1)) for c in range(3)],
                             p1=[float(np.percentile(I[..., c], 1)) for c in range(3)],
                             zero_px=int(np.any(I == 0, -1).sum()),
                             band_64224_65000=int(((mx >= lo) & (mx < sat)).sum()), ge_65000=int((mx >= sat).sum()),
                             n_px=int(mx.size), tags=meta["tags"], is_dng=meta["is_dng"], n_pages=meta["n_pages"]))
        for shape in ("square", "disc"):
            P = dict(cfg["patch"], shape=shape)
            for p in f["pairs"]:
                k = (f["set"], p["type"])
                sa = PT.patch_stats(img["lI"], img["valid"], p["A"]["x"], p["A"]["y"], p["A"]["r"], P)
                sb = PT.patch_stats(img["lI"], img["valid"], p["B"]["x"], p["B"]["y"], p["B"]["r"], P)
                c = qc[shape][k]
                c["pairs"] += 1
                if not (sa["ok"] and sb["ok"]):
                    c["invalid"] += 1
                    continue
                low, _ = PT.low_snr(sa, sb, tau, div)
                c["low_snr"] += low
                c["n_px_sum"] += sa["n_valid"] + sb["n_valid"]
        print(f"{f['set']:15s} {stem}", flush=True)
    (out / "ck1.json").write_text(json.dumps(dict(images=img_rows, tag_union=dict(tag_union),
                                                  qc={s: {f"{k[0]}/{k[1]}": dict(v) for k, v in d.items()} for s, d in qc.items()}),
                                             indent=1))
    L = ["# Checkpoint 1 read-only reports", ""]
    L += ["## 1. Black-level pedestal (raw DN; R, G, B)", "",
          f"TIFF tags seen on any of the {len(img_rows)} images: " + ", ".join(f"{k} ({n})" for k, n in sorted(tag_union.items())), "",
          "| set | image | size | min | 0.1th pct | 1st pct | px with a 0 channel |", "|---|---|---|---|---|---|---|"]
    for r in img_rows:
        L.append(f"| {r['set']} | {r['image']} | {r['H']}×{r['W']} | {'/'.join(map(str, r['min']))} | "
                 f"{'/'.join(f'{x:g}' for x in r['p0_1'])} | {'/'.join(f'{x:g}' for x in r['p1'])} | {r['zero_px']} |")
    L += ["", f"## 2. Pixels with max channel in [{lo}, {sat}) (valid now; invalid under §2's 2% rule)", "",
          "| set | image | in band | ≥ 65000 | band % |", "|---|---|---|---|---|"]
    for r in img_rows:
        L.append(f"| {r['set']} | {r['image']} | {r['band_64224_65000']} | {r['ge_65000']} | {100 * r['band_64224_65000'] / r['n_px']:.3f} |")
    L += ["", f"## 3. Patch QC, square vs disc (r = annotated half-size per end; tau {tau}, low-SNR if > tau/{div})", "",
          "| set | type | pairs | invalid sq | invalid disc | low-SNR sq | low-SNR disc | mean valid px/patch sq | disc |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(qc["square"]):
        s, d = qc["square"][k], qc["disc"][k]
        mp = lambda c: c["n_px_sum"] / max(2 * (c["pairs"] - c["invalid"]), 1)
        L.append(f"| {k[0]} | {k[1]} | {s['pairs']} | {s['invalid']} | {d['invalid']} | {s['low_snr']} | {d['low_snr']} | "
                 f"{mp(s):.1f} | {mp(d):.1f} |")
    (out / "ck1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
