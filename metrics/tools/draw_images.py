#!/usr/bin/env python3
"""Draw new images from val_set with a fixed seed (no hand-picking). Read-only apart from its outputs.

    metrics/run.sh tools/draw_images.py [--n 54] [--seed 20261118] [--name draw54]

Eligible = val_set TIFFs minus: every image already in config image_sets (all other sets), the glossy pool
(paths.glossy_pool), acharya_mrudula_068 (held-out protocol exclusion), visaria_keval_087 (dropped by the held-out
annotator as too hard), and always the stems at positions 17-109 of the held-out draw sequence
(illumination_edge/heldout_frozen/sample.json, seed 20261109): FRESH_PROTOCOL.md reserves 17-25 and counts 26-109 for
the fresh bundle test and forbids computing anything on them first (owner, Checkpoint 5: kept in every draw).
Draw: numpy default_rng(seed).permutation over the eligible stems sorted by name; the first n.
Writes metrics/splits/<name>.json (tracked; with SHA-256 per image) and metrics/runs/splits/<name>.png (contact sheet,
for an advisory look only: nothing is removed by looking).
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config as K  # noqa: E402

FRESH_SPAN = (17, 109)
FIXED = {"acharya_mrudula_068": "held-out protocol exclusion", "visaria_keval_087": "dropped by the held-out annotator (too hard)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=54)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--name", default="draw54")
    a = ap.parse_args()
    cfg = K.load_config()
    pool = K.repo_path(cfg["dataset"]["val_set"])
    stems = sorted(p.stem for p in pool.glob("*.tif"))
    annotated = {s for k, v in cfg["image_sets"].items() if k != a.name for s in v}
    glossy = K.glossy_pool(cfg)
    seq = json.loads((K.REPO / "illumination_edge" / "heldout_frozen" / "sample.json").read_text())["sequence"]
    fresh = {x["stem"] for x in seq if FRESH_SPAN[0] <= x["rank"] <= FRESH_SPAN[1]}
    excl = {}
    for s in stems:
        why = [w for w, c in (("already annotated", s in annotated), ("glossy pool", s in glossy),
                              (FIXED.get(s, ""), s in FIXED), ("fresh protocol span 17-109", s in fresh)) if c]
        if why:
            excl[s] = why
    elig = [s for s in stems if s not in excl]
    perm = np.random.default_rng(a.seed).permutation(len(elig))
    pick = [elig[i] for i in perm[: a.n]]
    rows, thumbs = [], []
    for k, s in enumerate(pick, 1):
        p = pool / f"{s}.tif"
        I = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[..., ::-1]
        ref = max(np.percentile(I, 99.5), 1)
        t = (np.clip(I / ref, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        H, W = t.shape[:2]
        sc = 240 / max(H, W)
        t = cv2.resize(t, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
        tile = np.full((266, 240, 3), 24, np.uint8)
        tile[:t.shape[0], (240 - t.shape[1]) // 2:(240 - t.shape[1]) // 2 + t.shape[1]] = t
        cv2.putText(tile, f"{k}. {s}", (3, 259), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1, cv2.LINE_AA)
        thumbs.append(tile[..., ::-1])
        rows.append(dict(rank=k, stem=s, H=H, W=W, sha256=K.sha256(p)))
    cols = 9
    while len(thumbs) % cols:
        thumbs.append(np.full_like(thumbs[0], 24))
    sheet = np.vstack([np.hstack(thumbs[r * cols:(r + 1) * cols]) for r in range(len(thumbs) // cols)])
    od = K.METRICS / "runs" / "splits"
    od.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(od / f"{a.name}.png"), sheet)
    out = dict(name=a.name, drawn_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), pool="dataset.val_set",
               pool_tiffs=len(stems), n_eligible=len(elig), seed=a.seed,
               method="numpy default_rng(seed).permutation over the eligible stems sorted by name; first n",
               excluded_counts={w: sum(w in v for v in excl.values()) for w in
                                ("already annotated", "glossy pool", "fresh protocol span 17-109", *FIXED.values())},
               images=rows)
    sd = K.METRICS / "splits"
    sd.mkdir(exist_ok=True)
    (sd / f"{a.name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "images"}, indent=1))
    print(" ".join(r["stem"] for r in rows))
    print(od / f"{a.name}.png")


if __name__ == "__main__":
    main()
