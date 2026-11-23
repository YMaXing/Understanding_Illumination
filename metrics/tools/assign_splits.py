#!/usr/bin/env python3
"""Assign the 100 images to train / val / test = 60 / 20 / 20 (owner, Checkpoint 5), stratified by annotator group
(A: the 46 migrated images; B: the 54 drawn ones) and by glossy-pool membership. Deterministic; no hand-picking.

    metrics/run.sh tools/assign_splits.py [--seed 20261118]

Allocation: per stratum, the ideal counts n * (0.6, 0.2, 0.2) are floored, then the remaining units go to the
(stratum, split) cells with the largest fractional parts, subject to each stratum's size and each split's global
total (60 / 20 / 20); ties break by stratum order, then split order. Within a stratum, the stems sorted by name are
permuted with numpy default_rng(seed + stratum index) and taken in order: train, val, test.
Writes metrics/splits/split100.json (tracked). tools/upgrade_schema_1_1.py and tools/init_images.py read it.
"""

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config as K  # noqa: E402

SPLITS = ("train", "val", "test")
FRAC = (0.6, 0.2, 0.2)


def allocate(sizes, totals):
    """Integer allocation with row sums = sizes and column sums = totals, closest to the proportional ideal."""
    S = len(sizes)
    ideal = [[n * f for f in FRAC] for n in sizes]
    cnt = [[math.floor(x) for x in row] for row in ideal]
    rrem = [sizes[s] - sum(cnt[s]) for s in range(S)]
    crem = [totals[k] - sum(cnt[s][k] for s in range(S)) for k in range(3)]
    cells = sorted(((ideal[s][k] - cnt[s][k], -s, -k, s, k) for s in range(S) for k in range(3)), reverse=True)
    for _, _, _, s, k in cells:
        if rrem[s] > 0 and crem[k] > 0:
            cnt[s][k] += 1
            rrem[s] -= 1
            crem[k] -= 1
    if any(rrem) or any(crem):
        raise SystemExit(f"allocation failed: rows {rrem}, columns {crem}")
    return cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20261118)
    a = ap.parse_args()
    cfg = K.load_config()
    glossy = K.glossy_pool(cfg)
    stems = K.all_stems(cfg)
    if len(stems) != 100 or len(set(stems)) != 100:
        raise SystemExit(f"expected 100 distinct images, found {len(set(stems))}")
    strata = {}
    for s in stems:
        st = K.image_set(cfg, s)
        key = (K.annotator_group(cfg, st), "glossy" if s in glossy else "non-glossy")
        strata.setdefault(key, []).append(s)
    keys = sorted(strata)
    n = len(stems)
    totals = [round(n * f) for f in FRAC]
    cnt = allocate([len(strata[k]) for k in keys], totals)
    assign, table = {}, []
    for si, k in enumerate(keys):
        st = sorted(strata[k])
        order = [st[i] for i in np.random.default_rng(a.seed + si).permutation(len(st))]
        c = cnt[si]
        for j, s in enumerate(order):
            assign[s] = SPLITS[0] if j < c[0] else (SPLITS[1] if j < c[0] + c[1] else SPLITS[2])
        table.append(dict(annotator_group=k[0], glossy=k[1], n=len(st), **dict(zip(SPLITS, c))))
    images = [dict(stem=s, split=assign[s], set=K.image_set(cfg, s), provenance_group=K.group_of(cfg, K.image_set(cfg, s)),
                   annotator_group=K.annotator_group(cfg, K.image_set(cfg, s)), glossy_pool=s in glossy) for s in sorted(stems)]
    out = dict(assigned_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), seed=a.seed, fractions=dict(zip(SPLITS, FRAC)),
               totals=dict(zip(SPLITS, totals)), method=__doc__.split("Allocation:")[1].split("Writes")[0].strip(),
               strata=table, images=images)
    p = K.repo_path(cfg["paths"]["split_assignment"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"{'annotator group':16s} {'glossy pool':12s} {'n':>4s} {'train':>6s} {'val':>5s} {'test':>5s}")
    for r in table:
        print(f"{r['annotator_group']:16s} {r['glossy']:12s} {r['n']:4d} {r['train']:6d} {r['val']:5d} {r['test']:5d}")
    print(f"{'total':16s} {'':12s} {n:4d} " + " ".join(f"{sum(r[k] for r in table):>{w}d}" for k, w in zip(SPLITS, (6, 5, 5))))
    by = {}
    for x in images:
        by.setdefault((x["provenance_group"], x["split"]), 0)
        by[(x["provenance_group"], x["split"])] += 1
    print("by provenance group:", {g: {sp: by.get((g, sp), 0) for sp in SPLITS} for g in cfg["aggregate"]["provenance_groups"]})
    print(p)


if __name__ == "__main__":
    main()
