"""TASK_GEOM_INTEGRATION G1: disagreement audit between tri_veto4 and geom_v0 on batch1 + batch2 (20 images).

Maps (network resolution, upright): tri-only = tri_veto4 w < 0.5 and geom_v0 w >= 0.9; geom-only = the reverse.
Components: 8-connected after merging those closer than 2 px (labels of the 1-px dilation, assigned back to the
original pixels); components under 10 px dropped.
Strata h = image x direction, A_h = the full (unsampled) stratum area, N_h its component count. Sample: 140 tri-only and
60 geom-only draws, allocated to the strata of a direction in proportion to A_h (at least 1 where N_h > 0; largest
remainders), drawn within a stratum with probability proportional to area **with replacement** (so that the
unweighted fraction of a stratum's draws labelled c estimates the stratum's area share of c; G4 estimator
share_c = sum_h A_h p_hc / sum_h A_h). A component drawn twice is labelled once; audit_draws.csv keeps every draw.
Seed 0.

Writes baseline_output/diag/geom/audit/:
  audit_labels.csv   id, stem, direction, area_px, bbox (u0 v0 u1 v1, upright network px), label (empty), note
  audit_draws.csv    draw, stem, direction, id                 (one row per draw; ids repeat for repeated draws)
  audit_strata.csv   batch, stem, direction, A_h, N_h, n_h (draws), n_unique
  audit_totals.json  full totals per direction and per image (count, area), and the parameters
  audit_masks.npz    per labelled id: flat indices into the upright network image, plus its shape
  components_<stem>.npz  the full component label maps (tri_only, geom_only) for re-scoring
  sheet_XX.png       contact sheets: per component RGB | log Z | tri_veto4 w | geom_v0 w, component outlined

Variant (--split-px N --tile T --out DIR): components larger than N px are split into the pieces that fall in each
T x T tile of the image (pieces >= 10 px), and those pieces are the sampling units (each labelled on its own). The
strata, the draws and the estimator are unchanged; the units partition the component area.

Run from geometry/dedicated:  env -u PYTHONPATH uv run python scripts/geom_audit.py [--split-px N --tile T --out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.stage_b import image_stems, load_upright  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
OUT = ROOT / "baseline_output" / "diag" / "geom" / "audit"
TRI, GEOM = "tri_veto4", "geom_v0"
N_DRAWS = {"tri-only": 140, "geom-only": 60}
MIN_PX, SEED, PER_SHEET, PANEL = 10, 0, 12, 160
ALLOWED = "step, limb, fold, crease, texture, artifact, clutter, unsure"


def components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Labels (0 = none) of the components of mask after merging those closer than 2 px; < MIN_PX dropped."""
    eight = np.ones((3, 3), dtype=bool)
    dil = ndimage.binary_dilation(mask, structure=eight)
    lab, _ = ndimage.label(dil, structure=eight)
    lab = np.where(mask, lab, 0)
    areas = np.bincount(lab.ravel())
    keep = np.nonzero(areas >= MIN_PX)[0]
    keep = keep[keep > 0]
    remap = np.zeros(areas.size, dtype=np.int32)
    remap[keep] = np.arange(1, keep.size + 1)
    return remap[lab], int(keep.size)


def split_units(lab: np.ndarray, split_px: int, tile: int) -> tuple[np.ndarray, int]:
    """Relabel: components larger than split_px become their pieces in each tile x tile cell (pieces >= MIN_PX)."""
    if split_px <= 0:
        return lab, int(lab.max())
    areas = np.bincount(lab.ravel())
    H, W = lab.shape
    vv, uu = np.mgrid[0:H, 0:W]
    cell = (vv // tile) * ((W + tile - 1) // tile) + uu // tile
    big = (lab > 0) & (areas[lab] > split_px)
    key = np.where(big, -(cell + 1).astype(np.int64) * (lab.max() + 1) - lab, lab)  # unique per (component, cell)
    uniq, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(lab.shape)
    out = np.where(lab > 0, inv + 1 - (uniq[0] == 0 if uniq.size else 0), 0)
    out = np.where(lab > 0, out, 0)
    a2 = np.bincount(out.ravel())
    keep = np.nonzero(a2 >= MIN_PX)[0]
    keep = keep[keep > 0]
    remap = np.zeros(a2.size, dtype=np.int32)
    remap[keep] = np.arange(1, keep.size + 1)
    return remap[out], int(keep.size)


def allocate(areas: np.ndarray, total: int) -> np.ndarray:
    """Draws per stratum: proportional to area, at least 1 where area > 0, largest remainders; sums to total."""
    have = areas > 0
    n = np.zeros(areas.size, dtype=int)
    n[have] = 1
    rest = total - n.sum()
    if rest > 0 and areas.sum() > 0:
        exact = rest * areas / areas.sum()
        n += np.floor(exact).astype(int)
        left = total - n.sum()
        order = np.argsort(-(exact - np.floor(exact)))
        n[order[:left]] += 1
    return n


def load_pair(batch: str, stem: str):
    root = BATCHES[batch]
    cg = load_config(root / "stage_b" / GEOM / "config_resolved.yaml")
    ct = load_config(root / "stage_b" / TRI / "config_resolved.yaml")
    g, _ = load_upright(cg, root / "stage_b" / GEOM, stem)
    t, _ = load_upright(ct, root / "stage_b" / TRI, stem)
    sa = json.loads((root / "stage_a" / f"{stem}.json").read_text())
    W, H = sa["network"]["W_n"], sa["network"]["H_n"]
    rgb = cv2.resize(cv2.imread(str(root / "stage_a" / "srgb" / f"{stem}.png")), (W, H), interpolation=cv2.INTER_AREA)
    return t["w_net"].astype(np.float64), g["w_net"].astype(np.float64), g["D"].astype(np.float64), \
        g["sky_mask"].astype(bool), rgb


def panel(img: np.ndarray, crop: tuple[int, int, int, int], mask: np.ndarray, color) -> np.ndarray:
    u0, u1, v0, v1 = crop
    c = img[v0:v1, u0:u1]
    m = mask[v0:v1, u0:u1].astype(np.uint8)
    s = PANEL / max(c.shape[0], c.shape[1])
    size = (max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s)))
    big = cv2.resize(c, size, interpolation=cv2.INTER_NEAREST)
    mb = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(big, cnts, -1, color, 1)
    out = np.full((PANEL, PANEL, 3), 40, np.uint8)
    out[:big.shape[0], :big.shape[1]] = big
    return out


def gray(w: np.ndarray) -> np.ndarray:
    g = (np.clip((w - 0.05) / 0.95, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def logz(D: np.ndarray, sky: np.ndarray, crop) -> np.ndarray:
    u0, u1, v0, v1 = crop
    lz = np.log(np.maximum(D, 1e-6))
    win = lz[v0:v1, u0:u1][~sky[v0:v1, u0:u1]]
    lo, hi = (np.percentile(win, [2, 98]) if win.size else (lz.min(), lz.max()))
    g = (np.clip((lz - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    out = cv2.applyColorMap(g, cv2.COLORMAP_VIRIDIS)
    out[sky] = (128, 128, 128)
    return out


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-px", type=int, default=0, help="split components larger than this into tile pieces")
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    comps, strata, data = {}, [], {}
    totals = {"per_image": [], "per_direction": {}}
    for batch, root in BATCHES.items():
        for stem in image_stems(root / "stage_b" / GEOM):
            tw, gw, D, sky, rgb = load_pair(batch, stem)
            data[stem] = (tw, gw, D, sky, rgb, batch)
            maps = {}
            for direction, mask in (("tri-only", (tw < 0.5) & (gw >= 0.9)), ("geom-only", (gw < 0.5) & (tw >= 0.9))):
                lab, n = components(mask)
                maps[direction.replace("-", "_")] = lab
                if args.split_px > 0:
                    lab, n = split_units(lab, args.split_px, args.tile)
                    maps[direction.replace("-", "_") + "_units"] = lab
                areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
                comps[(stem, direction)] = (lab, areas)
                strata.append({"batch": batch, "stem": stem, "direction": direction, "A_h": int(areas.sum()),
                               "N_h": int(n)})
                totals["per_image"].append({"batch": batch, "stem": stem, "direction": direction, "count": int(n),
                                            "area_px": int(areas.sum()), "image_px": int(lab.size),
                                            "raw_px": int(mask.sum())})
            np.savez_compressed(OUT / f"components_{stem}.npz", **maps)
    st = pd.DataFrame(strata)
    st["n_h"] = 0
    for direction, total in N_DRAWS.items():
        sel = st["direction"] == direction
        st.loc[sel, "n_h"] = allocate(st.loc[sel, "A_h"].to_numpy(float), total)
    draws, uniq = [], {}
    for i, row in st.iterrows():
        lab, areas = comps[(row.stem, row.direction)]
        if row.n_h == 0:
            continue
        picks = rng.choice(np.arange(1, areas.size + 1), size=int(row.n_h), replace=True, p=areas / areas.sum())
        for cl in picks:
            draws.append({"stem": row.stem, "direction": row.direction, "comp": int(cl)})
    # ids: tri-only first, then by image and area (descending)
    keys = sorted({(d["direction"] != "tri-only", d["stem"], d["comp"]) for d in draws},
                  key=lambda k: (k[0], k[1], -comps[(k[1], "geom-only" if k[0] else "tri-only")][1][k[2] - 1]))
    for j, k in enumerate(keys, start=1):
        uniq[k] = j
    dr = pd.DataFrame([{"draw": i + 1, "stem": d["stem"], "direction": d["direction"],
                        "id": uniq[(d["direction"] != "tri-only", d["stem"], d["comp"])]} for i, d in enumerate(draws)])
    st["n_unique"] = [dr[(dr.stem == r.stem) & (dr.direction == r.direction)]["id"].nunique() for r in st.itertuples()]
    rows, masks = [], {}
    for (is_geom, stem, cl), j in sorted(uniq.items(), key=lambda kv: kv[1]):
        direction = "geom-only" if is_geom else "tri-only"
        lab, areas = comps[(stem, direction)]
        m = lab == cl
        vv, uu = np.nonzero(m)
        rows.append({"id": j, "stem": stem, "direction": direction, "area_px": int(areas[cl - 1]),
                     "bbox": f"{uu.min()} {vv.min()} {uu.max()} {vv.max()}", "label": "", "note": ""})
        masks[f"id{j:03d}"] = np.flatnonzero(m).astype(np.int32)
        masks[f"id{j:03d}_shape"] = np.array(m.shape)
    lab_df = pd.DataFrame(rows)
    lab_df.to_csv(OUT / "audit_labels.csv", index=False)
    dr.to_csv(OUT / "audit_draws.csv", index=False)
    st.to_csv(OUT / "audit_strata.csv", index=False)
    np.savez_compressed(OUT / "audit_masks.npz", **masks)
    tot = pd.DataFrame(totals["per_image"])
    for direction, g in tot.groupby("direction"):
        totals["per_direction"][direction] = {"count": int(g["count"].sum()), "area_px": int(g["area_px"].sum()),
                                              "share_of_image_px": float(g["area_px"].sum() / g["image_px"].sum())}
    totals["params"] = {"tri_run": TRI, "geom_run": GEOM, "min_px": MIN_PX, "merge_px": 2, "seed": SEED,
                        "draws": N_DRAWS, "sampling": "pps with replacement within image x direction strata",
                        "split_px": args.split_px, "tile": args.tile if args.split_px > 0 else None,
                        "allowed_labels": ALLOWED, "n_unique_labelled": int(len(lab_df))}
    (OUT / "audit_totals.json").write_text(json.dumps(totals, indent=1))

    # contact sheets
    for f0 in range(0, len(lab_df), PER_SHEET):
        block = lab_df.iloc[f0:f0 + PER_SHEET]
        tiles = []
        for r in block.itertuples():
            tw, gw, D, sky, rgb, batch = data[r.stem]
            H, W = tw.shape
            m = np.zeros(H * W, bool)
            m[masks[f"id{r.id:03d}"]] = True
            m = m.reshape(H, W)
            u0, v0, u1, v1 = map(int, r.bbox.split())
            half = max(24, int(0.75 * max(u1 - u0 + 1, v1 - v0 + 1)))
            cu, cv = (u0 + u1) // 2, (v0 + v1) // 2
            crop = (max(0, cu - half), min(W, cu + half + 1), max(0, cv - half), min(H, cv + half + 1))
            col = (0, 0, 255) if r.direction == "tri-only" else (255, 255, 0)
            pans = [panel(rgb, crop, m, col), panel(logz(D, sky, crop), crop, m, col),
                    panel(gray(tw), crop, m, (0, 0, 255)), panel(gray(gw), crop, m, (0, 0, 255))]
            row = np.concatenate([np.concatenate([p, np.full((PANEL, 4, 3), 255, np.uint8)], 1) for p in pans], 1)
            head = np.full((18, row.shape[1], 3), 255, np.uint8)
            cv2.putText(head, f"id {r.id:03d}  {r.stem}  {r.direction}  {r.area_px} px  [{r.bbox}]", (4, 13), 0, 0.42,
                        (0, 0, 0), 1)
            tiles.append(np.concatenate([head, row, np.full((6, row.shape[1], 3), 255, np.uint8)], 0))
        half_n = (len(tiles) + 1) // 2
        cols = [np.concatenate(tiles[:half_n], 0), np.concatenate(tiles[half_n:], 0) if len(tiles) > half_n else None]
        if cols[1] is not None and cols[1].shape[0] < cols[0].shape[0]:
            cols[1] = np.concatenate([cols[1], np.full((cols[0].shape[0] - cols[1].shape[0], cols[1].shape[1], 3), 255,
                                                       np.uint8)], 0)
        sheet = cols[0] if cols[1] is None else np.concatenate([cols[0], np.full((cols[0].shape[0], 12, 3), 255,
                                                                                  np.uint8), cols[1]], 1)
        legend = np.full((20, sheet.shape[1], 3), 255, np.uint8)
        cv2.putText(legend, "panels: RGB | log Z | tri_veto4 w | geom_v0 w  (component outlined; red tri-only, cyan "
                            "geom-only on RGB / log Z)", (4, 14), 0, 0.45, (0, 0, 0), 1)
        cv2.imwrite(str(OUT / f"sheet_{f0 // PER_SHEET + 1:02d}.png"), np.concatenate([legend, sheet], 0))

    pd.set_option("display.width", 220)
    print(tot.pivot_table(index=["batch", "stem"], columns="direction", values=["count", "area_px"]).to_string())
    print("\nper direction:", json.dumps(totals["per_direction"], indent=1))
    print(f"\nsample: {len(dr)} draws, {len(lab_df)} unique components "
          f"({(lab_df.direction == 'tri-only').sum()} tri-only, {(lab_df.direction == 'geom-only').sum()} geom-only); "
          f"{(len(lab_df) + PER_SHEET - 1) // PER_SHEET} sheets -> {OUT}")


if __name__ == "__main__":
    main()
