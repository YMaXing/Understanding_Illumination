"""TASK_GEOM_INTEGRATION G3: crease rescue sweep (item 9), Delta Z0 report (item 3), thin-ring clusters (item 8).

1. Crease rescue: per image the context is built once (geom_v0's configuration); the fold channel runs with
   crease_rescue off (checked: bit-identical to the saved geom_v0 fold_w_net) and laplacian_corroborated over
   theta_rescue in {6, 8, 10} deg x k_L in {2, 3, 4}; combined w = min(saved geom_v0 depth_w_net, new fold w).
   Targets (batch1): 076 facade ledge and 088 rock ridge coverage >= 0.5; every texture ROI share <= geom_v0 + 0.01;
   083 bollard interior unchanged. Rescued counts per image on all 20 images.
2. Delta Z0 in {0.10, 0.30, 1.00} m: the depth weight re-rasterized from the saved candidate tables (w_c =
   max(w_min, dZ0 / (dZ + dZ0)), the channel's bands), combined with the saved fold w; deliverable shares (original
   resolution, all 20 images) and the G1 per-class metrics (batch1). A report, not a switch.
3. Thin-ring clusters: 32-px tiles in which more than 50 % of the accepted depth candidates are thin (tiles with at
   least 5 accepted candidates), per image, with an overlay of the cluster tiles for classification.
Writes baseline_output/diag/geom/g3/{g3.json, clusters_<stem>.png}.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geom_eval import CLASSES, roi_list, roi_metrics  # noqa: E402
from geom_g2_coverage import raster  # noqa: E402
from illum_baseline.config import dense_fit_params, detector_params, load_config  # noqa: E402
from illum_baseline.geom.camera import Intrinsics  # noqa: E402
from illum_baseline.geom.context import GeometryContext  # noqa: E402
from illum_baseline.geom.fold import FoldChannel, FoldParams  # noqa: E402
from illum_baseline.outputs import upsample_weights  # noqa: E402
from illum_baseline.stage_b import image_stems, load_upright  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
RUN = "geom_v0"
OUT = ROOT / "baseline_output" / "diag" / "geom" / "g3"
THETAS, KLS = (6.0, 8.0, 10.0), (2.0, 3.0, 4.0)
DZ0S = (0.10, 0.30, 1.00)
TILE, MIN_ACC = 32, 5
TEXTURE = ("070 floor", "079 grass", "079 grass sunlit", "083 bollard interior")


def class_metrics(wmaps: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """wmaps: {stem3: upright w}. Returns ({roi: metrics}, {class: metrics}) as in scripts/geom_eval.py."""
    per_roi, pooled = {}, {}
    for roi in roi_list():
        if roi["stem"] not in wmaps:
            continue
        m = roi_metrics(wmaps[roi["stem"]], roi)
        per_roi[roi["name"]] = {k: v for k, v in m.items() if not k.startswith("_")}
        if roi.get("legacy"):
            continue
        p = pooled.setdefault(roi["class"], {"v": [], "c": []})
        p["v"].append(m["_vals"])
        if m["_cov"] is not None:
            p["c"].append(m["_cov"])
    cls = {}
    for c in CLASSES:
        if c in pooled:
            v = np.concatenate(pooled[c]["v"])
            cv = np.concatenate(pooled[c]["c"]) if pooled[c]["c"] else None
            cls[c] = {"share_w<0.5": float((v < 0.5).mean()), "median_w": float(np.median(v)),
                      "coverage": float(cv.mean()) if cv is not None else None}
    return per_roi, cls


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rescue_settings = [None] + list(itertools.product(THETAS, KLS))
    fold_maps = {s: {} for s in rescue_settings}
    rescued_counts, off_identical, t_rescue = [], [], []
    dz_maps, dz_shares, clusters = {d: {} for d in DZ0S}, {d: [] for d in DZ0S}, []
    for batch, root in BATCHES.items():
        rd = root / "stage_b" / RUN
        cfg = load_config(rd / "config_resolved.yaml")
        base_fp = FoldParams.from_dict(dataclasses.asdict(cfg.geometry.fold))
        for stem in image_stems(rd):
            a, meta = load_upright(cfg, rd, stem)
            sa = root / "stage_a"
            meta_a = json.loads((sa / f"{stem}.json").read_text())
            z = np.load(sa / f"{stem}.npz")
            ctx = GeometryContext(z["D"], z["sky_prob"].astype(np.float64), Intrinsics.from_stage_a(meta_a),
                                  detector_params(cfg), dense_fit_params(cfg))
            wd = a["depth_w_net"].astype(np.float64)
            key = stem[-3:] if batch == "batch1" else stem
            row = {"batch": batch, "stem": stem}
            for sett in rescue_settings:
                fp = base_fp if sett is None else dataclasses.replace(base_fp, crease_rescue="laplacian_corroborated",
                                                                     rescue_theta_deg=sett[0], rescue_k_L=sett[1])
                t0 = time.perf_counter()
                res = FoldChannel().run(ctx, fp)
                if sett is None:
                    off_identical.append(bool(np.array_equal(res.w_net.astype(np.float16),
                                                             a["fold_w_net"].astype(np.float16))))
                else:
                    t_rescue.append(time.perf_counter() - t0)
                    row[f"rescued {sett[0]:g}/{sett[1]:g}"] = res.scalars["n_rescued"]
                    if sett == (8.0, 3.0):
                        row["eligible 8/3"] = res.scalars["n_rescue_eligible"]
                        row["veto-pass 8/3"] = res.scalars["n_rescue_veto_pass"]
                fold_maps[sett][key] = np.minimum(wd, res.w_net)
            rescued_counts.append(row)

            # Delta Z0: re-rasterize the depth weight
            dc = pd.read_parquet(rd / f"{stem}_depth_candidates.parquet")
            d = dc[dc["accepted"]]
            band = np.maximum(int(cfg.geometry.depth.band_min), np.ceil(d["sigma_wc"].to_numpy()).astype(int) + 1)
            up_wh = tuple(meta["geometry"]["upright_WxH"])
            for dz0 in DZ0S:
                dZ = d["dZ"].to_numpy()
                with np.errstate(invalid="ignore", divide="ignore"):
                    wc = np.maximum(0.05, dz0 / (dZ + dz0))
                wc = np.where(np.isinf(dZ), 0.05, wc)
                w = np.minimum(raster(wd.shape, d["u"].to_numpy(), d["v"].to_numpy(), wc, band),
                               a["fold_w_net"].astype(np.float64))
                if dz0 == 0.10:
                    assert np.array_equal(w.astype(np.float16), a["w_net"].astype(np.float16)), stem
                dz_maps[dz0][key] = w
                wf = upsample_weights(w, up_wh, 0.05)
                dz_shares[dz0].append({"stem": stem, "w<0.5": float((wf < 0.5).mean()),
                                       "w<0.9": float((wf < 0.9).mean())})

            # thin-ring clusters
            H, W = wd.shape
            ti, tj = (d["v0"].to_numpy() // TILE).astype(int), (d["u0"].to_numpy() // TILE).astype(int)
            tab = pd.DataFrame({"ti": ti, "tj": tj, "thin": d["thin"].to_numpy()})
            g = tab.groupby(["ti", "tj"]).agg(n=("thin", "size"), thin=("thin", "mean")).reset_index()
            cl = g[(g["n"] >= MIN_ACC) & (g["thin"] > 0.5)]
            clusters.append({"batch": batch, "stem": stem, "tiles_with_accepted": int((g["n"] >= MIN_ACC).sum()),
                             "cluster_tiles": int(len(cl)),
                             "thin_in_clusters": int(round((cl["n"] * cl["thin"]).sum())),
                             "accepted_thin_total": int(d["thin"].sum()),
                             "tiles": [[int(r.tj * TILE), int(r.ti * TILE)] for r in cl.itertuples()]})
            if len(cl):
                img = cv2.resize(cv2.imread(str(sa / "srgb" / f"{stem}.png")), (W, H), interpolation=cv2.INTER_AREA)
                for r in cl.itertuples():
                    cv2.rectangle(img, (r.tj * TILE, r.ti * TILE), (r.tj * TILE + TILE - 1, r.ti * TILE + TILE - 1),
                                  (0, 0, 255), 1)
                for r in d[d["thin"]].itertuples():
                    cv2.circle(img, (int(r.u0), int(r.v0)), 0, (0, 255, 255), -1)
                cv2.imwrite(str(OUT / f"clusters_{stem}.png"), cv2.resize(img, (2 * W, 2 * H),
                                                                             interpolation=cv2.INTER_NEAREST))
            print(stem, "done", flush=True)

    # rescue evaluation (batch1)
    b1 = lambda maps: {k: v for k, v in maps.items() if len(k) == 3}  # noqa: E731
    base_roi, base_cls = class_metrics(b1(fold_maps[None]))
    sweep = []
    for sett in rescue_settings[1:]:
        roi, cls = class_metrics(b1(fold_maps[sett]))
        tex_ok = all(roi[t]["share_w<0.5"] <= base_roi[t]["share_w<0.5"] + 0.01 for t in TEXTURE)
        bol_same = roi["083 bollard interior"]["share_w<0.5"] == base_roi["083 bollard interior"]["share_w<0.5"]
        cov = {k: roi[k]["coverage"] for k in ("076 facade ledge", "088 rock ridge")}
        sweep.append({"theta": sett[0], "k_L": sett[1], **{f"cov {k}": v for k, v in cov.items()},
                      **{f"tex {t}": roi[t]["share_w<0.5"] for t in TEXTURE},
                      "crease_class_cov": cls["crease"]["coverage"], "fold_class_cov": cls["fold"]["coverage"],
                      "texture_ok": tex_ok, "bollard_unchanged": bool(bol_same),
                      "meets": all(v >= 0.5 for v in cov.values()) and tex_ok and bool(bol_same)})
    dz = {}
    for dz0 in DZ0S:
        roi, cls = class_metrics(b1(dz_maps[dz0]))
        sh = pd.DataFrame(dz_shares[dz0])
        dz[str(dz0)] = {"deliverable_w<0.5_mean": float(sh["w<0.5"].mean()),
                        "deliverable_w<0.9_mean": float(sh["w<0.9"].mean()), "per_image": dz_shares[dz0],
                        "classes": cls, "texture": {t: roi[t]["share_w<0.5"] for t in TEXTURE}}
    rep = {"rescue": {"off_bit_identical": off_identical, "baseline_rois": base_roi, "baseline_classes": base_cls,
                      "sweep": sweep, "rescued_counts": rescued_counts,
                      "fold_runtime_with_rescue_s_P50": float(np.median(t_rescue))},
           "dZ0": dz, "clusters": clusters}
    (OUT / "g3.json").write_text(json.dumps(rep, indent=1, default=float))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print("fold rescue off bit-identical to geom_v0:", all(off_identical), f"({sum(off_identical)}/{len(off_identical)})")
    print("\nbaseline (geom_v0) creases:", {k: base_roi[k]["coverage"] for k in ("076 facade ledge", "088 rock ridge")})
    print("\nrescue sweep:\n" + pd.DataFrame(sweep).round(4).to_string(index=False))
    rc = pd.DataFrame(rescued_counts)
    print("\nrescued per image:\n" + rc.to_string(index=False))
    print("\nDelta Z0:")
    for k, v in dz.items():
        print(f"  {k} m: deliverable w<0.5 {v['deliverable_w<0.5_mean']:.4f}, w<0.9 {v['deliverable_w<0.9_mean']:.4f}; "
              + "; ".join(f"{c} share {m['share_w<0.5']:.3f} cov {m['coverage'] if m['coverage'] is None else round(m['coverage'], 3)}"
                          for c, m in v["classes"].items()) + f"; texture {v['texture']}")
    print("\nclusters:\n" + pd.DataFrame(clusters).drop(columns="tiles").to_string(index=False))


if __name__ == "__main__":
    main()
