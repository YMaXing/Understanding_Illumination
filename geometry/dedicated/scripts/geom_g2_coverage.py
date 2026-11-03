"""TASK_GEOM_INTEGRATION G2(c): band coverage of RGB edges by the geom_v0 deliverable, per setting (diagnostic only).

Settings: band_min in {2, 3, 4} (both channels: depth b_c = max(band_min, ceil(sigma_wc) + 1), fold
b_c = max(band_min, ceil(width / 2))) x sky_band in {depth, depth_and_mask}; the deliverable shift stays off (G2(a):
not stable). w is re-rasterized from the saved geom_v0 candidate tables with the channels' own rule (min over disks,
then min over channels; at band_min 2 / depth it reproduces the saved w_net bit for bit, checked); the sky ring is
geom.depth.sky_boundary_ring within 4 px of an accepted sky edge; the deliverable is outputs.upsample_weights.
Metrics:
  depth edges: strong accepted depth edges (scripts/geom_g2_align.py), the RGB edge x0 + s* n (|s*| <= 8) mapped to
    upright original pixels, covered if the deliverable w < 0.5 there (20 images); also at the DA3 position x0;
  fold targets (batch1): fold_null_rois TARGETS (fold + crease) and the depth_rois fold polylines, every 0.5 px, the
    RGB edge along the polyline normal, as above;
  sky: the 070 sky band (scripts/depth_rois.py) share with w_net < 0.5;
  texture (constraint): share of w_net < 0.5 in the 070 floor, 079 grass, 079 grass sunlit, 083 bollard interior.
Selection (proposed): the smallest setting with depth-edge coverage >= 0.90, sky band >= 0.95 and every texture share
<= geom_v0 + 0.01. Writes baseline_output/diag/geom/g2/coverage.json.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from depth_rois import BOXES, POLYLINES, box_mask, sky_band_070  # noqa: E402
from fold_null_rois import TARGETS, _poly_samples, rois  # noqa: E402
from rgb_edges import luminance, rgb_offset  # noqa: E402
from geom_g2_align import strong  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.detector import disk  # noqa: E402
from illum_baseline.geom.depth import sky_boundary_ring  # noqa: E402
from illum_baseline.outputs import upsample_weights  # noqa: E402
from illum_baseline.stage_b import image_stems, load_upright  # noqa: E402

BATCHES = {"batch1": ROOT / "baseline_output", "batch2": ROOT / "baseline_output_batch2"}
RUN = "geom_v0"
OUT = ROOT / "baseline_output" / "diag" / "geom" / "g2"
W_MIN = 0.05


def raster(shape, u, v, w_c, band) -> np.ndarray:
    H, W = shape
    w = np.ones(shape)
    u, v = np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)
    for b in np.unique(band):
        a = band == b
        wi = np.ones(shape)
        np.minimum.at(wi, (v[a], u[a]), w_c[a])
        w = np.minimum(w, ndimage.grey_erosion(wi, footprint=disk(int(b)), mode="constant", cval=1.0))
    return w


def channel_w(dc, fc, shape, band_min, sky_band, S):
    d = dc[dc["accepted"]]
    wd = raster(shape, d["u"].to_numpy(), d["v"].to_numpy(), d["w_c"].to_numpy(),
                np.maximum(band_min, np.ceil(d["sigma_wc"].to_numpy()).astype(int) + 1))
    if sky_band == "depth_and_mask":
        sk = d[d["sky_edge"]]
        if len(sk):
            seed = np.ones(shape, dtype=bool)
            seed[np.clip(sk["v"], 0, shape[0] - 1), np.clip(sk["u"], 0, shape[1] - 1)] = False
            ring = sky_boundary_ring(S, 2) & (ndimage.distance_transform_edt(seed) <= 4.0)
            wd[ring] = W_MIN
    f = fc[fc["accepted"]]
    wf = raster(shape, f["u"].to_numpy(), f["v"].to_numpy(), f["w_c"].to_numpy(),
                np.maximum(band_min, np.ceil(f["width"].to_numpy() / 2).astype(int)))
    return np.minimum(wd, wf)


def to_full(p: np.ndarray, net_wh, up_wh) -> np.ndarray:
    """Network-pixel (u, v) -> upright original pixel (inverse of the half-pixel-centred resize)."""
    return (p + 0.5) * (np.array(up_wh, float) / np.array(net_wh, float)) - 0.5


def covered(wfull: np.ndarray, pf: np.ndarray) -> np.ndarray:
    H, W = wfull.shape
    iu = np.clip(np.rint(pf[:, 0]).astype(int), 0, W - 1)
    iv = np.clip(np.rint(pf[:, 1]).astype(int), 0, H - 1)
    return wfull[iv, iu] < 0.5


def poly_normals(poly):
    pts = _poly_samples(poly)
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return pts, np.stack([-t[:, 1], t[:, 0]], -1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    settings = list(itertools.product((2, 3, 4), ("depth", "depth_and_mask")))
    acc = {s: {"edge_rgb": [], "edge_x0": [], "fold_rgb": [], "sky": None, "texture": {}} for s in settings}
    check = []
    for batch, root in BATCHES.items():
        rd = root / "stage_b" / RUN
        cfg = load_config(rd / "config_resolved.yaml")
        for stem in image_stems(rd):
            a, meta = load_upright(cfg, rd, stem)
            dc = pd.read_parquet(rd / f"{stem}_depth_candidates.parquet")
            fc = pd.read_parquet(rd / f"{stem}_fold_candidates.parquet")
            shape = a["w_net"].shape
            S = a["sky_mask"].astype(bool)
            net_wh, up_wh = (shape[1], shape[0]), tuple(meta["geometry"]["upright_WxH"])
            Y = luminance(root / "stage_a", stem)
            st = strong(dc)
            x0, n = st[["u0", "v0"]].to_numpy(), st[["n_u", "n_v"]].to_numpy()
            s = rgb_offset(Y, x0, n)
            okd = np.isfinite(s)
            p_rgb, p_x0 = to_full(x0[okd] + s[okd, None] * n[okd], net_wh, up_wh), to_full(x0, net_wh, up_wh)
            fold_pts = []
            if batch == "batch1":
                polys = [poly for (st_, _, poly) in TARGETS.values() if stem.endswith(st_)]
                polys += [poly for (st_, cls, poly) in POLYLINES.values() if stem.endswith(st_) and cls == "fold"]
                for poly in polys:
                    pts, nn = poly_normals(poly)
                    o = rgb_offset(Y, pts, nn)
                    ok = np.isfinite(o)
                    fold_pts.append(to_full(pts[ok] + o[ok, None] * nn[ok], net_wh, up_wh))
            for sett in settings:
                w = channel_w(dc, fc, shape, *sett, S)
                if sett == (2, "depth"):
                    check.append(bool(np.array_equal(w.astype(np.float16), a["w_net"].astype(np.float16))))
                wfull = upsample_weights(w, up_wh, W_MIN)
                acc[sett]["edge_rgb"].append(covered(wfull, p_rgb))
                acc[sett]["edge_x0"].append(covered(wfull, p_x0))
                for fp in fold_pts:
                    acc[sett]["fold_rgb"].append(covered(wfull, fp))
                if batch == "batch1" and stem.endswith("070"):
                    acc[sett]["sky"] = float((w[sky_band_070()] < 0.5).mean())
                if batch == "batch1":
                    tex = {k: m for k, m in rois(stem, shape).items()}
                    for name, (st_, _, box) in BOXES.items():
                        if stem.endswith(st_) and "legacy" not in name:
                            tex[name] = box_mask(shape, box)
                    for k, m in tex.items():
                        acc[sett]["texture"][k] = float((w[m] < 0.5).mean())
    base = acc[(2, "depth")]["texture"]
    rows = []
    for sett in settings:
        r = acc[sett]
        e = np.concatenate(r["edge_rgb"])
        tex_ok = all(r["texture"][k] <= base[k] + 0.01 for k in base)
        row = {"band_min": sett[0], "sky_band": sett[1], "depth_edge_cov_rgb": float(e.mean()),
               "depth_edge_cov_x0": float(np.concatenate(r["edge_x0"]).mean()), "n_edges": int(e.size),
               "fold_target_cov_rgb": float(np.concatenate(r["fold_rgb"]).mean()), "sky_band_070": r["sky"],
               **{f"tex {k}": v for k, v in r["texture"].items()}, "texture_ok": tex_ok}
        row["meets"] = row["depth_edge_cov_rgb"] >= 0.90 and row["sky_band_070"] >= 0.95 and tex_ok
        rows.append(row)
    df = pd.DataFrame(rows)
    pick = df[df["meets"]].sort_values(["band_min", "sky_band"]).head(1)
    rep = {"settings": rows, "rebuild_bit_identical": check, "selected": pick.to_dict(orient="records"),
           "shift": "off (G2(a): not stable)"}
    (OUT / "coverage.json").write_text(json.dumps(rep, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    print("re-rasterization bit-identical to the saved w_net:", all(check), f"({sum(check)}/{len(check)})")
    print("selected:", pick.to_dict(orient="records") or "none meets the rule")


if __name__ == "__main__":
    main()
