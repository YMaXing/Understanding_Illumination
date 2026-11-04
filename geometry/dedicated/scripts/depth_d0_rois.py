"""TASK_DEPTH_CHANNEL D0: proposed ROIs for the depth channel, snapped and drawn for checking.

Rough polylines (upright network px) are snapped onto the edge they name: silhouettes and limbs onto the maximum
|d log D / ds| (depth step), folds onto the maximum |d(normal angle) / ds| (scripts/fold_v2_f6.snap). The 070 sky
boundary is not hand-drawn: it is the boundary of the saved sky mask (tri_veto4 run), excluding the image border,
dilated to a 2-px band. Writes baseline_output/diag/depth_v1/d0/{rois.json, overlay_<stem>.png}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fold_null_rois import TARGETS  # noqa: E402
from fold_v2_f6 import canon, setup, snap  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.geom.sampling import bilinear  # noqa: E402

OUT = ROOT / "baseline_output" / "diag" / "depth_v1" / "d0"
# name: (stem, class, snap mode, rough polyline)
PROPOSED = {
    "079 box silhouette left": ("079", "step", "depth", [(118, 128), (113, 248)]),
    "079 box silhouette top-back L": ("079", "step", "depth", [(123, 113), (233, 90)]),
    "079 box silhouette top-back R": ("079", "step", "depth", [(240, 92), (313, 145)]),
    "079 box lit/shaded fold": ("079", "fold", "normal", [(199, 222), (189, 263)]),
    "083 bollard limb L": ("083", "limb", "depth", [(156, 215), (158, 312)]),
    "083 bollard limb R": ("083", "limb", "depth", [(229, 215), (226, 312)]),
    "083 bollard base rim": ("083", "fold", "normal", [(170, 322), (188, 333), (210, 325)]),
}


def snap_depth(D, pts, every=8.0, search=5.0):
    """Snap a rough polyline onto the maximum |d log D / ds| along each segment's normal; refit by PCA."""
    logD = np.log(D)
    ok = np.ones(D.shape, dtype=bool)
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        t = (b - a) / np.linalg.norm(b - a)
        e = canon([-t[1], t[0]])
        hits = []
        for c in np.linspace(0, np.linalg.norm(b - a), max(int(np.linalg.norm(b - a) / every), 2) + 1):
            x = a + c * t
            s = np.arange(-search, search + 1e-9, 0.25)
            p = x[None] + s[:, None] * e[None]
            z, okz = bilinear(logD, ok, p[:, 0], p[:, 1])
            if okz.sum() < 5:
                continue
            g = np.abs(np.gradient(z, s))
            g[~okz] = -1
            hits.append(x + s[int(np.argmax(g))] * e)
        if len(hits) < 2:
            out.append((a, b))
            continue
        h = np.array(hits)
        cen = h.mean(0)
        dvec = np.linalg.svd(h - cen)[2][0]
        proj = lambda q: cen + ((q - cen) @ dvec) * dvec  # noqa: E731
        out.append((proj(a), proj(b)))
    poly = [out[0][0]] + [0.5 * (out[i][1] + out[i + 1][0]) for i in range(len(out) - 1)] + [out[-1][1]]
    return np.array(poly)


def sky_boundary_band(sky):
    H, W = sky.shape
    b = sky & ~ndimage.binary_erosion(sky, structure=np.ones((3, 3), bool), border_value=1)
    b[:1, :] = b[-1:, :] = False
    b[:, :1] = b[:, -1:] = False
    return ndimage.binary_dilation(b, structure=np.ones((3, 3), bool), iterations=2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "candidate_fold_v2.yaml")
    res = {"polylines": {}, "areas": {}}
    figs = {}
    for name, (s, cls, mode, pts) in PROPOSED.items():
        stem = f"taiwei_cui_{s}"
        if mode == "depth":
            D = np.load(ROOT / "baseline_output/stage_a" / f"{stem}.npz")["D"].astype(float)
            poly = snap_depth(D, pts)
        else:
            ctx, _ = setup(stem, cfg)
            poly = snap(ctx, pts)
        res["polylines"][name] = {"stem": s, "class": cls, "polyline": poly.round(1).tolist()}
        figs.setdefault(s, []).append((name, cls, poly, False))
    # existing line targets, with their classes for the depth evaluation
    for name, (s, kind, poly) in TARGETS.items():
        figs.setdefault(s, []).append((name, "fold", np.array(poly), True))
    sky = np.load(ROOT / "baseline_output/stage_b/tri_veto4/taiwei_cui_070.npz")["sky_mask"]
    band = sky_boundary_band(sky)
    res["areas"]["070 sky boundary"] = {"stem": "070", "class": "step",
                                        "definition": "2-px band around the tri_veto4 sky-mask boundary, image border "
                                                      "excluded", "n_px": int(band.sum())}
    np.save(OUT / "070_sky_boundary_band.npy", band)
    colors = {"step": (0, 0, 255), "limb": (255, 0, 255), "fold": (0, 200, 0), "texture": (0, 200, 255)}
    for s in sorted(set(figs) | {"070"}):
        stem = f"taiwei_cui_{s}"
        meta = json.loads((ROOT / "baseline_output/stage_a" / f"{stem}.json").read_text())
        W, H = meta["network"]["W_n"], meta["network"]["H_n"]
        im = cv2.resize(cv2.imread(str(ROOT / "baseline_output/stage_a/srgb" / f"{stem}.png")), (W, H),
                        interpolation=cv2.INTER_AREA)
        f = 2
        big = cv2.resize(im, (W * f, H * f), interpolation=cv2.INTER_NEAREST)
        if s == "070":
            floor = np.load(ROOT / "baseline_output/diag/070/roi_mask.npy").astype(bool)
            big[np.kron(floor, np.ones((f, f), bool))] //= 2  # 070 floor ROI darkened
            big[np.kron(band, np.ones((f, f), bool))] = (0, 0, 255)
        if s == "079":
            for u0, u1, v0, v1 in ((5, 110, 5, 110), (5, 60, 360, 495), (110, 290, 5, 85)):
                cv2.rectangle(big, (f * u0, f * v0), (f * u1, f * v1), colors["texture"], 1)
            for v in (230, 254):
                cv2.line(big, (f * 180, f * v), (f * 215, f * v), colors["fold"], 2)
        if s == "083":
            cv2.rectangle(big, (f * 175, f * 260), (f * 205, f * 330), (255, 255, 0), 1)
        for name, cls, poly, existing in figs.get(s, []):
            p = (f * poly).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(big, [p], False, colors[cls], 1 if existing else 2)
            if not existing:
                cv2.putText(big, name.split(" ", 1)[1], tuple((f * poly[0] + [4, -4]).astype(int)), 0, 0.4, colors[cls], 1)
        cv2.putText(big, "red: step  magenta: limb  green: fold (thin: existing)  orange: texture  cyan: bollard ROI",
                    (5, 14), 0, 0.42, (255, 255, 255), 1)
        cv2.imwrite(str(OUT / f"overlay_{s}.png"), big)
    (OUT / "rois.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
