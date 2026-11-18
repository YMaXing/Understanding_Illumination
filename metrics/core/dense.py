"""Dense and anchor diagnostics (DESIGN.md §6). Per image; never headline numbers.

  D1  mean ||∇ℓL|| where the smoothness weight >= d1_weight_min (w_union: 1 away from edges)
  D2  Σ||∇ℓR|| / Σ||∇ℓI|| over illumination-edge pixels (default detector)
  D3  Σ||∇ℓL|| / Σ||∇ℓI|| over material-edge pixels not on geometric edges (geom_v1_1 w < 0.5) and farther than
      d3_exclude_illum_px from an illumination pixel
  D4, A1, A2   need material regions: "n/a: no region source" (Delta B2)
Gradients: central differences, valid where the pixel and its 4 neighbours are valid; channel norm.
A diagnostic whose map is missing for the image's set is "n/a: map not available".
"""

import numpy as np
from scipy import ndimage as ndi

from . import config as K

NA_REGION = "n/a: no region source"
NA_MAP = "n/a: map not available"


def grad_norm(X, valid):
    """||∇X|| per pixel (channel norm of the central-difference gradient) and its validity mask."""
    H, W, _ = X.shape
    g = np.zeros((H, W))
    ok = np.zeros((H, W), bool)
    gx = (X[1:-1, 2:] - X[1:-1, :-2]) / 2
    gy = (X[2:, 1:-1] - X[:-2, 1:-1]) / 2
    g[1:-1, 1:-1] = np.sqrt(np.mean(gx ** 2 + gy ** 2, axis=-1))
    ok[1:-1, 1:-1] = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1])
    return g, ok


def load_map(cfg, name, stem, set_name, shape):
    e = cfg["maps"].get(name, {}).get(set_name)
    if not e:
        return None, None
    p = K.repo_path(e["dir"]) / f"{stem}.npz"
    if not p.exists():
        return None, None
    with np.load(p) as z:
        m = z[e["key"]]
    if m.shape != tuple(shape[:2]):
        raise ValueError(f"{p}: {e['key']} {m.shape} vs image {shape[:2]}")
    return m, dict(path=str(p.relative_to(K.REPO)), key=e["key"], sha256=K.sha256(p))


class ImageMaps:
    """Per-image maps shared by every method (loaded once)."""

    def __init__(self, cfg, stem, set_name, img):
        sh = img["lI"].shape
        self.w, self.w_src = load_map(cfg, "smoothness_weight", stem, set_name, sh)
        self.illum, self.illum_src = load_map(cfg, "illum_edges", stem, set_name, sh)
        self.mat, self.mat_src = load_map(cfg, "material_edges", stem, set_name, sh)
        self.geom = img.get("geom_edge")
        self.gI, self.okI = grad_norm(img["lI"], img["valid"])
        D = cfg["dense"]
        if self.mat is not None and self.illum is not None and self.geom is not None:
            far = ndi.distance_transform_edt(~self.illum.astype(bool)) > D["d3_exclude_illum_px"]
            self.d3_mask = self.mat.astype(bool) & ~self.geom & far
        else:
            self.d3_mask = None
        self.sources = dict(smoothness_weight=self.w_src, illum_edges=self.illum_src, material_edges=self.mat_src)


def diagnostics(cfg, img, maps, lR, lL):
    D = cfg["dense"]
    gR, okR = grad_norm(lR, img["valid"])
    gL, okL = grad_norm(lL, img["valid"])
    ok = maps.okI & okR & okL
    out = {}
    if maps.w is not None:
        m = ok & (maps.w >= D["d1_weight_min"])
        out["D1"] = float(gL[m].mean()) if m.any() else NA_MAP
        out["D1_n"] = int(m.sum())
    else:
        out["D1"], out["D1_n"] = NA_MAP, 0
    if maps.illum is not None:
        m = ok & maps.illum.astype(bool)
        out["D2"] = float(gR[m].sum() / gI_sum) if (gI_sum := maps.gI[m].sum()) > 0 else NA_MAP
        out["D2_n"] = int(m.sum())
    else:
        out["D2"], out["D2_n"] = NA_MAP, 0
    if maps.d3_mask is not None:
        m = ok & maps.d3_mask
        out["D3"] = float(gL[m].sum() / gI_sum) if (gI_sum := maps.gI[m].sum()) > 0 else NA_MAP
        out["D3_n"] = int(m.sum())
    else:
        out["D3"], out["D3_n"] = NA_MAP, 0
    out["D4"] = NA_REGION
    for q in D["anchor_q"]:
        out[f"A1_q{q:.2f}"] = NA_REGION
        out[f"A2_q{q:.2f}"] = NA_REGION
    return out
