"""TASK_GEOM_V2 V5: the Laplacian channel as a fallback (geometry.laplacian.role: fallback).

The Laplacian's weight is set to 1 within d_excl px of an accepted geometric edge, so near those edges the depth and fold
channels alone set the band; the fusion stays the elementwise min. Exclusion edges (V3 review): the accepted fold
candidates (rescued included; at their V2b site when one is recorded) and the accepted depth candidates except thin
doublets, sky edges kept although they carry the thin flag. The distance is a Euclidean distance transform from the
rasterized centreline pixels, at network resolution. Shared by Stage B and the V5 sweep (scripts/geom_v5_fallback.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage


def centrelines(depth: pd.DataFrame | None, fold: pd.DataFrame | None, shape, include_thin: bool = False) -> np.ndarray:
    """Boolean (H, W) map of the exclusion edges' centreline pixels."""
    H, W = shape
    c = np.zeros((H, W), dtype=bool)
    if depth is not None and len(depth):
        sel = np.array(depth["accepted"], dtype=bool)
        if not include_thin:
            sel = sel & (~np.array(depth["thin"], dtype=bool) | np.array(depth["sky_edge"], dtype=bool))
        d = depth[sel]
        c[np.clip(d["v"].to_numpy(), 0, H - 1), np.clip(d["u"].to_numpy(), 0, W - 1)] = True
    if fold is not None and len(fold):
        f = fold[np.array(fold["accepted"], dtype=bool)]
        uu, vv = (f["u_site"], f["v_site"]) if "u_site" in f else (f["u"], f["v"])
        c[np.clip(vv.to_numpy(), 0, H - 1), np.clip(uu.to_numpy(), 0, W - 1)] = True
    return c


def exclusion_zone(depth, fold, shape, d_excl: float, include_thin: bool = False) -> np.ndarray:
    """Pixels within d_excl (Euclidean, px) of an exclusion-edge centreline."""
    c = centrelines(depth, fold, shape, include_thin)
    if not c.any():
        return np.zeros(shape, dtype=bool)
    return ndimage.distance_transform_edt(~c) <= d_excl


def apply_fallback(w_lap: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """The Laplacian's weight entering the min: 1 inside the exclusion zone, w_lap elsewhere."""
    return np.where(zone, 1.0, w_lap)
