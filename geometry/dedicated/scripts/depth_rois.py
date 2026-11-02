"""Pinned evaluation ROIs for the depth channel (TASK_DEPTH_CHANNEL; proposed in D0, confirmed before D3).

Upright network px (u, v). Polylines were snapped in D0 (scripts/depth_d0_rois.py): silhouettes and limbs onto the
maximum |d log D / ds|, folds onto the maximum normal-angle slope. Changes at confirmation:
- 079 box silhouette left trimmed to v <= 238 (excludes the plinth contact, where Delta Z -> 0);
- 080 side face edge R (a fold line target in scripts/fold_null_rois.py) is evaluated as class step (pre-D6);
- 083 bollard interior: the trimmed box u 175-205 x v 260-318 is the one to use from D5 on. The legacy box
  (v 260-330, scripts/fold_null_rois.py) overlaps the base rim (v ~ 326-332), which explains the F2 rim-only
  w_fold < 0.5 pixels; it is kept for comparability with earlier reports.

The fold line targets (narrow-face edges, creases) stay in scripts/fold_null_rois.py: TARGETS (class fold).
The 070 sky boundary is a definition, not a stored mask: sky_boundary_band() on the tri_veto4 run's saved sky mask
(a bit-for-bit reference run).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]

POLYLINES = {  # name: (stem, class, polyline)
    "079 box silhouette left": ("079", "step", [(118.5, 128.0), (113.4, 238.0)]),
    "079 box silhouette top-back L": ("079", "step", [(122.8, 111.8), (232.9, 89.5)]),
    "079 box silhouette top-back R": ("079", "step", [(237.4, 95.8), (312.3, 146.1)]),
    "079 box lit/shaded fold": ("079", "fold", [(201.5, 222.8), (189.3, 263.1)]),
    "083 bollard limb L": ("083", "limb", [(149.9, 215.6), (160.0, 311.8)]),
    "083 bollard limb R": ("083", "limb", [(230.5, 215.1), (224.1, 311.9)]),
    "083 bollard base rim": ("083", "fold", [(169.0, 324.5), (188.0, 332.0), (210.4, 326.7)]),
    # relabelled before D6: its upper half is the side face's outer silhouette against the background (an occlusion);
    # scripts/fold_null_rois.py TARGETS keeps it as a narrow-face fold target (legacy label, F9 fold evaluation)
    "080 side face edge R": ("080", "step", [(260.5, 95.1), (233.5, 199.2), (223.9, 289.8)]),
}

RELABELLED = {"080 side face edge R": ("fold", "step")}  # fold-target names whose class the depth evaluation overrides

BOXES = {  # name: (stem, class, (u0, u1, v0, v1)) inclusive pixel ranges
    "083 bollard interior": ("083", "texture", (175, 205, 260, 318)),
    "083 bollard interior (legacy)": ("083", "texture", (175, 205, 260, 330)),
}


def sky_boundary_band(sky: np.ndarray) -> np.ndarray:
    """2-px band around the boundary of a sky mask, the image border excluded (070 sky boundary ROI, class step)."""
    b = sky & ~ndimage.binary_erosion(sky, structure=np.ones((3, 3), bool), border_value=1)
    b[:1, :] = b[-1:, :] = False
    b[:, :1] = b[:, -1:] = False
    return ndimage.binary_dilation(b, structure=np.ones((3, 3), bool), iterations=2)


def sky_band_070() -> np.ndarray:
    sky = np.load(ROOT / "baseline_output/stage_b/tri_veto4/taiwei_cui_070.npz")["sky_mask"]
    return sky_boundary_band(sky)


def box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    u0, u1, v0, v1 = box
    m = np.zeros(shape, dtype=bool)
    m[v0:v1 + 1, u0:u1 + 1] = True
    return m
