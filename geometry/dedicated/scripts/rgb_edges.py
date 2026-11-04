"""RGB edge helpers for evaluation and diagnostic scripts only (never used by a channel).

luminance(): the upright sRGB PNG of Stage A area-resized to network resolution exactly as Stage A feeds DA3 (dither
none), linearized, Rec. 709 weights. rgb_offset(): along x0 + s n, s in [-half, half], the s of the maximum
|grad Y . n| (Gaussian derivative sigma = 1 px, parabola-refined); NaN if a sample leaves the image.
(Same definitions as scripts/depth_d2_report.py, parameterized by the Stage A folder.)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from illum_baseline.detector import gaussian_gradient
from illum_baseline.geom.sampling import bilinear


def network_size(stage_a: Path, stem: str) -> tuple[int, int]:
    m = json.loads((stage_a / f"{stem}.json").read_text())["network"]
    return m["W_n"], m["H_n"]


def luminance(stage_a: Path, stem: str) -> np.ndarray:
    W, H = network_size(stage_a, stem)
    bgr = cv2.imread(str(stage_a / "srgb" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    net = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float64) / 255.0
    lin = np.where(net <= 0.04045, net / 12.92, ((net + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def rgb_offset(Y: np.ndarray, x0: np.ndarray, n: np.ndarray, half: float = 8.0, step: float = 0.25) -> np.ndarray:
    return rgb_offset_strength(Y, x0, n, half, step)[0]


def gradient_median(Y: np.ndarray) -> float:
    """Median of |grad Y| (Gaussian derivative sigma = 1 px) over the image: the scale for a reliable RGB edge."""
    gu, gv = gaussian_gradient(Y, 1.0)
    return float(np.median(np.hypot(gu, gv)))


def rgb_offset_strength(Y: np.ndarray, x0: np.ndarray, n: np.ndarray, half: float = 8.0, step: float = 0.25
                        ) -> tuple[np.ndarray, np.ndarray]:
    """rgb_offset and the edge strength max |grad Y . n| at it (NaN where a sample leaves the image)."""
    gu, gv = gaussian_gradient(Y, 1.0)
    s = np.arange(-half, half + step / 2, step)
    pts = x0[:, None, :] + s[None, :, None] * n[:, None, :]
    ok_all = np.ones(Y.shape, dtype=bool)
    a, ok1 = bilinear(gu, ok_all, pts[..., 0], pts[..., 1])
    b, ok2 = bilinear(gv, ok_all, pts[..., 0], pts[..., 1])
    d = np.abs(a * n[:, None, 0] + b * n[:, None, 1])
    j = np.argmax(np.where(ok1 & ok2, d, -np.inf), axis=1)
    r = np.arange(len(x0))
    jm, jp = np.clip(j - 1, 0, len(s) - 1), np.clip(j + 1, 0, len(s) - 1)
    y0, ym, yp = d[r, j], d[r, jm], d[r, jp]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = ym - 2 * y0 + yp
        dl = np.where((j > 0) & (j < len(s) - 1) & (c < 0), 0.5 * (ym - yp) / c, 0.0)
    out = s[j] + np.clip(dl, -0.5, 0.5) * step
    inside = (ok1 & ok2).all(1)
    return np.where(inside, out, np.nan), np.where(inside, y0, np.nan)


def fit_shift(s: np.ndarray, n: np.ndarray, lim: float = 2.0) -> tuple[np.ndarray, np.ndarray, int]:
    """Least-squares (dx, dy) with s = n . (dx, dy) over the samples with |s| <= lim: (shift, SE, n used)."""
    m = np.isfinite(s) & (np.abs(s) <= lim)
    if m.sum() < 3:
        return np.full(2, np.nan), np.full(2, np.nan), int(m.sum())
    t, *_ = np.linalg.lstsq(n[m], s[m], rcond=None)
    res = s[m] - n[m] @ t
    cov = np.linalg.inv(n[m].T @ n[m]) * res.var(ddof=2)
    return t, np.sqrt(np.diag(cov)), int(m.sum())
