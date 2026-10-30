"""Stage B outputs (spec B8): upsampling, frame transforms, pair weights.

Network-resolution arrays only get `to_stored`; the full-resolution `w` is upsampled in the upright
frame first. Never differentiate an upsampled map.
"""

from __future__ import annotations

import cv2
import numpy as np

from .io_tiff import to_stored


def upsample_weights(w_net: np.ndarray, size_up: tuple[int, int], w_min: float) -> np.ndarray:
    """Upright network-res w -> upright (W_up, H_up) with bilinear, half-pixel centers, clipped to [w_min, 1].

    cv2.resize maps destination x to source (x + 0.5) * W_n / W_up - 0.5 (pixel-center alignment).
    """
    up = cv2.resize(np.asarray(w_net, dtype=np.float64), size_up, interpolation=cv2.INTER_LINEAR)
    return np.clip(up, w_min, 1.0)


def deliverable_w(w_net: np.ndarray, size_up: tuple[int, int], orientation: int, w_min: float) -> np.ndarray:
    """Full-resolution w in the stored frame: upsample (upright), clip, then to_stored."""
    return to_stored(upsample_weights(w_net, size_up, w_min), orientation)


def net_to_stored(a: np.ndarray, orientation: int) -> np.ndarray:
    return to_stored(a, orientation)


def pair_weights(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(w_x [H, W-1], w_y [H-1, W]): elementwise min of horizontal and of vertical neighbors."""
    w = np.asarray(w)
    return np.minimum(w[:, :-1], w[:, 1:]), np.minimum(w[:-1, :], w[1:, :])
