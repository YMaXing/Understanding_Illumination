"""Degenerate baselines (DESIGN.md §7), run alongside every model.

  all_R   ℓR = ℓI                  (everything in reflectance)
  all_L   ℓR = 0                   (everything in illumination)
  blur_L  ℓR = ℓI - G_sigma * ℓI    (L = the input blurred, sigma = 8 px; per channel, reflect borders)
"""

import numpy as np
from scipy import ndimage as ndi


def all_R(lI, cfg=None):
    return np.array(lI, dtype=np.float64, copy=True)


def all_L(lI, cfg=None):
    return np.zeros_like(lI, dtype=np.float64)


def blur_L(lI, cfg):
    B = cfg["baselines"]
    s = float(B["blur_sigma_px"])
    G = np.stack([ndi.gaussian_filter(lI[..., c].astype(np.float64), s, mode="reflect", truncate=B["blur_truncate"])
                  for c in range(lI.shape[-1])], axis=-1)
    return lI - G


BASELINES = {"all_R": all_R, "all_L": all_L, "blur_L": blur_L}


def predict(name, lI, cfg):
    return BASELINES[name](lI, cfg)
