"""GeoCalib wrapper (see docs/api_notes.md, GeoCalib section).

Relies on geocalib/extractor.py: GeoCalib(weights).calibrate(img[3,H,W] float RGB in [0,1], camera_model),
which returns the camera in input-image pixels (GeoCalib._post_process -> BaseCamera.undo_scale_crop).

GeoCalib is stochastic: geocalib/modules.py: NMF2D._build_bases draws torch.rand bases on every forward
(about 1% spread in f across repeats, also on CPU). The torch RNG is reseeded before each call so the
result depends only on the image and the seed, not on processing order.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


class Calibrator:
    def __init__(self, weights: str, camera_model: str, device: torch.device, seed: int = 0):
        from geocalib import GeoCalib

        self.model = GeoCalib(weights=weights).to(device).eval()
        self.camera_model = camera_model
        self.device = device
        self.weights = weights
        self.seed = seed

    @torch.no_grad()
    def calibrate(self, srgb_upright: np.ndarray) -> dict[str, Any]:
        """Full-resolution upright sRGB uint8 (H, W, 3) -> focal, gravity and their uncertainties.

        fx, fy, focal_uncertainty are in upright full-resolution pixels; angles in degrees.
        cx, cy follow GeoCalib's corner-origin convention (w/2, h/2) and are for reference only.
        """
        h, w = srgb_upright.shape[:2]
        img = torch.from_numpy(srgb_upright).permute(2, 0, 1).float().div(255.0).to(self.device)
        torch.manual_seed(self.seed)
        out = self.model.calibrate(img, camera_model=self.camera_model)
        cam, grav = out["camera"], out["gravity"]
        size = cam.size[0].tolist()
        if not np.allclose(size, [w, h], atol=1e-2):
            raise RuntimeError(f"GeoCalib camera size {size} != input (W, H) = ({w}, {h})")
        fx, fy = (float(v) for v in cam.f[0])
        roll, pitch = (float(v) for v in grav.rp[0])

        def unc(k: str) -> float:
            return float(out[k].reshape(-1)[0])

        return {
            "fx": fx,
            "fy": fy,
            "cx_corner": float(cam.c[0, 0]),
            "cy_corner": float(cam.c[0, 1]),
            "focal_uncertainty": unc("focal_uncertainty"),
            "vfov_deg": math.degrees(float(cam.vfov[0])),
            "vfov_uncertainty_deg": math.degrees(unc("vfov_uncertainty")),
            "roll_deg": math.degrees(roll),
            "pitch_deg": math.degrees(pitch),
            "roll_uncertainty_deg": math.degrees(unc("roll_uncertainty")),
            "pitch_uncertainty_deg": math.degrees(unc("pitch_uncertainty")),
            "gravity_uncertainty_deg": math.degrees(unc("gravity_uncertainty")),
            "input_WxH": [w, h],
            "weights": self.weights,
            "camera_model": self.camera_model,
            "torch_seed": self.seed,
        }
