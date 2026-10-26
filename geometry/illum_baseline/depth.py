"""DA3METRIC-LARGE wrapper (see docs/api_notes.md, DA3 section).

Uses DA3's own inference steps from depth_anything_3/api.py (DepthAnything3._preprocess_inputs,
_prepare_model_inputs, _run_model_forward) instead of inference(), because inference() passes the
output through OutputProcessor._extract_sky, which thresholds the sky score to a bool mask.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

SKY_DA3_FILL_THRESHOLD = 0.3  # depth_anything_3/model/da3.py: _process_mono_sky_estimation


class DepthEstimator:
    def __init__(self, model_id: str, process_res: int, process_res_method: str,
                 device: torch.device, seed: int):
        from depth_anything_3.api import DepthAnything3

        self.model = DepthAnything3.from_pretrained(model_id).to(device).eval()
        self.model_id = model_id
        self.process_res = process_res
        self.process_res_method = process_res_method
        self.device = device
        self.seed = seed

    @torch.inference_mode()
    def predict(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        """Same-shape upright uint8 RGB images (H_n, W_n, 3) -> per image D, sky score, and a record.

        DA3 must not resize: the processed shape is asserted to equal the input shape.
        The torch RNG is reseeded per call because _process_mono_sky_estimation subsamples
        depths with torch.randint to compute the sky fill value.
        """
        shapes = {im.shape for im in images}
        if len(shapes) != 1:
            raise ValueError(f"batch must share one shape, got {shapes}")
        (h, w, c), = shapes
        if c != 3 or images[0].dtype != np.uint8:
            raise ValueError("expected uint8 (H, W, 3) RGB")

        m = self.model
        imgs_cpu, _, _ = m._preprocess_inputs(images, None, None, self.process_res, self.process_res_method)
        if tuple(imgs_cpu.shape) != (len(images), 3, h, w):
            raise RuntimeError(f"DA3 resized the input: processed {tuple(imgs_cpu.shape)} for {(len(images), 3, h, w)}")
        imgs, _, _ = m._prepare_model_inputs(imgs_cpu, None, None)
        torch.manual_seed(self.seed)
        raw = m._run_model_forward(imgs, None, None, [], False, False, "saddle_balanced")
        depth = raw["depth"].float().cpu().numpy()[0]  # (N, H, W)
        sky = raw["sky"].float().cpu().numpy()[0] if "sky" in raw else None
        if depth.shape != (len(images), h, w):
            raise RuntimeError(f"DA3 depth shape {depth.shape} != {(len(images), h, w)}")

        autocast = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
        results = []
        for i in range(len(images)):
            D = depth[i].astype(np.float32)
            rec: dict[str, Any] = {
                "model_id": self.model_id,
                "process_res": self.process_res,
                "process_res_method": self.process_res_method,
                "processed_HxW": [h, w],
                "batch_size": len(images),
                "backbone_autocast": autocast if self.device.type == "cuda" else "cpu-autocast",
                "D_min": float(D.min()), "D_median": float(np.median(D)), "D_max": float(D.max()),
                "D_n_nonfinite": int((~np.isfinite(D)).sum()),
                "D_n_nonpositive": int((D <= 0).sum()),
                "conf": None,
            }
            if sky is None:
                rec["sky_output"] = False
                results.append({"D": D, "sky_prob": None, "record": rec})
                continue
            s = sky[i]
            fill_mask = s >= SKY_DA3_FILL_THRESHOLD
            rec.update({
                "sky_output": True,
                "sky_activation": "relu score, clipped to [0, 1] as sky_prob",
                "sky_score_max": float(s.max()),
                "sky_frac_score_gt_1": float((s > 1).mean()),
                "sky_frac_ge_0.5": float((s >= 0.5).mean()),
                "sky_frac_0.3_to_0.5": float(((s >= 0.3) & (s < 0.5)).mean()),
                "sky_fill_frac": float(fill_mask.mean()),
                "sky_fill_depth": float(D[fill_mask][0]) if fill_mask.any() else None,
            })
            results.append({"D": D, "sky_prob": np.clip(s, 0.0, 1.0).astype(np.float16), "record": rec})
        return results
