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


# ---------------------------------------------------------------- test-time augmentation (TASK_070_STRIPES T3)

SPEC_SHIFTS = [(0, 0), (7, 0), (0, 7), (7, 7)]


def tta_shifts(shift_set: str) -> list[tuple[int, int]]:
    """(dx, dy) shifts. spec: the task's 4 shifts. latin14: 14 shifts (k-7, (3k mod 14)-7), every x and every
    y phase modulo 14 exactly once, so ripples locked to a 7- or 14-px grid along either axis average out."""
    if shift_set == "spec":
        return list(SPEC_SHIFTS)
    if shift_set == "latin14":
        return [(k - 7, (3 * k) % 14 - 7) for k in range(14)]
    raise ValueError(f"unknown tta_shift_set {shift_set!r}")


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Move content by (+dx, +dy) px: reflect-pad top/left (positive) or bottom/right (negative), crop back."""
    H, W = img.shape[:2]
    pad = [(max(dy, 0), max(-dy, 0)), (max(dx, 0), max(-dx, 0))] + [(0, 0)] * (img.ndim - 2)
    p = np.pad(img, pad, mode="reflect")
    y0, x0 = max(-dy, 0), max(-dx, 0)
    return np.ascontiguousarray(p[y0:y0 + H, x0:x0 + W])


def unshift(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Inverse of shift_image for an output map: back to original coordinates, NaN where not covered."""
    H, W = a.shape
    out = np.full((H, W), np.nan)
    ys = slice(max(-dy, 0), H - max(dy, 0)); yd = slice(max(dy, 0), H - max(-dy, 0))
    xs = slice(max(-dx, 0), W - max(dx, 0)); xd = slice(max(dx, 0), W - max(-dx, 0))
    out[ys, xs] = a[yd, xd]
    return out


def _feather_weight(cov: np.ndarray, m: int) -> np.ndarray:
    """Smoothstep of (distance to the edge of the pass's rectangular coverage + 1) / m, 0 outside coverage."""
    rows, cols = np.nonzero(cov.any(axis=1))[0], np.nonzero(cov.any(axis=0))[0]
    v, u = np.mgrid[0:cov.shape[0], 0:cov.shape[1]]
    d = np.minimum.reduce([v - rows[0], rows[-1] - v, u - cols[0], cols[-1] - u]).astype(np.float64)
    s = np.clip((d + 1) / m, 0.0, 1.0)
    return np.where(cov, s * s * (3 - 2 * s), 0.0)


def tta_predict(est: DepthEstimator, image: np.ndarray, shift_set: str, feather_px: int = 0) -> dict[str, Any]:
    """8 (spec) or 28 (latin14) passes: every shift with and without a horizontal flip.

    Each output is unshifted and unflipped; rho = 1/D of each pass is scaled to the unshifted, unflipped pass by
    the median ratio over their overlap (outside DA3's sky fill, score >= 0.3); then rho and the sky probability
    are averaged over the passes covering each pixel, and D = 1 / mean rho.

    feather_px > 0: weighted average instead, each pass weighted by a smoothstep ramp of the distance to the
    edge of its coverage (image border or crop line), so the averaging set changes continuously near borders.
    """
    passes = []
    for dx, dy in tta_shifts(shift_set):
        for flip in (False, True):
            x = image[:, ::-1] if flip else image
            pred = est.predict([shift_image(x, dx, dy)])[0]
            D = unshift(pred["D"].astype(np.float64), dx, dy)
            sky = unshift(pred["sky_prob"].astype(np.float64), dx, dy) if pred["sky_prob"] is not None else None
            if flip:
                D = D[:, ::-1]
                sky = sky[:, ::-1] if sky is not None else None
            passes.append({"dx": dx, "dy": dy, "flip": flip, "D": D, "sky": sky, "record": pred["record"]})
    ref = next(p for p in passes if (p["dx"], p["dy"], p["flip"]) == (0, 0, False))
    ref_rho = 1.0 / ref["D"]
    rho_sum = np.zeros_like(ref_rho)
    sky_sum = np.zeros_like(ref_rho)
    n = np.zeros(ref_rho.shape, dtype=np.int32)
    has_sky = ref["sky"] is not None
    wsum = np.zeros_like(ref_rho)
    for p in passes:
        rho = 1.0 / p["D"]
        m = np.isfinite(rho) & np.isfinite(ref_rho)
        if has_sky:
            m &= (np.nan_to_num(p["sky"]) < 0.3) & (ref["sky"] < 0.3)
        p["scale"] = float(np.median(ref_rho[m] / rho[m]))
        cov = np.isfinite(rho)
        wt = _feather_weight(cov, feather_px) if feather_px > 0 else cov.astype(np.float64)
        rho_sum[cov] += wt[cov] * p["scale"] * rho[cov]
        if has_sky:
            sky_sum[cov] += wt[cov] * p["sky"][cov]
        wsum += wt
        n += cov
    assert n.min() >= 1 and wsum.min() > 0
    D = (wsum / rho_sum).astype(np.float32)
    sky = np.clip(sky_sum / wsum, 0.0, 1.0).astype(np.float16) if has_sky else None
    rec = dict(ref["record"])
    rec.update({
        "tta": "shift_flip", "tta_shift_set": shift_set, "tta_n_passes": len(passes), "tta_feather_px": feather_px,
        "tta_scales": [p["scale"] for p in passes],
        "tta_coverage_min": int(n.min()), "tta_coverage_max": int(n.max()),
        "D_min": float(D.min()), "D_median": float(np.median(D)), "D_max": float(D.max()),
        "sky_note": "sky_prob = mean over covering passes of the clipped per-pass score" if has_sky else None,
    })
    return {"D": D, "sky_prob": sky, "record": rec, "passes": passes}
