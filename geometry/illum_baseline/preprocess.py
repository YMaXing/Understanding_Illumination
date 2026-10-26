"""Linear raw -> sRGB uint8 network input (spec A1). The linear data itself is never modified."""

from __future__ import annotations

from typing import Any

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722])


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 encoding: 12.92x for x <= 0.0031308, else 1.055 x^(1/2.4) - 0.055."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def resolve_layout(arr: np.ndarray, tags: dict[str, Any], layout: str) -> str:
    """'rgb' or 'mosaic'; 'auto' decides from shape/tags and fails loudly if they disagree."""
    if layout != "auto":
        return layout
    photometric = str(tags.get("PhotometricInterpretation", ""))
    has_cfa = "CFAPattern" in tags or photometric in ("CFA", "32803")
    if arr.ndim == 3 and arr.shape[2] == 3 and photometric == "RGB" and not has_cfa:
        return "rgb"
    if arr.ndim == 2 and has_cfa:
        return "mosaic"
    raise ValueError(
        f"cannot decide layout: shape={arr.shape}, Photometric={photometric!r}, CFA tags={has_cfa}; "
        "set preprocess.layout explicitly"
    )


def _per_channel(v: float | list[float], n: int) -> np.ndarray:
    a = np.atleast_1d(np.asarray(v, dtype=np.float64))
    if a.size == 1:
        a = np.repeat(a, n)
    if a.size != n:
        raise ValueError(f"expected 1 or {n} level values, got {a.size}")
    return a


def linear_to_srgb(
    raw: np.ndarray,
    *,
    black_level: float | list[float],
    white_level: float,
    white_balance: str,
    exposure_percentile: float,
    exposure_target: float,
    exposure_gain_max: float,
    wb_gains_metadata: list[float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """RGB linear raw (H, W, 3), stored frame -> sRGB uint8 (H, W, 3), stored frame, plus a record.

    Steps (spec A1.1, A1.3-A1.5): levels, white balance, exposure, OETF.
    Saturated = any channel at or above white_level in the raw data; such pixels are excluded
    from the gray-world and exposure statistics.
    """
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {raw.shape}; mosaic input is not supported yet")
    black = _per_channel(black_level, 3)
    white = float(white_level)
    x = (raw.astype(np.float32) - black.astype(np.float32)) / (white - black).astype(np.float32)
    saturated = np.any(raw >= white, axis=2)
    x = np.clip(x, 0.0, 1.0)
    unsat = ~saturated

    if white_balance == "grayworld":
        means = x[unsat].astype(np.float64).mean(axis=0)
        gains = means[1] / np.maximum(means, 1e-12)
    elif white_balance == "metadata":
        if wb_gains_metadata is None:
            raise ValueError("white_balance=metadata but the file has no WB metadata")
        gains = np.asarray(wb_gains_metadata, dtype=np.float64) / wb_gains_metadata[1]
    elif white_balance == "none":
        gains = np.ones(3)
    else:
        raise ValueError(f"unknown white_balance {white_balance!r}")
    x = x * gains.astype(np.float32)

    y = x[unsat].astype(np.float64) @ LUMA
    p = float(np.percentile(y, exposure_percentile)) if y.size else 0.0
    gain = min(exposure_target / p, exposure_gain_max) if p > 0 else exposure_gain_max
    x = np.clip(x * np.float32(gain), 0.0, 1.0)

    srgb = np.round(srgb_oetf(x) * 255.0).astype(np.uint8)
    record = {
        "black_level": black.tolist(),
        "white_level": white,
        "white_balance": white_balance,
        "wb_gains": gains.tolist(),
        "exposure_percentile": exposure_percentile,
        "exposure_p_value": p,
        "exposure_gain": gain,
        "exposure_gain_clamped": bool(p <= 0 or exposure_target / p > exposure_gain_max),
        "saturation_frac": float(saturated.mean()),
    }
    return srgb, record
