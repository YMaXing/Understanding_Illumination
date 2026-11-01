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


ROLLOFF_MEDIAN_TARGET = 0.18       # exposure_mode=median_rolloff: gain = 0.18 / median(Y)
ROLLOFF_WHITE_PERCENTILE = 99.5    # extended-Reinhard white point: P99.5 of g*Y


def linear_to_srgb_float(
    raw: np.ndarray,
    *,
    black_level: float | list[float],
    white_level: float,
    white_balance: str,
    exposure_percentile: float,
    exposure_target: float,
    exposure_gain_max: float,
    exposure_mode: str = "p99",
    wb_gains_metadata: list[float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """RGB linear raw (H, W, 3), stored frame -> sRGB-encoded float64 in [0, 1] (before quantization), record.

    Steps (spec A1.1, A1.3-A1.5): levels, white balance, exposure, OETF.
    Saturated = any channel at or above white_level in the raw data; such pixels are excluded
    from the gray-world and exposure statistics.

    exposure_mode:
      p99            gain = min(target / P_pct(Y), gain_max), clip (baseline)
      median_rolloff gain g = min(0.18 / median(Y), gain_max), then a global extended-Reinhard curve on
                     L = g*Y with white point Lw = P99.5(L): L' = L (1 + L/Lw^2) / (1 + L), applied to RGB
                     as the luminance ratio L'/L (hue preserved), then clip. Global and monotone.
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
    record: dict[str, Any] = {
        "black_level": black.tolist(),
        "white_level": white,
        "white_balance": white_balance,
        "wb_gains": gains.tolist(),
        "exposure_mode": exposure_mode,
    }
    if exposure_mode == "p99":
        p = float(np.percentile(y, exposure_percentile)) if y.size else 0.0
        gain = min(exposure_target / p, exposure_gain_max) if p > 0 else exposure_gain_max
        x = np.clip(x * np.float32(gain), 0.0, 1.0)
        record.update({
            "exposure_percentile": exposure_percentile,
            "exposure_p_value": p,
            "exposure_gain": gain,
            "exposure_gain_clamped": bool(p <= 0 or exposure_target / p > exposure_gain_max),
        })
    elif exposure_mode == "median_rolloff":
        med = float(np.median(y)) if y.size else 0.0
        gain = min(ROLLOFF_MEDIAN_TARGET / med, exposure_gain_max) if med > 0 else exposure_gain_max
        L = (x.astype(np.float64) @ LUMA) * gain
        Lw = float(np.percentile(L[unsat], ROLLOFF_WHITE_PERCENTILE)) if unsat.any() else 1.0
        ratio = (1.0 + L / Lw**2) / (1.0 + L)  # = L'/L, finite at L = 0
        x = np.clip(x.astype(np.float64) * (gain * ratio)[..., None], 0.0, 1.0)
        record.update({
            "exposure_median_target": ROLLOFF_MEDIAN_TARGET,
            "exposure_median_Y": med,
            "exposure_gain": gain,
            "exposure_gain_clamped": bool(med <= 0 or ROLLOFF_MEDIAN_TARGET / med > exposure_gain_max),
            "rolloff_white_percentile": ROLLOFF_WHITE_PERCENTILE,
            "rolloff_white_point": Lw,
            "frac_clipped_after_rolloff": float((x >= 1.0).any(axis=2).mean()),
        })
    else:
        raise ValueError(f"unknown exposure_mode {exposure_mode!r}")
    record["saturation_frac"] = float(saturated.mean())
    return srgb_oetf(x), record


def quantize(srgb: np.ndarray, dither: str = "none", rng: np.random.Generator | None = None) -> np.ndarray:
    """sRGB float in [0, 1] -> uint8. dither=tpdf adds triangular-PDF noise (sum of two U(-0.5, 0.5),
    in code units) before rounding."""
    v = srgb * 255.0
    if dither == "none":
        return np.round(v).astype(np.uint8)
    if dither == "tpdf":
        if rng is None:
            raise ValueError("dither=tpdf needs a seeded rng")
        v = v + rng.uniform(-0.5, 0.5, v.shape) + rng.uniform(-0.5, 0.5, v.shape)
        return np.clip(np.round(v), 0, 255).astype(np.uint8)
    raise ValueError(f"unknown dither {dither!r}")


def linear_to_srgb(raw: np.ndarray, **kw) -> tuple[np.ndarray, dict[str, Any]]:
    """linear_to_srgb_float + quantize without dither: sRGB uint8 (H, W, 3), stored frame, plus a record."""
    srgb, record = linear_to_srgb_float(raw, **kw)
    return quantize(srgb), record
