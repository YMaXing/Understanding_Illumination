"""Network input size and scale factors."""

from __future__ import annotations


def network_size(w: int, h: int, long_side: int = 504, multiple: int = 14) -> tuple[int, int]:
    """(W_n, H_n): long side -> long_side, short side -> multiple*round(short*long_side/long/multiple)."""
    if w >= h:
        return long_side, multiple * round(h * long_side / w / multiple)
    return multiple * round(w * long_side / h / multiple), long_side


def scale_factors(w_up: int, h_up: int, w_n: int, h_n: int) -> tuple[float, float]:
    """(s_x, s_y) = (W_n / W_up, H_n / H_up)."""
    return w_n / w_up, h_n / h_up


def focal_at_network(fx: float, fy: float, s_x: float, s_y: float) -> float:
    """Active focal at network resolution: f_n = (fx*s_x + fy*s_y) / 2 (fx, fy in upright full-res px)."""
    return (fx * s_x + fy * s_y) / 2.0
