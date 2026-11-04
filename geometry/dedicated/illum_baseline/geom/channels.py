"""Geometric-edge channels: result container, protocol, registry and combiner (docs/geometry_channels.md section 3.1).

Each channel reads the shared GeometryContext and returns a ChannelResult; Stage B combines the channels' w_net with
`combine` (elementwise min, the only mode). A channel is a class with `name` and
`run(ctx, params) -> ChannelResult`, registered in CHANNELS ("laplacian", "fold", "depth").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .context import GeometryContext


@dataclass
class ChannelResult:
    name: str
    w_net: np.ndarray                  # (H, W) in [w_min, 1], network res, upright
    strength: np.ndarray | None        # dimensionless per-pixel strength (fold: J, depth: r = Delta Z / Z_near at
                                       # rasterized pixels, else 0; depth: inf at sky edges, so any fusion on
                                       # strengths must treat inf as saturated)
    candidates: pd.DataFrame | None    # one row per candidate (all, with accept/reject reasons)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)  # saved as "<name>_<key>"
    scalars: dict[str, float] = field(default_factory=dict)
    w_min: float = 0.05
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)  # other per-image tables (fold: the null sample)


class Channel(Protocol):
    name: str

    def run(self, ctx: GeometryContext, params: dict) -> ChannelResult: ...


def combine(results: list[ChannelResult], mode: str = "min") -> np.ndarray:
    """Elementwise min of the channels' w_net (one channel: its w_net unchanged, bit for bit)."""
    if mode != "min":
        raise ValueError(f"combine mode {mode!r} is not implemented; only 'min'")
    if not results:
        raise ValueError("combine needs at least one channel result")
    if len(results) == 1:
        return results[0].w_net
    return np.minimum.reduce([r.w_net for r in results])


def __getattr__(name: str):
    """CHANNELS is built on first access: the channel modules import ChannelResult from here."""
    if name == "CHANNELS":
        from .depth import DepthChannel
        from .fold import FoldChannel
        from .laplacian import LaplacianChannel

        return {"laplacian": LaplacianChannel, "fold": FoldChannel, "depth": DepthChannel}
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
