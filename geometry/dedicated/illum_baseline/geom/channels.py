"""Channel results (docs/geometry_channels.md section 3.1). The Channel protocol, registry and combiner follow in F3."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ChannelResult:
    name: str
    w_net: np.ndarray                  # (H, W) in [w_min, 1], network res, upright
    strength: np.ndarray | None        # dimensionless per-pixel strength (fold: J at rasterized pixels, else 0)
    candidates: pd.DataFrame | None    # one row per candidate (all, with accept/reject reasons)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)  # saved as "<name>_<key>"
    scalars: dict[str, float] = field(default_factory=dict)
    w_min: float = 0.05
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)  # other per-image tables (fold: the null sample)
