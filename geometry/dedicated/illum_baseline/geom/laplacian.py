"""Laplacian channel: adapter around the existing detector (detector.compute_weights), unchanged.

Strength: q_pool / eps (dimensionless), so that w = max(w_min, 1 / (1 + strength)) where q_pool > 0.
"""

from __future__ import annotations

import numpy as np

from ..detector import Params, compute_weights
from .channels import ChannelResult
from .context import GeometryContext


class LaplacianChannel:
    name = "laplacian"

    def run(self, ctx: GeometryContext, params: dict | Params) -> ChannelResult:
        p = params if isinstance(params, Params) else Params(**params)
        out = compute_weights(ctx.D, ctx.sky_prob, p)
        sc = out["scalars"]
        eps = sc["eps"]
        strength = out["q_pool"] / eps if eps > 0 else np.where(out["q_pool"] > 0, np.inf, 0.0)
        return ChannelResult(name=self.name, w_net=out["w"], strength=strength, candidates=None,
                             arrays={"q": out["q"], "L_over_rho": out["L_over_rho"]}, scalars=sc, w_min=p.w_min)
