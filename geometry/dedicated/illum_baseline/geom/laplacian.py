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
        hys = None
        if isinstance(params, dict) and params.get("hysteresis", "off") != "off":
            params = dict(params)
            hys = {k: params.pop(k) for k in ("hysteresis", "hysteresis_kappa", "hysteresis_max_gap")}
        elif isinstance(params, dict):
            params = {k: v for k, v in params.items() if not k.startswith("hysteresis")}
        p = params if isinstance(params, Params) else Params(**params)
        out = compute_weights(ctx.D, ctx.sky_prob, p)
        sc = out["scalars"]
        eps = sc["eps"]
        arrays = {"q": out["q"], "L_over_rho": out["L_over_rho"]}
        w, q_pool = out["w"], out["q_pool"]
        if hys is not None:  # TASK_GEOM_V2 W2 (off: nothing here runs and every output is unchanged)
            if hys["hysteresis"] != "ridge_bridge":
                raise ValueError(f"unknown hysteresis {hys['hysteresis']!r}; expected off or ridge_bridge")
            from scipy import ndimage

            from ..detector import disk
            from .hysteresis import ridge_hysteresis

            h = ridge_hysteresis(out["q"], sc["tau"], hys["hysteresis_kappa"], hys["hysteresis_max_gap"])
            q_eff = np.maximum(out["q_eff"], h["q_cont"])
            r_pool = sc["r"] if p.pool_radius is None else p.pool_radius
            q_pool = ndimage.grey_dilation(q_eff, footprint=disk(r_pool), mode="constant", cval=0.0)
            w = np.maximum(p.w_min, eps / (q_pool + eps)) if eps > 0 else np.where(q_pool > 0, p.w_min, 1.0)
            w[q_pool == 0] = 1.0
            arrays["continued"] = h["bridged"]
            r = h["runs"]
            sc = sc | {"hysteresis": "ridge_bridge", "hysteresis_kappa": hys["hysteresis_kappa"],
                       "hysteresis_max_gap": hys["hysteresis_max_gap"], "n_ridge_px": int(h["ridge"].sum()),
                       "n_seed_px": int(h["seed"].sum()), "n_continuation_px": int(h["cont"].sum()),
                       "n_continuation_runs": int(len(r)), "n_bridged_runs": int(r.bridged.sum()),
                       "n_bridged_px": int(h["bridged"].sum()),
                       "frac_w_lt_0.5_before_hysteresis": sc["frac_w_lt_0.5"],
                       "frac_w_lt_0.5": float((w < 0.5).mean()), "frac_w_lt_0.9": float((w < 0.9).mean()),
                       "frac_w_lt_1": float((w < 1).mean())}
        strength = q_pool / eps if eps > 0 else np.where(q_pool > 0, np.inf, 0.0)
        return ChannelResult(name=self.name, w_net=w, strength=strength, candidates=None,
                             arrays=arrays, scalars=sc, w_min=p.w_min)
