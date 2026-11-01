"""Depth-edge channel (TASK_DEPTH_CHANNEL; docs/depth_channel.md): depth discontinuities found Canny-style in the
disparity, their metric jump Delta Z, and the weight w_depth.

D1: the parameters; the field (kernels, detrended orientation, q) is geom/candidates.py: depth_field. The candidates
(D-3), the jump, gates, magnitude and weights (D-4 ... D-7) and DepthChannel follow in D2 and D3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DepthParams:
    # D-K, D-1, D-2: field
    disparity: str = "rho_f"          # rho_f (the context's trilateral-filtered disparity) | rho (ablation)
    sigma: float = 1.25               # px, Hessian scale (as the Laplacian channel)
    truncate: float = 4.0
    sigma_w_prior: float = 1.0        # px, DA3 boundary-smear prior; sigma_tot = sqrt(sigma^2 + sigma_w_prior^2)
    sigma_L_factor: float = 4.0       # sigma_L = sigma_L_factor sigma_tot (detrending scale)
    k_g: float = 1.0                  # reject |grad Dd| < k_g median(|grad Dd| over Omega) (lean; calibrated in D2)
    # D-3: candidates
    lobe_half_len: float = 8.0        # px, lobe search along n
    lobe_step: float = 0.25           # px
    k_h: float = 3.0                  # two-sided height: min(|q_p|, |q_m|) > k_h s_q
    s_lo_factor: float = 0.8          # s_lo = s_lo_factor 2 sigma
    sigma_w_max: float = 3.0          # px, prior; calibrated in D2 (P95 of the boundary smear at occlusions)
    symmetry: bool = False            # |q_p| / |q_m| in [1/symmetry_ratio, symmetry_ratio]
    symmetry_ratio: float = 4.0
    k_min: int = 11                   # k_c = max(k_min, ceil(2 sigma_w_c) + 1)
    # D-4: jump
    m: int = 8
    w_t: int = 4
    m_min: int = 4
    max_excluded: float = 0.2
    planarity_factor: float = 3.0
    b_tri_margin: float = 1.25
    null_rms_quantile: float = 0.95
    gap_mode: str = "at_x0"           # at_x0 | min_span
    doublet_coef: float = 1.03        # Delta rho_doublet = coef |q|_lobe rho_sigma(x0) s_sep^2 (thin fallback)
    # D-5: gates
    tau_low: float = 0.02
    tau_high: float = 0.04
    min_component_px: int = 5
    z: float = 3.29                   # two-sided p_fa = 1e-3
    null_stride: int = 4
    lambda_clamp: tuple[float, float] = (0.5, 1.0e4)
    # D-6: magnitude
    smooth_n: int = 5                 # running median of rho_hat+- along contours
    sky_far_frac: float = 0.5         # far window at least this share sky -> Delta Z = inf
    canonical_focal: float = 300.0    # Z = f_n D / canonical_focal
    # D-7: weight
    dZ0: float = 0.10                 # m, knee of w_c = max(w_min, dZ0 / (dZ + dZ0))
    w_min: float = 0.05
    band_min: int = 2
