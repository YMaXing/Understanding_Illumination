"""Fold channel (F2.2-F2.8): surface-normal discontinuities from the dense normals and two-sided plane fits.

Per candidate (candidates.fold_candidates): rounding width -> skip k1 -> two-sided fit at k1 -> chi^2 gate on T ->
concentration veto C = dtheta(k1) / dtheta(k2) -> component size -> magnitude J = sin(min(dtheta, pi/2)) -> weight
w_c -> min-rasterized w_fold. Geometry only: no RGB.

Window contamination check (k1 windows and the null sample): rms(residual)/alpha must not exceed
max(planarity_factor eps_hat, b_tri_margin b_tri(k, m), Q_{null_rms_quantile} of the null windows' rms). The veto's k2
windows get no contamination check, only the exclusion rule.

Null models for the gate (both are always computed and recorded; null_model selects the one that gates):
  multiplicative: T = delta^T (Sigma_+ + Sigma_-)^-1 delta, accept T > lambda_hat 2 ln(1/p_fa),
                  lambda_hat = median(T_null) / (2 ln 2) clamped to lambda_clamp;
  additive:       T_add = delta^T (lambda_n (Sigma_+ + Sigma_-) + sigma_geo^2 I)^-1 delta, accept T_add > 2 ln(1/p_fa),
                  lambda_n the noise inflation from the synthetic Monte Carlo, sigma_geo solved per image so that
                  median(T_add over the null sample) = 2 ln 2 (0 if already below).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from ..detector import disk
from .candidates import fold_candidates, rounding_width
from .channels import ChannelResult
from .context import GeometryContext
from .normals import focal_bracket, normals
from .sampling import bilinear
from .twosided import b_tri, fold_angle, null_projection, quad_form, two_sided_fit, window_rms


@dataclass(frozen=True)
class FoldParams:
    cand_sigma_d: float = 1.0
    cand_sigma_T: float = 1.5
    cand_rate_factor: float = 3.0
    profile_half_len: int = 12
    width_min: float = 2.0
    width_max: float = 12.0
    k_min: int = 6
    k_margin: int = 2
    m: int = 8
    m_min: int = 4
    w_t: int = 4
    max_excluded: float = 0.2
    planarity_factor: float = 3.0
    b_tri_margin: float = 1.25        # safety factor on the measured trilateral contamination envelope
    null_rms_quantile: float = 0.95   # per-image quantile of the null windows' rms in the contamination threshold
    p_fa: float = 1.0e-3
    null_stride: int = 4
    lambda_clamp: tuple[float, float] = (0.5, 1.0e4)
    null_model: str = "multiplicative"  # multiplicative | additive (additive failed the 079 sunlit-grass ROI rule)
    lambda_n: float = 93.2              # noise-only inflation for the additive null, k_min = 6 (data/lambda_n.json)
    veto_k_factor: float = 2.0
    C_min: float = 0.85
    J_min: float = 0.0872
    J0: float = 0.25
    w_min: float = 0.05
    band_min: int = 2
    min_component_px: int = 5
    depth_jump_rel: float = 0.02

    @classmethod
    def from_dict(cls, d: dict | None) -> FoldParams:
        d = dict(d or {})
        if "lambda_clamp" in d:
            d["lambda_clamp"] = tuple(d["lambda_clamp"])
        return cls(**d)


def _J(dtheta: np.ndarray) -> np.ndarray:
    return np.sin(np.minimum(dtheta, np.pi / 2))


class FoldChannel:
    name = "fold"

    def run(self, ctx: GeometryContext, params: dict | FoldParams | None = None) -> ChannelResult:
        p = params if isinstance(params, FoldParams) else FoldParams.from_dict(params)
        H, W = ctx.shape
        fitkw = dict(m=p.m, w_t=p.w_t, m_min=p.m_min, max_excluded=p.max_excluded)
        tri = (ctx.det.tri_sigma_s, ctx.det.tri_sigma_r, ctx.det.tri_sigma_g)
        if p.null_model not in ("multiplicative", "additive"):
            raise ValueError(f"unknown null_model {p.null_model!r}")

        # F2.1 candidates, F2.2 rounding width and skip
        cands = fold_candidates(ctx, p.cand_sigma_d, p.cand_sigma_T, p.cand_rate_factor)
        x0, e1, N = cands.x0, cands.e1, len(cands.x0)
        width, profile_ok = rounding_width(ctx, x0, e1, p.profile_half_len, p.width_min, p.width_max)
        k1 = np.maximum(p.k_min, np.ceil(width / 2).astype(int) + p.k_margin)

        # F2.3 contamination threshold from the null grid (each pixel's own e1, k = k_min), no check in this pass
        s0 = p.null_stride // 2
        vv, uu = np.mgrid[s0:H:p.null_stride, s0:W:p.null_stride]
        sel = cands.valid_map[vv, uu]
        xn = np.stack([uu[sel], vv[sel]], -1).astype(np.float64)
        en = cands.e1_map[vv[sel], uu[sel]]
        cal_rms = window_rms(two_sided_fit(ctx, xn, en, p.k_min, rms_max=None, fallback=False, **fitkw))
        eps_hat = float(np.median(cal_rms)) if cal_rms.size else float("nan")
        q_null = float(np.quantile(cal_rms, p.null_rms_quantile)) if cal_rms.size else float("nan")

        def rms_max(k: np.ndarray, m: int) -> np.ndarray:
            return np.fmax(np.fmax(p.planarity_factor * eps_hat, q_null), p.b_tri_margin * b_tri(k, m, tri))

        # F2.5 null sample through the same pipeline as the candidates
        null = two_sided_fit(ctx, xn, en, p.k_min, rms_max=rms_max, fallback=True, **fitkw)
        nv = null.valid & ~np.isnan(null.T)
        T_null = null.T[nv]
        lam_raw = float(np.median(T_null) / (2 * math.log(2))) if T_null.size else float("nan")
        lam = float(np.clip(lam_raw, *p.lambda_clamp)) if T_null.size else 1.0
        T_crit = 2 * math.log(1 / p.p_fa)
        T_gate = lam * T_crit
        # additive null: sigma_geo^2 such that median(T_add, null) = 2 ln 2
        sig2 = float("nan")
        if np.isfinite(p.lambda_n) and T_null.size:
            dn, Sn = null_projection(null.p.nhat[nv], null.m.nhat[nv], null.p.Sigma_n[nv], null.m.Sigma_n[nv])
            target = 2 * math.log(2)
            med = lambda a: float(np.median(quad_form(dn, Sn, p.lambda_n, a)))  # noqa: E731
            if med(0.0) <= target:
                sig2 = 0.0
            else:
                lo, hi = -40.0, 2.0  # log10 rad^2
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if med(10.0**mid) > target else (lo, mid)
                sig2 = 10.0 ** (0.5 * (lo + hi))

        # F2.3-F2.5 at k1
        r1 = two_sided_fit(ctx, x0, e1, k1, rms_max=rms_max, fallback=True, **fitkw)
        dtheta1 = r1.dtheta
        T_add = np.full(N, np.nan)
        if np.isfinite(sig2) and r1.valid.any():
            d1, S1 = null_projection(r1.p.nhat[r1.valid], r1.m.nhat[r1.valid], r1.p.Sigma_n[r1.valid],
                                     r1.m.Sigma_n[r1.valid])
            T_add[r1.valid] = quad_form(d1, S1, p.lambda_n, sig2)
        with np.errstate(invalid="ignore"):
            gate_mult = r1.valid & (r1.T > T_gate)
            gate_add = r1.valid & (T_add > T_crit)
        if p.null_model == "additive" and not np.isfinite(sig2):
            raise ValueError("null_model additive needs a finite lambda_n and a non-empty null sample")
        gate = gate_add if p.null_model == "additive" else gate_mult
        sig_pair = np.sqrt(0.5 * (r1.p.sigma_theta**2 + r1.m.sigma_theta**2))  # per-axis sigma of delta
        dtheta_gate = np.sqrt(T_crit * lam) * sig_pair
        dtheta_gate_add = np.sqrt(T_crit * (p.lambda_n * sig_pair**2 + sig2)) if np.isfinite(sig2) else np.full(N, np.nan)

        # F2.6 concentration veto: k2 windows without the contamination check (exclusion rule only); sides without
        # a planar fit count as invalid (no_veto)
        k2 = np.rint(p.veto_k_factor * k1).astype(int)
        dtheta2 = np.full(N, np.nan)
        valid2 = np.zeros(N, dtype=bool)
        k2_clean = np.zeros(N, dtype=bool)  # record: both k2 windows would also pass the contamination check
        g = np.nonzero(gate)[0]
        if g.size:
            r2 = two_sided_fit(ctx, x0[g], e1[g], k2[g], rms_max=None, fallback=False, **fitkw)
            dtheta2[g], valid2[g] = r2.dtheta, r2.valid
            tol2 = rms_max(k2[g].astype(float), p.m)
            with np.errstate(invalid="ignore"):
                k2_clean[g] = r2.valid & (r2.p.rms_rel <= tol2) & (r2.m.rms_rel <= tol2)
        no_veto = gate & ~valid2
        with np.errstate(invalid="ignore", divide="ignore"):
            C = dtheta1 / dtheta2
            veto_pass = gate & (no_veto | (C >= p.C_min))

        # F2.8 component size (8-connected, on the NMS pixels)
        u, v = cands.ij[:, 0], cands.ij[:, 1]
        acc = np.zeros((H, W), dtype=bool)
        acc[v[veto_pass], u[veto_pass]] = True
        lab, _ = ndimage.label(acc, structure=np.ones((3, 3), dtype=bool))
        sizes = np.bincount(lab.ravel())
        comp = np.where(veto_pass, sizes[lab[v, u]], 0)
        accepted = veto_pass & (comp >= p.min_component_px)

        # F2.7 magnitude, focal diagnostics (never used in w)
        J = _J(dtheta1)
        chord = 2 * np.sin(dtheta1 / 2)
        Jb = []
        for Kb in focal_bracket(ctx.K):
            nb = [normals(s.alpha, s.beta, np.zeros((N, 2, 2)), s.c[:, 0], s.c[:, 1], Kb).n_hat for s in (r1.p, r1.m)]
            Jb.append(_J(fold_angle(*nb)))
        J_lo, J_hi = np.fmin(*Jb), np.fmax(*Jb)

        # F2.8 weight and rasterization
        with np.errstate(invalid="ignore"):
            w_c = np.maximum(p.w_min, p.J0 / (np.maximum(J - p.J_min, 0.0) + p.J0))
        band = np.maximum(p.band_min, np.ceil(width / 2).astype(int))
        w_fold = np.ones((H, W))
        strength = np.zeros((H, W))
        for b in np.unique(band[accepted]):
            a = accepted & (band == b)
            wi = np.ones((H, W))
            np.minimum.at(wi, (v[a], u[a]), w_c[a])
            si = np.zeros((H, W))
            np.maximum.at(si, (v[a], u[a]), J[a])
            fp = disk(int(b))
            w_fold = np.minimum(w_fold, ndimage.grey_erosion(wi, footprint=fp, mode="constant", cval=1.0))
            strength = np.maximum(strength, ndimage.grey_dilation(si, footprint=fp, mode="constant", cval=0.0))

        # records: convex (+1: disparity ridge, L < 0) / concave (-1), depth-jump flag
        L0, okL = bilinear(ctx.L_over_rho_f, ctx.omega, x0[:, 0], x0[:, 1])
        convexity = np.where(okL, -np.sign(L0), 0).astype(int)
        with np.errstate(invalid="ignore", divide="ignore"):
            depth_jump = (r1.dZ_rel > p.depth_jump_rel) & (np.abs(r1.d_rho) / r1.d_rho_se > 3)

        reason = np.full(N, "accepted", dtype=object)
        reason[~(veto_pass & (comp >= p.min_component_px))] = "component"
        reason[~veto_pass] = "veto"
        reason[~gate] = "gate"
        reason[~r1.valid] = "invalid_window"

        df = pd.DataFrame({
            "u": u, "v": v, "u0": x0[:, 0], "v0": x0[:, 1], "e1_u": e1[:, 0], "e1_v": e1[:, 1],
            "r_n": cands.rate, "width": width, "profile_ok": profile_ok, "k1": k1, "k2": k2,
            **r1.side_fields(),
            **{f"n_{t}_{c}": s.nhat[:, i] for t, s in (("p", r1.p), ("m", r1.m)) for i, c in enumerate("xyz")},
            "valid": r1.valid, "dtheta": dtheta1, "T": r1.T, "T_over_lambda": r1.T / lam, "T_add": T_add,
            "gate_mult": gate_mult, "gate_add": gate_add, "gate": gate,
            "dtheta_gate": dtheta_gate, "dtheta_gate_add": dtheta_gate_add,
            "dtheta_k2": dtheta2, "C": C, "no_veto": no_veto, "veto_pass": veto_pass, "k2_clean": k2_clean,
            "component_px": comp, "accepted": accepted, "reason": reason,
            "J": J, "chord": chord, "J_lo": J_lo, "J_hi": J_hi, "w_c": w_c, "band": band,
            "d_rho": r1.d_rho, "d_rho_se": r1.d_rho_se, "s_gap": r1.s_gap, "planes_cross": r1.planes_cross,
            "d_rho_x0": r1.d_rho_x0, "rho_p_gap": r1.rho_p_x0, "rho_m_gap": r1.rho_m_x0, "dZ_rel": r1.dZ_rel, "depth_jump": depth_jump, "convexity": convexity, "L_over_rho_x0": L0,
        })
        T_null_add = np.full(len(xn), np.nan)
        if np.isfinite(sig2) and nv.any():
            T_null_add[nv] = quad_form(dn, Sn, p.lambda_n, sig2)
        null_df = pd.DataFrame({"u": xn[:, 0], "v": xn[:, 1], "T": null.T, "T_add": T_null_add, "valid": null.valid,
                                "dtheta": null.dtheta, "fallback_p": null.p.fallback, "fallback_m": null.m.fallback})
        scalars = {
            "lambda_hat": lam, "lambda_hat_raw": lam_raw, "eps_hat": eps_hat, "null_rms_q": q_null,
            "T_gate": T_gate, "sigma_geo": math.sqrt(sig2) if np.isfinite(sig2) else float("nan"),
            "n_gate_mult": int(gate_mult.sum()), "n_gate_add": int(gate_add.sum()),
            "rate_median": cands.median_rate, "rate_threshold": cands.threshold,
            "n_null": int(len(xn)), "n_null_valid": int(T_null.size),
            "n_cand": int(N), "n_valid": int(r1.valid.sum()), "n_gate": int(gate.sum()),
            "n_veto": int(veto_pass.sum()), "n_accepted": int(accepted.sum()), "n_no_veto": int(no_veto.sum()),
            "n_fallback": int((r1.valid & (r1.p.fallback | r1.m.fallback)).sum()),
            "n_accepted_depth_jump": int((accepted & depth_jump).sum()),
            "frac_w_lt_0.5": float((w_fold < 0.5).mean()), "frac_w_lt_0.9": float((w_fold < 0.9).mean()),
            "params": dataclasses.asdict(p),
        }
        return ChannelResult(name=self.name, w_net=w_fold, strength=strength, candidates=df,
                             arrays={"r_n": cands.rate_map}, scalars=scalars, w_min=p.w_min,
                             tables={"null": null_df})
