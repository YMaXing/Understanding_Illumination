"""Fold channel (F2.2-F2.8): surface-normal discontinuities from the dense normals and two-sided plane fits.

Per candidate (candidates.fold_candidates): rounding width -> skip k1 -> two-sided fit at k1 -> chi^2 gate on T ->
concentration veto C = dtheta(k1) / dtheta(k2) -> component size -> magnitude J = sin(min(dtheta, pi/2)) -> weight
w_c -> min-rasterized w_fold. Geometry only: no RGB.

Window contamination check (k1 windows and the null sample): rms(residual)/alpha must not exceed
max(planarity_factor eps_hat, b_tri_margin b_tri(k, m), Q_{null_rms_quantile} of the null windows' rms). The veto's k2
windows get no contamination check, only the exclusion rule.

Null models for the gate (all are always computed and recorded; null_model selects the one that gates):
  multiplicative: T = delta^T (Sigma_+ + Sigma_-)^-1 delta, accept T > lambda_hat 2 ln(1/p_fa),
                  lambda_hat = median(T_null) / (2 ln 2) clamped to lambda_clamp;
  additive:       T_add = delta^T (lambda_n (Sigma_+ + Sigma_-) + sigma_geo^2 I)^-1 delta, accept T_add > 2 ln(1/p_fa),
                  lambda_n the noise inflation from the synthetic Monte Carlo, sigma_geo solved per image so that
                  median(T_add over the null sample) = 2 ln 2 (0 if already below).
  additive_local: as additive, with sigma_geo solved per local_tile x local_tile px tile on the null samples inside
                  it (tiles with fewer than local_min_samples valid null samples take the global sigma_geo) and
                  interpolated bilinearly between tile centres.
"""

from __future__ import annotations

import dataclasses
import math
import time
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
    null_model: str = "multiplicative"  # multiplicative | additive | additive_local
    lambda_n: float = 93.2              # noise-only inflation for the additive nulls, k_min = 6 (data/lambda_n.json)
    local_tile: int = 48                # additive_local: tile size (px)
    local_min_samples: int = 50         # additive_local: fewer valid null samples in a tile -> global sigma_geo
    veto_k_factor: float = 2.0
    C_min: float = 0.85
    veto_mode: str = "fixed"            # fixed (k2 = 2 k1, C >= C_min) | capped (TASK_FOLD_V2 F7)
    veto_curv_coef: float = 0.4         # capped: C_min_eff = 1 - coef (1 - C_curv)
    veto_curv_max: float = 0.8          # capped: decidable iff C_curv <= this (task 0.7; calibrated 0.8, F7)
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


NULL_MODELS = ("multiplicative", "additive", "additive_local")


def _solve_sig2(d: np.ndarray, S: np.ndarray, lam_n: float) -> float:
    """sigma^2 >= 0 with median(delta^T (lam_n S + sigma^2 I)^-1 delta) = 2 ln 2 (bisection on log10 sigma^2)."""
    target = 2 * math.log(2)
    med = lambda a: float(np.median(quad_form(d, S, lam_n, a)))  # noqa: E731
    if med(0.0) <= target:
        return 0.0
    lo, hi = -40.0, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if med(10.0**mid) > target else (lo, mid)
    return 10.0 ** (0.5 * (lo + hi))


V2_VETO_KEYS = ("veto_mode", "veto_curv_coef", "veto_curv_max")


def params_record(p: dict) -> dict:
    """The fold parameters as recorded in the outputs: TASK_FOLD_V2 keys only when their feature is on, so runs with
    the new options off keep the records of the runs made before they existed."""
    d = dict(p)
    if d.get("veto_mode", "fixed") == "fixed":
        for k in V2_VETO_KEYS:
            d.pop(k, None)
    return d


def _J(dtheta: np.ndarray) -> np.ndarray:
    return np.sin(np.minimum(dtheta, np.pi / 2))


class FoldChannel:
    name = "fold"

    def run(self, ctx: GeometryContext, params: dict | FoldParams | None = None) -> ChannelResult:
        p = params if isinstance(params, FoldParams) else FoldParams.from_dict(params)
        H, W = ctx.shape
        fitkw = dict(m=p.m, w_t=p.w_t, m_min=p.m_min, max_excluded=p.max_excluded)
        tri = (ctx.det.tri_sigma_s, ctx.det.tri_sigma_r, ctx.det.tri_sigma_g)
        if p.null_model not in NULL_MODELS:
            raise ValueError(f"unknown null_model {p.null_model!r}; expected one of {NULL_MODELS}")
        t_start = time.perf_counter()
        ctx.normals  # shared context fields (timed separately: another channel may already have built them)
        t_ctx = time.perf_counter() - t_start

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
        # additive null: sigma_geo^2 such that median(T_add, null) = 2 ln 2; additive_local: per tile
        sig2 = float("nan")
        n_t, m_t = math.ceil(H / p.local_tile), math.ceil(W / p.local_tile)
        sig_tiles = np.full((n_t, m_t), np.nan)
        n_local = 0
        if np.isfinite(p.lambda_n) and T_null.size:
            dn, Sn = null_projection(null.p.nhat[nv], null.m.nhat[nv], null.p.Sigma_n[nv], null.m.Sigma_n[nv])
            sig2 = _solve_sig2(dn, Sn, p.lambda_n)
            sig_tiles[:] = math.sqrt(sig2)
            ti = (xn[nv, 1] // p.local_tile).astype(int)
            tj = (xn[nv, 0] // p.local_tile).astype(int)
            for i in range(n_t):
                for j in range(m_t):
                    m_ij = (ti == i) & (tj == j)
                    if m_ij.sum() >= p.local_min_samples:
                        sig_tiles[i, j] = math.sqrt(_solve_sig2(dn[m_ij], Sn[m_ij], p.lambda_n))
                        n_local += 1

        def sig_local(x: np.ndarray) -> np.ndarray:
            """Bilinear interpolation of the tile sigma_geo between tile centres at points x (N, 2) (u, v)."""
            c = (p.local_tile - 1) / 2
            coords = np.stack([(x[:, 1] - c) / p.local_tile, (x[:, 0] - c) / p.local_tile])
            return ndimage.map_coordinates(sig_tiles, coords, order=1, mode="nearest")

        # F2.3-F2.5 at k1
        r1 = two_sided_fit(ctx, x0, e1, k1, rms_max=rms_max, fallback=True, **fitkw)
        dtheta1 = r1.dtheta
        T_add, T_add_local = np.full(N, np.nan), np.full(N, np.nan)
        sig_loc = sig_local(x0) if np.isfinite(sig2) and N else np.full(N, np.nan)
        if np.isfinite(sig2) and r1.valid.any():
            v1 = r1.valid
            d1, S1 = null_projection(r1.p.nhat[v1], r1.m.nhat[v1], r1.p.Sigma_n[v1], r1.m.Sigma_n[v1])
            T_add[v1] = quad_form(d1, S1, p.lambda_n, sig2)
            T_add_local[v1] = quad_form(d1, S1, p.lambda_n, sig_loc[v1] ** 2)
        with np.errstate(invalid="ignore"):
            gate_mult = r1.valid & (r1.T > T_gate)
            gate_add = r1.valid & (T_add > T_crit)
            gate_add_local = r1.valid & (T_add_local > T_crit)
        if p.null_model != "multiplicative" and not np.isfinite(sig2):
            raise ValueError(f"null_model {p.null_model} needs a finite lambda_n and a non-empty null sample")
        gate = {"multiplicative": gate_mult, "additive": gate_add, "additive_local": gate_add_local}[p.null_model]
        sig_pair = np.sqrt(0.5 * (r1.p.sigma_theta**2 + r1.m.sigma_theta**2))  # per-axis sigma of delta
        dtheta_gate = np.sqrt(T_crit * lam) * sig_pair
        dtheta_gate_add = np.sqrt(T_crit * (p.lambda_n * sig_pair**2 + sig2)) if np.isfinite(sig2) else np.full(N, np.nan)
        dtheta_gate_add_local = np.sqrt(T_crit * (p.lambda_n * sig_pair**2 + sig_loc**2))

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
        if p.veto_mode not in ("fixed", "capped"):
            raise ValueError(f"unknown veto_mode {p.veto_mode!r}")
        capped = None
        if p.veto_mode == "capped":
            capped = self._capped_veto(ctx, p, cands, gate, x0, e1, width, k1, r1, dtheta1, eps_hat, tri, fitkw,
                                       rms_max)
            dtheta2, valid2, k2_clean, no_veto, C, veto_pass = (capped.pop(k) for k in (
                "dtheta2", "valid2", "k2_clean", "no_veto", "C", "veto_pass"))

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
        if capped is not None:
            reason[capped["veto_undecidable"]] = "veto_undecidable"

        df = pd.DataFrame({
            "u": u, "v": v, "u0": x0[:, 0], "v0": x0[:, 1], "e1_u": e1[:, 0], "e1_v": e1[:, 1],
            "r_n": cands.rate, "width": width, "profile_ok": profile_ok, "k1": k1, "k2": k2,
            **r1.side_fields(),
            **{f"n_{t}_{c}": s.nhat[:, i] for t, s in (("p", r1.p), ("m", r1.m)) for i, c in enumerate("xyz")},
            "valid": r1.valid, "dtheta": dtheta1, "T": r1.T, "T_over_lambda": r1.T / lam, "T_add": T_add,
            "T_add_local": T_add_local, "sigma_geo_local": sig_loc,
            "gate_mult": gate_mult, "gate_add": gate_add, "gate_add_local": gate_add_local, "gate": gate,
            "dtheta_gate": dtheta_gate, "dtheta_gate_add": dtheta_gate_add, "dtheta_gate_add_local": dtheta_gate_add_local,
            "dtheta_k2": dtheta2, "C": C, "no_veto": no_veto, "veto_pass": veto_pass, "k2_clean": k2_clean,
            "component_px": comp, "accepted": accepted, "reason": reason,
            "J": J, "chord": chord, "J_lo": J_lo, "J_hi": J_hi, "w_c": w_c, "band": band,
            "d_rho": r1.d_rho, "d_rho_se": r1.d_rho_se, "s_gap": r1.s_gap, "planes_cross": r1.planes_cross,
            "d_rho_x0": r1.d_rho_x0, "rho_p_gap": r1.rho_p_gap, "rho_m_gap": r1.rho_m_gap, "dZ_rel": r1.dZ_rel, "depth_jump": depth_jump, "convexity": convexity, "L_over_rho_x0": L0,
        })
        if capped is not None:  # capped-veto records (the fixed mode's table is unchanged)
            for k_ in ("k2_p", "k2_m", "d_next_p", "d_next_m", "C_curv", "C_min_eff", "decidable", "strict_planar",
                       "narrow_face", "veto_undecidable", "C_high"):
                df[k_] = capped[k_]
        T_null_add, T_null_add_local = np.full(len(xn), np.nan), np.full(len(xn), np.nan)
        if np.isfinite(sig2) and nv.any():
            T_null_add[nv] = quad_form(dn, Sn, p.lambda_n, sig2)
            T_null_add_local[nv] = quad_form(dn, Sn, p.lambda_n, sig_local(xn[nv]) ** 2)
        null_df = pd.DataFrame({"u": xn[:, 0], "v": xn[:, 1], "T": null.T, "T_add": T_null_add,
                                "T_add_local": T_null_add_local, "valid": null.valid,
                                "dtheta": null.dtheta, "fallback_p": null.p.fallback, "fallback_m": null.m.fallback})
        scalars = {
            "lambda_hat": lam, "lambda_hat_raw": lam_raw, "eps_hat": eps_hat, "null_rms_q": q_null,
            "T_gate": T_gate, "T_crit": T_crit, "null_model": p.null_model,
            "sigma_geo": math.sqrt(sig2) if np.isfinite(sig2) else float("nan"),
            "sigma_geo_local_tiles": int(n_local), "sigma_geo_local_min": float(np.nanmin(sig_tiles)) if n_local else float("nan"),
            "sigma_geo_local_max": float(np.nanmax(sig_tiles)) if n_local else float("nan"),
            "n_gate_mult": int(gate_mult.sum()), "n_gate_add": int(gate_add.sum()),
            "n_gate_add_local": int(gate_add_local.sum()),
            "rate_median": cands.median_rate, "rate_threshold": cands.threshold,
            "n_null": int(len(xn)), "n_null_valid": int(T_null.size),
            "n_cand": int(N), "n_valid": int(r1.valid.sum()), "n_gate": int(gate.sum()),
            "n_veto": int(veto_pass.sum()), "n_accepted": int(accepted.sum()), "n_no_veto": int(no_veto.sum()),
            "n_fallback": int((r1.valid & (r1.p.fallback | r1.m.fallback)).sum()),
            "n_accepted_depth_jump": int((accepted & depth_jump).sum()),
            "frac_w_lt_0.5": float((w_fold < 0.5).mean()), "frac_w_lt_0.9": float((w_fold < 0.9).mean()),
            "params": params_record(dataclasses.asdict(p)),
        }
        if capped is not None:
            scalars |= {"veto_mode": "capped", "n_narrow_face": int(capped["narrow_face"].sum()),
                        "n_veto_undecidable": int(capped["veto_undecidable"].sum()),
                        "n_C_high": int(capped["C_high"].sum())}
        scalars["context_s"] = t_ctx
        scalars["runtime_s"] = time.perf_counter() - t_start - t_ctx
        return ChannelResult(name=self.name, w_net=w_fold, strength=strength, candidates=df,
                             arrays={"r_n": cands.rate_map}, scalars=scalars, w_min=p.w_min,
                             tables={"null": null_df})

    @staticmethod
    def _capped_veto(ctx, p, cands, gate, x0, e1, width, k1, r1, dtheta1, eps_hat, tri, fitkw, rms_max) -> dict:
        """TASK_FOLD_V2 F7: the veto with k2 capped by the next obstacle on each side.

        d_next+- = distance along +-e1 from x0 to the first obstacle, checked over the window's width (|t| <= w_t,
        nearest pixel): an invalid normal or the outside of the image beyond the candidate's own rounding
        (|s| > width/2 + 1), or another gate-passed candidate (NMS pixels dilated by 1 px) beyond the whole rounding
        band (|s| > width + 1: a rounded fold has parallel sibling ridges inside its band, which are not a next face).
        k2+- = clamp(min(veto_k_factor k1, d_next+- - m - 2), k1, veto_k_factor k1);
        C_curv = (2 k1 + m) / (k2+ + k2- + m) (the constant-curvature ratio for these skips);
        C_min_eff = 1 - veto_curv_coef (1 - C_curv); decidable iff C_curv <= veto_curv_max.
        Decidable and both k2 windows valid: accept iff C >= C_min_eff (C_high: C > 1 / C_min_eff, a flag).
        Otherwise accept only if both k1 sides are strictly planar (planar fit, not fallback, rms/alpha <=
        max(3 eps_hat, b_tri_margin b_tri(k1, m_used))): narrow_face; else reject: veto_undecidable.
        Records: no_veto (column, n_no_veto) keeps its name but means here "gate-passed with an invalid capped k2
        window"; such candidates are never accepted on that ground (they go to the narrow_face rule).
        """
        H, W = ctx.shape
        N = len(x0)
        gp = np.zeros((H, W), dtype=bool)
        gp[cands.ij[gate, 1], cands.ij[gate, 0]] = True
        gpd = ndimage.binary_dilation(gp, structure=np.ones((3, 3), dtype=bool))
        bad = ~ctx.normal_valid
        k_cap = np.rint(p.veto_k_factor * k1).astype(int)
        g = np.nonzero(gate)[0]
        d_next = {+1: np.full(N, np.inf), -1: np.full(N, np.inf)}
        if g.size:
            et = np.stack([-e1[g, 1], e1[g, 0]], -1)
            t = np.arange(-p.w_t, p.w_t + 1, dtype=np.float64)
            s_lo = width[g] / 2 + 1     # invalid normals, outside
            s_lo_c = width[g] + 1       # other gate-passed candidates
            s_hi = float((k_cap[g] + p.m + 2).max())
            for sign in (+1, -1):
                d = np.full(g.size, np.inf)
                for sv in np.arange(0.5, s_hi + 0.51, 0.5):
                    active = np.isinf(d) & (sv > s_lo)
                    if not active.any():
                        continue
                    pts = (x0[g, None, :] + sign * sv * e1[g, None, :] + t[None, :, None] * et[:, None, :])
                    uu, vv = np.rint(pts[..., 0]).astype(int), np.rint(pts[..., 1]).astype(int)
                    inside = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                    uc, vc = np.clip(uu, 0, W - 1), np.clip(vv, 0, H - 1)
                    hit = (~inside | bad[vc, uc]).any(1) | ((sv > s_lo_c) & gpd[vc, uc].any(1))
                    d[active & hit] = sv
                d_next[sign][g] = d
        cap = lambda dn: np.clip(np.minimum(k_cap, np.floor(dn - p.m - 2)), k1, k_cap).astype(int)  # noqa: E731
        with np.errstate(invalid="ignore"):
            k2p, k2m = cap(d_next[+1]), cap(d_next[-1])
        C_curv = (2 * k1 + p.m) / (k2p + k2m + p.m)
        C_min_eff = 1 - p.veto_curv_coef * (1 - C_curv)
        decidable = C_curv <= p.veto_curv_max

        dtheta2 = np.full(N, np.nan)
        valid2 = np.zeros(N, dtype=bool)
        k2_clean = np.zeros(N, dtype=bool)
        if g.size:
            r2 = two_sided_fit(ctx, x0[g], e1[g], k2p[g], rms_max=None, fallback=False, k_m=k2m[g], **fitkw)
            dtheta2[g], valid2[g] = r2.dtheta, r2.valid
            with np.errstate(invalid="ignore"):
                k2_clean[g] = (r2.valid & (r2.p.rms_rel <= rms_max(k2p[g].astype(float), p.m))
                               & (r2.m.rms_rel <= rms_max(k2m[g].astype(float), p.m)))

        def strict_tol(side) -> np.ndarray:
            bt = np.where(side.m_used == 4, b_tri(k1, 4, tri), b_tri(k1, p.m, tri))
            return np.fmax(p.planarity_factor * eps_hat, p.b_tri_margin * bt)

        with np.errstate(invalid="ignore", divide="ignore"):
            strict = (r1.p.planar & r1.m.planar & (r1.p.rms_rel <= strict_tol(r1.p))
                      & (r1.m.rms_rel <= strict_tol(r1.m)))
            C = dtheta1 / dtheta2
            decided = gate & decidable & valid2
            undecided = gate & ~(decidable & valid2)
            veto_pass = (decided & (C >= C_min_eff)) | (undecided & strict)
            C_high = decided & (C > 1 / C_min_eff)
        return {"dtheta2": dtheta2, "valid2": valid2, "k2_clean": k2_clean, "no_veto": gate & ~valid2, "C": C,
                "veto_pass": veto_pass, "k2_p": np.where(gate, k2p, -1), "k2_m": np.where(gate, k2m, -1),
                "d_next_p": d_next[+1], "d_next_m": d_next[-1], "C_curv": np.where(gate, C_curv, np.nan),
                "C_min_eff": np.where(gate, C_min_eff, np.nan), "decidable": gate & decidable,
                "strict_planar": strict, "narrow_face": undecided & strict,
                "veto_undecidable": undecided & ~strict, "C_high": C_high}
