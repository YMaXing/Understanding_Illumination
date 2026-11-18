"""Per-pair metrics (DESIGN.md §2, §4 and the Checkpoint 0/1 deltas).

    rec, prof = measure(pair, ctx, cfg)      # tau-independent values: Δ vectors, norms, alpha, H, SEs, flags
    j = judge(rec, tau, cfg)                 # tau-dependent: low_snr, pass (plain / widened), bins, exclusions

ctx = dict(lI, lR, valid, I (raw DN), noise [(g, r2)] or None, geom_edge (bool map) or None, regions or None).

Kinds: i, m, b, s_random, s_targeted, d_R (same material; material and illumination labels both same/different),
d_L (same illumination, different material: drift diagnostic), d_unscored (other label combinations).
  E: i, s_*, d_R -> ||ΔℓR||;  m, d_L -> ||ΔℓL||;  b, d_unscored -> none.
  H: i, s_* -> H_R;  m -> H_L;  b, d_* -> none (d is endpoint-only).
Tolerances: plain = tau for E and H. Widened (Delta Q1) for the R-flat kinds (i, s_*, d_R): E <= max(tau,
k_tex * se_tex), H_R <= max(tau, H_I) [or H_ref with texture.profile_ref: iso]; m and d_L keep plain tau.
low_contrast (i, m): ||ΔℓI|| <= tau at the judged tau; excluded from Split (no split information). "pass" (the headline) is the widened rule.
Endpoint P(ℓL) := P(ℓI) - P(ℓR) and p_L := p_I - p_R (one degree of freedom per pixel, §1), so ΔℓL = ΔℓI - ΔℓR exactly.
Split fraction alpha = ΔℓL·ΔℓI / ΔℓI·ΔℓI, undefined when ||ΔℓI|| < alpha_min_factor * tau. Bins (failing i: alpha,
failing m: beta = 1 - alpha): reversed < -eps <= full leak < 0.2 <= partial <= 1 + eps < overshoot (eps =
thresholds.bin_eps, numerical slack for float32 predictions); n/a when undefined.
"""

import math

import numpy as np

from . import patches as PT
from . import profiles as PF

R_FLAT = {"i", "s_random", "s_targeted", "d_R"}
L_FLAT = {"m", "d_L"}
PROFILE_KINDS = {"i": "H_R", "s_random": "H_R", "s_targeted": "H_R", "m": "H_L"}
HEADLINE = {"i": "split", "m": "split", "s_random": "leak", "d_R": "long_range"}
BINS = ("reversed", "full_leak", "partial", "overshoot", "n/a")


def cn(v):
    return float(PF.channel_norm(np.asarray(v, float)))


def kind_of(p):
    t = p["type"]
    if t == "s":
        return f"s_{p['pool']}"
    if t == "d":
        lab = {"same", "different"}
        if p["material"] in lab and p["illumination"] in lab:
            if p["material"] == "same":
                return "d_R"
            if p["illumination"] == "same":
                return "d_L"
        return "d_unscored"
    return t


def _segment_hits(edge, A, B):
    """True if any pixel on the A-B segment (0.5 px steps, nearest pixel) is an edge pixel."""
    H, W = edge.shape
    T = math.hypot(B["x"] - A["x"], B["y"] - A["y"])
    n = max(int(math.ceil(2 * T)) + 1, 2)
    s = np.linspace(0, 1, n)
    xs = np.clip(np.rint(A["x"] + s * (B["x"] - A["x"])).astype(int), 0, W - 1)
    ys = np.clip(np.rint(A["y"] + s * (B["y"] - A["y"])).astype(int), 0, H - 1)
    return bool(edge[ys, xs].any())


def measure(pair, ctx, cfg):
    P, PR, PRS = cfg["patch"], cfg["profile"], cfg["pairs"]
    A, B = pair["A"], pair["B"]
    lI, lR, valid = ctx["lI"], ctx["lR"], ctx["valid"]
    sIA = PT.patch_stats(lI, valid, A["x"], A["y"], A["r"], P, I=ctx.get("I"), noise=ctx.get("noise"))
    sIB = PT.patch_stats(lI, valid, B["x"], B["y"], B["r"], P, I=ctx.get("I"), noise=ctx.get("noise"))
    sRA = PT.patch_stats(lR, valid, A["x"], A["y"], A["r"], P)
    sRB = PT.patch_stats(lR, valid, B["x"], B["y"], B["r"], P)
    PIA, PIB, PRA, PRB = sIA["median"], sIB["median"], sRA["median"], sRB["median"]
    PLA, PLB = PIA - PRA, PIB - PRB
    dI, dR = PIB - PIA, PRB - PRA
    dL = dI - dR
    nI, nR, nL = cn(dI), cn(dR), cn(dL)
    ii = float(dI @ dI)
    alpha = float(dL @ dI) / ii if ii > 0 and np.isfinite(ii) else float("nan")
    se_tex = PT.combined_se(sIA, sIB, "se")
    textured = bool(se_tex > cfg["texture"]["textured_se_tex"]) if np.isfinite(se_tex) else None
    st_tex = PR.get("sigma_t_textured_px")
    prof = PF.profiles(lI, lR, valid, A, B, PR, sigma_t=st_tex if (st_tex and textured) else None)
    H_I = PF.excursion(prof["I"], PIA, PIB)
    H_R = PF.excursion(prof["R"], PRA, PRB)
    H_L = PF.excursion(prof["L"], PLA, PLB)
    H_ref = PF.iso_excursion(prof["I"]) if cfg["texture"].get("profile_ref", "H_I") == "iso" else float("nan")
    kind = kind_of(pair)
    length = math.hypot(B["x"] - A["x"], B["y"] - A["y"])
    E_kind = "R" if kind in R_FLAT else ("L" if kind in L_FLAT else None)
    E = {"R": nR, "L": nL, None: float("nan")}[E_kind]
    H_key = PROFILE_KINDS.get(kind)
    H = {"H_R": H_R, "H_L": H_L, None: float("nan")}[H_key]
    se_noise = PT.combined_se(sIA, sIB, "se_noise") if "se_noise" in sIA else float("nan")
    near = pair["type"] in PRS["near_types"]
    out_of_range = (length > PRS["near_max_px"]) if near else (length < PRS["distant_min_px"])
    ge = ctx.get("geom_edge")
    regions = ctx.get("regions")
    if regions is not None and kind in ("i", "s_random", "s_targeted", "d_R"):
        det_split = bool(regions[A["y"], A["x"]] != regions[B["y"], B["x"]])
    else:
        det_split = None                                   # n/a: no region source (B2) or not applicable
    rec = dict(id=pair["id"], image=pair["image"], set=ctx.get("set"), annotator=pair["annotator"],
               type=pair["type"], kind=kind, confidence=pair["confidence"], used_in_loss=bool(pair.get("used_in_loss", False)),
               pool=pair.get("pool"), lighting=pair.get("lighting"), material=pair.get("material"),
               illumination=pair.get("illumination"), lit_end=pair.get("lit_end"),
               ax=A["x"], ay=A["y"], ar=A["r"], bx=B["x"], by=B["y"], br=B["r"], length_px=length,
               dI=dI.tolist(), dR=dR.tolist(), dL=dL.tolist(), nI=nI, nR=nR, nL=nL, alpha=alpha,
               PA_I=PIA.tolist(), PB_I=PIB.tolist(), PA_R=PRA.tolist(), PB_R=PRB.tolist(),
               H_I=H_I, H_R=H_R, H_L=H_L, H_ref=H_ref, E_kind=E_kind, E=E, H_kind=H_key, H=H,
               leaked_frac=(E / nI if kind in ("i", "m") and nI > 0 else float("nan")),
               se_noise=se_noise, se_tex=se_tex,
               fracA=sIA["frac_valid"], fracB=sIB["frac_valid"], invalid=not (sIA["ok"] and sIB["ok"]),
               geom_cross=(None if ge is None else _segment_hits(ge, A, B)), detector_split=det_split,
               length_out_of_range=bool(out_of_range),
               textured=textured, sigma_t=prof["sigma_t"])
    return rec, prof


def _bin(f, eps=0.0):
    """Bins of §4 with numerical slack eps at the 0 and 1 edges (config thresholds.bin_eps)."""
    if not np.isfinite(f):
        return "n/a"
    if f < -eps:
        return "reversed"
    if f < 0.2:
        return "full_leak"
    if f <= 1 + eps:
        return "partial"
    return "overshoot"


def _le(x, tol):
    return bool(np.isfinite(x) and x <= tol)


def judge(rec, tau, cfg):
    T = cfg["thresholds"]
    k = rec["kind"]
    low_snr = not (rec["se_noise"] <= tau / T["snr_divisor"])   # NaN (no noise model / empty patch) -> True
    alpha_defined = bool(np.isfinite(rec["alpha"]) and rec["nI"] >= T["alpha_min_factor"] * tau)
    alpha = rec["alpha"] if alpha_defined else float("nan")
    out = dict(tau=tau, low_snr=low_snr, alpha_defined=alpha_defined,
               input_excursion=bool(np.isfinite(rec["H_I"]) and rec["H_I"] > tau),   # §4 structured_path, renamed (D2)
               low_contrast=bool(k in ("i", "m") and rec["nI"] <= tau),               # Checkpoint 2: no split info
               tolE_plain=None, tolH_plain=None, tolE_wide=None, tolH_wide=None,
               pass_plain=None, pass_wide=None, bin_plain=None, bin_wide=None, both_step=None)
    if rec["E_kind"] is not None:
        tolE_w = tau
        tolH_w = tau
        if k in R_FLAT:
            st = rec["se_tex"]
            tolE_w = max(tau, cfg["texture"]["k_tex"] * st) if np.isfinite(st) else tau
            ref = rec["H_ref"] if cfg["texture"].get("profile_ref", "H_I") == "iso" else rec["H_I"]
            tolH_w = max(tau, ref) if np.isfinite(ref) else tau
        has_H = rec["H_kind"] is not None
        for name, tE, tH in (("plain", tau, tau), ("wide", tolE_w, tolH_w)):
            ok = _le(rec["E"], tE) and (not has_H or _le(rec["H"], tH))
            out[f"tolE_{name}"], out[f"tolH_{name}"] = tE, (tH if has_H else None)
            out[f"pass_{name}"] = ok
            if k in ("i", "m") and not ok:
                out[f"bin_{name}"] = _bin(alpha if k == "i" else 1.0 - alpha, T.get("bin_eps", 0.0))
    if k == "b":
        out["both_step"] = bool(rec["nR"] > tau and rec["nL"] > tau)
    ex = []
    if low_snr:
        ex.append("low_snr")
    if rec["invalid"]:
        ex.append("invalid")
    if rec["confidence"] == "unsure":
        ex.append("unsure")
    if out["low_contrast"]:
        ex.append("low_contrast")
    if k not in HEADLINE:
        ex.append({"b": "b_pair", "s_targeted": "targeted_s", "d_L": "d_same_illumination",
                   "d_unscored": "d_labels_unscored"}[k])
    out["exclusions"] = ex
    out["headline"] = HEADLINE.get(k) if not ex else None
    out["pass"] = out["pass_wide"]
    out["bin"] = out["bin_wide"]
    return out


def score_pairs(pairs, ctx, cfg, taus=None):
    """measure + judge at each tau (default: the config tau). Returns [(rec, {tau: judge}, prof)]."""
    taus = taus or [cfg["thresholds"]["tau"]]
    res = []
    for p in pairs:
        rec, prof = measure(p, ctx, cfg)
        res.append((rec, {t: judge(rec, t, cfg) for t in taus}, prof))
    return res
