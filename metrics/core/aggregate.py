"""Aggregation (DESIGN.md §5 + deltas). Rows: dict(rec=..., J={tau: judge}) for ONE method.

Headline numbers (widened rule = "pass"): Split = mean(ER_i, ER_m), Leak = ER_s_random, Long-range = ER_d_R, each
micro-averaged over eligible pairs (no exclusion at that tau). A rate over zero pairs is "n/a: 0 pairs"; Split is
n/a if either term is. The optional composite (mean of the three) is n/a unless all three exist, and is never shown
alone. CIs: cluster bootstrap over images (resample images with replacement, keep all their pairs), fixed seed;
resamples where a rate is undefined (no eligible pair drawn) are dropped and counted.
"""

from collections import Counter, defaultdict

import numpy as np

from . import pair_metrics as PM
from . import summary as SM

HEAD_KINDS = {"split": ("i", "m"), "leak": ("s_random",), "long_range": ("d_R",)}


def elig(r, tau):
    return not r["J"][tau]["exclusions"]


def fails(r, tau, rule="wide"):
    return not r["J"][tau][f"pass_{rule}"]


def kind_rate(rows, kind, tau, rule="wide", pred=None):
    sel = [r for r in rows if r["rec"]["kind"] == kind and elig(r, tau) and (pred is None or pred(r))]
    return SM.rate(fails(r, tau, rule) for r in sel)


def headline(rows, tau, rule="wide"):
    er = {k: kind_rate(rows, k, tau, rule) for k in ("i", "m", "s_random", "d_R")}
    split = None if er["i"]["rate"] is None or er["m"]["rate"] is None else (er["i"]["rate"] + er["m"]["rate"]) / 2
    vals = dict(split=split, leak=er["s_random"]["rate"], long_range=er["d_R"]["rate"])
    comp = None if any(v is None for v in vals.values()) else float(np.mean(list(vals.values())))
    txt = lambda v, why: why if v is None else f"{v:.3f}"
    return dict(ER=er, split=split, leak=vals["leak"], long_range=vals["long_range"], composite=comp,
                text=dict(split=txt(split, "n/a: " + ("0 i pairs" if er["i"]["rate"] is None else "0 m pairs")
                                    if split is None else ""),
                          leak=txt(vals["leak"], "n/a: 0 pairs"), long_range=txt(vals["long_range"], "n/a: 0 pairs"),
                          composite=txt(comp, "n/a: a headline number is n/a")))


def headline_fixed_pop(rows, tau, pop_tau, rule="wide"):
    """Headline at tau, on the pair population eligible at pop_tau (separates the threshold from the population)."""
    fixed = [dict(rec=r["rec"], J={tau: dict(r["J"][tau], exclusions=r["J"][pop_tau]["exclusions"])}) for r in rows]
    return headline(fixed, tau, rule)


def texture_strata_counts(rows, tau, cfg):
    """Eligible i / m pairs per texture stratum (smooth: se_tex <= textured_se_tex; moderate; strong > strong_se_tex)."""
    X = cfg["texture"]
    out = {k: dict(smooth=0, moderate=0, strong=0) for k in ("i", "m")}
    for r in rows:
        k = r["rec"]["kind"]
        if k in out and elig(r, tau):
            s = r["rec"]["se_tex"]
            out[k]["smooth" if not s > X["textured_se_tex"] else ("strong" if s > X["strong_se_tex"] else "moderate")] += 1
    return out


def gt_floor(counts, ref, option):
    """Approximate perfect-decomposition Split floor from the synthetic Monte Carlo (ref = texture_mc_d1.json).
    i: smooth 0 (smooth material) .. the not-textured-on-textured-material rate; moderate = grey 0.10 textured rate;
    strong = grey 0.20 textured rate. m: the Monte Carlo m rates (textured / not)."""
    F = ref["floor"]
    rate = lambda g, kind, tex: (lambda v: v["fail_wide"] / v["n"] if v and v["n"] else 0.0)(
        F.get(g, {}).get(f"{option}|{kind}|{'textured' if tex else 'not_textured'}"))
    gm, gs = "0.1", "0.2"
    out = {}
    for lo_hi in ("lo", "hi"):
        er = {}
        for k in ("i", "m"):
            c = counts[k]
            n = sum(c.values())
            if not n:
                er[k] = None
                continue
            smooth = rate(gm, k, False) if lo_hi == "hi" else 0.0
            er[k] = (c["smooth"] * smooth + c["moderate"] * rate(gm, k, True) + c["strong"] * rate(gs, k, True)) / n
        out[lo_hi] = dict(ER=er, split=None if None in er.values() else (er["i"] + er["m"]) / 2)
    return out


def bootstrap(rows, tau, images, B, seed, rule="wide"):
    """Percentile 95% CIs for ER per headline kind and the headline numbers; cluster = image."""
    images = sorted(images)
    ix = {s: k for k, s in enumerate(images)}
    kinds = ("i", "m", "s_random", "d_R")
    n = np.zeros((len(kinds), len(images)))
    f = np.zeros_like(n)
    for r in rows:
        k = r["rec"]["kind"]
        if k in kinds and elig(r, tau) and r["rec"]["image"] in ix:
            n[kinds.index(k), ix[r["rec"]["image"]]] += 1
            f[kinds.index(k), ix[r["rec"]["image"]]] += fails(r, tau, rule)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(images), size=(B, len(images)))
    N = n[:, idx].sum(-1)                         # (kinds, B)
    F = f[:, idx].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ER = np.where(N > 0, F / N, np.nan)
    est = dict(zip(kinds, ER))
    est["split"] = (ER[0] + ER[1]) / 2
    est["leak"], est["long_range"] = ER[2], ER[3]
    out = {}
    for k, v in est.items():
        ok = np.isfinite(v)
        out[k] = dict(lo=float(np.percentile(v[ok], 2.5)) if ok.any() else None,
                      hi=float(np.percentile(v[ok], 97.5)) if ok.any() else None,
                      n_resamples=int(ok.sum()), n_undefined=int((~ok).sum()))
    return dict(B=B, seed=seed, n_images=len(images), ci=out)


def exclusion_counts(rows, tau):
    c = defaultdict(Counter)
    for r in rows:
        for e in r["J"][tau]["exclusions"]:
            c[r["rec"]["kind"]][e] += 1
        c[r["rec"]["kind"]]["_total"] += 1
        c[r["rec"]["kind"]]["_eligible"] += elig(r, tau)
    return {k: dict(v) for k, v in c.items()}


def companions(rows, tau):
    out = {}
    for k in SM.RATED:
        sel = [r["rec"] for r in rows if r["rec"]["kind"] == k and elig(r, tau)]
        E = np.array([x["E"] for x in sel], float)
        H = np.array([x["H"] for x in sel], float)
        st = lambda a: dict(mean=float(np.nanmean(a)), median=float(np.nanmedian(a))) if np.isfinite(a).any() else None
        d = dict(n=len(sel), E=st(E), H=st(H))
        if k in ("i", "m"):
            lf = np.array([x["leaked_frac"] for x in sel], float)
            d["leaked_frac_mean"] = float(np.nanmean(lf)) if np.isfinite(lf).any() else None
        out[k] = d
    return out


def bins(rows, tau, rule="wide"):
    out = {}
    for k in ("i", "m"):
        c = Counter(r["J"][tau][f"bin_{rule}"] for r in rows
                    if r["rec"]["kind"] == k and elig(r, tau) and r["J"][tau][f"bin_{rule}"])
        out[k] = {b: c[b] for b in PM.BINS}
    return out


def b_diag(rows, tau):
    sel = [r for r in rows if r["rec"]["kind"] == "b" and not r["rec"]["invalid"] and not r["J"][tau]["low_snr"]]
    a = np.array([r["rec"]["alpha"] for r in sel], float)
    return dict(n=len(sel), both_step=SM.rate(r["J"][tau]["both_step"] for r in sel),
                alpha_quartiles=np.round(np.nanpercentile(a, [25, 50, 75]), 3).tolist() if np.isfinite(a).any() else None)


def strata(rows, tau, kind, key):
    """ER of an eligible kind split by key(r) -> stratum label."""
    g = defaultdict(list)
    for r in rows:
        if r["rec"]["kind"] == kind and elig(r, tau):
            g[key(r)].append(fails(r, tau))
    return {str(k): SM.rate(v) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}


def lab(v, yes, no):
    return "n/a" if v is None else (yes if v else no)


def standard_strata(rows, tau):
    J = lambda r: r["J"][tau]
    R = lambda r: r["rec"]
    return {
        "i by detector_split": strata(rows, tau, "i", lambda r: lab(R(r)["detector_split"], "split", "same region")),
        "i by geom_cross": strata(rows, tau, "i", lambda r: lab(R(r)["geom_cross"], "crosses", "no")),
        "i by input_excursion": strata(rows, tau, "i", lambda r: lab(J(r)["input_excursion"], "yes", "no")),
        "m by geom_cross": strata(rows, tau, "m", lambda r: lab(R(r)["geom_cross"], "crosses", "no")),
        "m by input_excursion": strata(rows, tau, "m", lambda r: lab(J(r)["input_excursion"], "yes", "no")),
        "s by pool x lighting": {**strata(rows, tau, "s_random", lambda r: f"random/{R(r)['lighting']}"),
                                 **strata(rows, tau, "s_targeted", lambda r: f"targeted/{R(r)['lighting']}")},
        "d same-material by illumination": strata(rows, tau, "d_R", lambda r: R(r)["illumination"]),
        "i by textured": strata(rows, tau, "i", lambda r: lab(R(r)["textured"], "textured", "smooth")),
        "s_random by textured": strata(rows, tau, "s_random", lambda r: lab(R(r)["textured"], "textured", "smooth")),
        "d_R by textured": strata(rows, tau, "d_R", lambda r: lab(R(r)["textured"], "textured", "smooth")),
        "i by provenance group": strata(rows, tau, "i", lambda r: R(r).get("group")),
        "m by provenance group": strata(rows, tau, "m", lambda r: R(r).get("group")),
        "i by annotator group": strata(rows, tau, "i", lambda r: R(r).get("annotator_group")),
        "m by annotator group": strata(rows, tau, "m", lambda r: R(r).get("annotator_group")),
        "i by reflective_advisory": strata(rows, tau, "i", lambda r: lab(R(r).get("reflective_advisory"), "reflective (advisory)", "not flagged")),
        "m by reflective_advisory": strata(rows, tau, "m", lambda r: lab(R(r).get("reflective_advisory"), "reflective (advisory)", "not flagged")),
        "i by glossy pool": strata(rows, tau, "i", lambda r: lab(R(r).get("glossy_pool"), "glossy pool", "not glossy")),
        "m by glossy pool": strata(rows, tau, "m", lambda r: lab(R(r).get("glossy_pool"), "glossy pool", "not glossy")),
        "i by annotator": strata(rows, tau, "i", lambda r: R(r)["annotator"]),
        "m by annotator": strata(rows, tau, "m", lambda r: R(r)["annotator"]),
        "i by length_out_of_range": strata(rows, tau, "i", lambda r: lab(R(r)["length_out_of_range"], "out", "in")),
        "m by length_out_of_range": strata(rows, tau, "m", lambda r: lab(R(r)["length_out_of_range"], "out", "in")),
    }


def m_with_without_excursion(rows, tau):
    return dict(all=kind_rate(rows, "m", tau),
                without_input_excursion=kind_rate(rows, "m", tau, pred=lambda r: not r["J"][tau]["input_excursion"]))


def crosstab(rows, tau):
    """input_excursion x textured x geom_cross per type (all pairs of the type, with eligible ER per cell)."""
    out = {}
    for k in ("i", "m", "b"):
        cells = defaultdict(lambda: dict(n=0, eligible=[], ))
        for r in rows:
            if r["rec"]["kind"] != k:
                continue
            key = (lab(r["J"][tau]["input_excursion"], "exc", "no-exc"), lab(r["rec"]["textured"], "tex", "smooth"),
                   lab(r["rec"]["geom_cross"], "geom", "no-geom"))
            c = cells[key]
            c["n"] += 1
            if elig(r, tau) and k != "b":
                c["eligible"].append(fails(r, tau))
        out[k] = {"/".join(key): dict(n=c["n"], ER=SM.rate(c["eligible"]) if k != "b" else None)
                  for key, c in sorted(cells.items())}
    return out


def per_image(rows, tau):
    g = defaultdict(list)
    for r in rows:
        g[r["rec"]["image"]].append(r)
    out = {}
    for s, rr in sorted(g.items()):
        out[s] = {k: kind_rate(rr, k, tau) for k in SM.RATED}
        out[s]["_n"] = Counter(r["rec"]["kind"] for r in rr)
    return out


def summarize(rows, taus, tau0, images, cfg):
    A = cfg["aggregate"]
    return dict(
        headline={t: dict(wide=headline(rows, t, "wide"), plain=headline(rows, t, "plain")) for t in taus},
        headline_fixed_pop={t: dict(wide=headline_fixed_pop(rows, t, tau0, "wide"), plain=headline_fixed_pop(rows, t, tau0, "plain"))
                            for t in taus},
        texture_strata=texture_strata_counts(rows, tau0, cfg),
        ci=bootstrap(rows, tau0, images, A["bootstrap_n"], A["bootstrap_seed"]),
        exclusions={t: exclusion_counts(rows, t) for t in taus},
        companions=companions(rows, tau0), bins=bins(rows, tau0), bins_plain=bins(rows, tau0, "plain"),
        b=b_diag(rows, tau0), strata=standard_strata(rows, tau0), m_excursion=m_with_without_excursion(rows, tau0),
        crosstab=crosstab(rows, tau0), per_image=per_image(rows, tau0))
