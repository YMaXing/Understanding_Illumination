"""Per-kind error rates with explicit empties (Delta: a rate over zero pairs is "n/a: 0 pairs", never 0).
Phase 3's aggregate.py (headlines, bootstrap, strata) builds on these."""

KINDS = ("i", "m", "b", "s_random", "s_targeted", "d_R", "d_L", "d_unscored")
RATED = ("i", "m", "s_random", "s_targeted", "d_R", "d_L")


def rate(fails):
    """fails: iterable of bool (True = failed). Error rate, or n/a when empty."""
    f = [bool(x) for x in fails]
    n, k = len(f), sum(f)
    if n == 0:
        return dict(n=0, k=0, rate=None, text="n/a: 0 pairs")
    return dict(n=n, k=k, rate=k / n, text=f"{k / n:.3f} ({k}/{n})")


def by_kind(results, tau, rule="wide", headline_only=True):
    """results: [(rec, {tau: judge}, prof)]. Error rate per kind under rule 'plain' or 'wide'. With headline_only,
    pairs with any exclusion are left out (targeted s, d_L etc. are then empty by design)."""
    out = {}
    for k in RATED:
        sel = [J[tau] for rec, J, _ in results if rec["kind"] == k and (not headline_only or not J[tau]["exclusions"])]
        out[k] = rate(not j[f"pass_{rule}"] for j in sel)
    return out
