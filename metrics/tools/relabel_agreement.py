#!/usr/bin/env python3
"""Agreement between the original pair labels and blind relabels (annotation tool, blind relabel mode).

    metrics/run.sh tools/relabel_agreement.py [--out metrics/runs/relabel_agreement] [--include-own]

Reads metrics/annotations/<stem>.json and metrics/annotations/relabels/<stem>.json (read-only). One comparison per
relabel entry: the pair's current type and labels (original annotator) vs the relabel (relabeler). By default
own-pair relabels (test-retest) are reported separately from cross-annotator ones.
Reported per group (cross-annotator; own-pair):
  - type: per-type agreement (share of pairs of each original type relabelled the same), confusion matrix,
    overall agreement, Cohen's kappa (n/a when undefined);
  - d pairs relabelled as d: agreement and Cohen's kappa for material and for illumination (same / different / unsure);
  - i pairs relabelled as i: lit-end agreement.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config as K  # noqa: E402


def kappa(pairs):
    """Cohen's kappa for a list of (rater1, rater2) labels; None if undefined (no pairs or chance agreement 1)."""
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    c1, c2 = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(c1[k] * c2[k] for k in set(c1) | set(c2)) / n ** 2
    return None if pe >= 1 else (po - pe) / (1 - pe)


def report(cmp):
    T = "imbsd"
    out = dict(n=len(cmp))
    tp = [(c["orig"]["type"], c["new"]["type"]) for c in cmp]
    out["type_agreement"] = sum(a == b for a, b in tp) / len(tp) if tp else None
    out["type_kappa"] = kappa(tp)
    out["per_type"] = {t: dict(n=sum(a == t for a, _ in tp), agree=(sum(a == t and b == t for a, b in tp) / n) if (n := sum(a == t for a, _ in tp)) else None) for t in T}
    out["confusion"] = {t: {u: sum(a == t and b == u for a, b in tp) for u in T} for t in T}
    dd = [c for c in cmp if c["orig"]["type"] == "d" and c["new"]["type"] == "d"]
    for lab in ("material", "illumination"):
        pr = [(c["orig"][lab], c["new"][lab]) for c in dd]
        out[f"d_{lab}"] = dict(n=len(pr), agreement=sum(a == b for a, b in pr) / len(pr) if pr else None, kappa=kappa(pr))
    ii = [c for c in cmp if c["orig"]["type"] == "i" and c["new"]["type"] == "i"]
    out["i_lit_end"] = dict(n=len(ii), agreement=sum(c["orig"]["lit_end"] == c["new"]["lit_end"] for c in ii) / len(ii) if ii else None)
    return out


def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(K.METRICS / "runs" / "relabel_agreement"))
    a = ap.parse_args()
    cfg = K.load_config()
    ann = K.repo_path(cfg["paths"]["annotations"])
    rel = K.repo_path(cfg["annotate"]["relabels"])
    groups = defaultdict(lambda: {"cross": [], "own": []})
    for rp in sorted(rel.glob("*.json")) if rel.exists() else []:
        rl = json.loads(rp.read_text())
        f = json.loads((ann / f"{rl['image']}.json").read_text())
        pairs = {p["id"]: p for p in f["pairs"]}
        g = K.group_of(cfg, f["set"])
        for e in rl["relabels"]:
            p = pairs.get(e["pair_id"])
            if p is None:
                continue                      # pair deleted since
            groups[g]["own" if e.get("own_pair") else "cross"].append(dict(orig=p, new=e, image=rl["image"]))
    res = {g: {k: report(v) for k, v in d.items()} for g, d in groups.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "agreement.json").write_text(json.dumps(res, indent=1))
    L = ["# Blind relabel agreement", ""]
    if not res:
        L.append("No relabels yet (metrics/annotations/relabels/ is empty).")
    for g, d in res.items():
        for k, r in d.items():
            if not r["n"]:
                continue
            L += [f"## {g}: {'cross-annotator' if k == 'cross' else 'own pairs (test-retest)'} ({r['n']} relabels)", "",
                  f"type: agreement {fmt(r['type_agreement'])}, Cohen's kappa {fmt(r['type_kappa'])}", "",
                  "| original \\ relabel | " + " | ".join("imbsd") + " | n | agree |", "|---|" + "---|" * 7]
            for t in "imbsd":
                L.append(f"| {t} | " + " | ".join(str(r["confusion"][t][u]) for u in "imbsd") + f" | {r['per_type'][t]['n']} | {fmt(r['per_type'][t]['agree'])} |")
            L += ["", f"d material: n {r['d_material']['n']}, agreement {fmt(r['d_material']['agreement'])}, kappa {fmt(r['d_material']['kappa'])}  ",
                  f"d illumination: n {r['d_illumination']['n']}, agreement {fmt(r['d_illumination']['agreement'])}, kappa {fmt(r['d_illumination']['kappa'])}  ",
                  f"i lit end: n {r['i_lit_end']['n']}, agreement {fmt(r['i_lit_end']['agreement'])}", ""]
    (out / "agreement.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
