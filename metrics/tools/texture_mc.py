#!/usr/bin/env python3
"""Monte Carlo of the Delta Q1 texture tolerances on the synthetic scene (read-only; writes only --out).

For each texture strength (grey log SD; chroma SD fixed) and seed, dense textured i pairs (every 6 px across mat1,
A 25 px above / B 25 px below the shadow edge, r = 1..3) are scored at tau 0.10 under
  GT    ground-truth ℓR (every pair should pass)
  halo  L blurred across the shadow (sigma 6 px; every pair should fail the profile test, E ~ 0)
for each profile option: the ruled rule (sigma_t 1.5) and §2's "raise sigma_t for textured materials" (3, 4, 6 px
for pairs with se_tex > textured_se_tex). Reported: GT false-fail rates (plain tau; widened, split E / H), the GT E
false-fail rate for k_tex in {2, 2.5, 3}, the halo detection rate (H > tolH_wide) and how often halo E passes.

    metrics/run.sh tools/texture_mc.py [--seeds 30] [--grey 0.10 0.20] [--out metrics/runs/texture_mc.json]
"""
import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

M = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(M), str(M / "tests")]
import numpy as np  # noqa: E402
from core import config as K, pair_metrics as PM  # noqa: E402
import synth as S  # noqa: E402

KS = (2.0, 2.5, 3.0)
# profile options (all at the config k_tex): name -> (sigma_t_textured_px, profile_ref)
OPTIONS = {"ruled": (None, "H_I"), "sigma3": (3, "H_I"), "sigma4": (4, "H_I"), "sigma6": (6, "H_I"),
           "iso": (None, "iso"), "iso_sigma3": (3, "iso")}


def textured_pairs(rs=(1, 2, 3)):
    P = []
    for r in rs:
        for x in range(S.TEX_X[0] + 6, S.TEX_X[1] - 5, 6):
            P.append(dict(S._p("i", (x, 135), (x, 185), r=r, lit_end="A"), tag=f"r{r}"))
        for x0 in S.TEX_X:                                 # m pairs with one textured side (L flat; plain tau)
            for y in list(range(20, 141, 6)) + list(range(180, 301, 6)):
                P.append(dict(S._p("m", (x0 - 20, y), (x0 + 20, y), r=r), tag=f"m_r{r}"))
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--grey", type=float, nargs="+", default=[0.10, 0.20])
    ap.add_argument("--chroma", type=float, default=0.04)
    ap.add_argument("--options", nargs="+", default=["ruled", "sigma3", "iso"], choices=list(OPTIONS))
    ap.add_argument("--out")
    a = ap.parse_args()
    base = K.load_config()
    tau = base["thresholds"]["tau"]
    P = textured_pairs()
    cfgs = {}
    for s in a.options:
        c = copy.deepcopy(base)
        c["profile"]["sigma_t_textured_px"], c["texture"]["profile_ref"] = OPTIONS[s]
        cfgs[s] = c
    out, allT, allF = {}, {}, {}
    for g in a.grey:
        T = defaultdict(lambda: defaultdict(float))
        F = defaultdict(lambda: defaultdict(float))
        allT[g] = T
        allF[g] = F
        for seed in range(a.seeds):
            sc = S.scene(seed, tex_grey=g, tex_chroma=a.chroma)
            img = S.image_dict(base, sc)
            preds = {"GT": sc["lR_gt"], "halo": S.halo(img["lI"], sc["lR_gt"])}
            for s, cfg in cfgs.items():
                for pn, lR in preds.items():
                    ctx = S.ctx(img, lR)
                    for p in P:
                        rec, _ = PM.measure(p, ctx, cfg)
                        j = PM.judge(rec, tau, cfg)
                        if pn == "GT":                     # per kind x textured status (floor of a perfect decomposition)
                            q = F[(s, p["type"], bool(rec["textured"]))]
                            q["n"] += 1
                            q["fail_wide"] += not j["pass_wide"]
                            q["se_tex_sum"] += rec["se_tex"]
                        if p["type"] != "i":
                            continue
                        c = T[(s, p["tag"], pn)]
                        c["n"] += 1
                        c["textured"] += bool(rec["textured"])
                        c["fail_plain"] += not j["pass_plain"]
                        c["fail_wide"] += not j["pass_wide"]
                        c["fail_wide_E"] += not rec["E"] <= j["tolE_wide"]
                        c["fail_wide_H"] += not rec["H"] <= j["tolH_wide"]
                        for k in KS:
                            c[f"E_gt_{k}"] += not rec["E"] <= max(tau, k * rec["se_tex"])
        print(f"\ntexture grey SD {g}, chroma SD {a.chroma}; {a.seeds} seeds x {len(P) // 3} pairs per r; tau {tau}")
        print(f"k_tex {base['texture']['k_tex']}")
        print(f"{'option':11s} {'r':3s} {'textured':>8s} | GT: {'plain':>6s} {'wide':>6s} {'(E/H)':>13s} "
              f"{'E k2/2.5/3':>17s} | halo: {'H det':>6s} {'E ok':>6s}")
        for s in a.options:
            for r in ("r1", "r2", "r3", "all"):
                if r == "all":
                    for pn in ("GT", "halo"):
                        agg = defaultdict(float)
                        for rr in ("r1", "r2", "r3"):
                            for kk, vv in T[(s, rr, pn)].items():
                                agg[kk] += vv
                        T[(s, "all", pn)] = agg
                G, Hh = T[(s, r, "GT")], T[(s, r, "halo")]
                n = G["n"]
                pc = lambda c, k: f"{100 * c[k] / c['n']:5.1f}%"
                print(f"{s:11s} {r:3s} {pc(G, 'textured'):>8s} | "
                      f"{pc(G, 'fail_plain'):>10s} {pc(G, 'fail_wide'):>6s} {pc(G, 'fail_wide_E') + '/' + pc(G, 'fail_wide_H'):>13s} "
                      f"{'/'.join(f'{100 * G[f'E_gt_{k}'] / n:.1f}' for k in KS):>17s} | "
                      f"{pc(Hh, 'fail_wide_H'):>12s} {100 - 100 * Hh['fail_wide_E'] / Hh['n']:5.1f}%")
    for g, T in allT.items():
        out[g] = {f"{k[0]}|{k[1]}|{k[2]}": dict(v) for k, v in T.items()}
    floor = {g: {f"{k[0]}|{k[1]}|{'textured' if k[2] else 'not_textured'}": dict(v) for k, v in F.items()} for g, F in allF.items()}
    print("\nGT false fails by kind and textured status (widened rule):")
    for g, F in allF.items():
        for k, v in sorted(F.items()):
            print(f"  grey {g} {k[0]:7s} {k[1]} {'textured' if k[2] else 'not textured':12s} n {int(v['n']):6d} "
                  f"fail {100 * v['fail_wide'] / v['n']:5.1f}%  mean se_tex {v['se_tex_sum'] / v['n']:.3f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(args=vars(a), tau=tau, k_tex=base["texture"]["k_tex"], options={k: OPTIONS[k] for k in a.options},
                                               textured_threshold=base["texture"]["textured_se_tex"], results=out, floor=floor), indent=1))


if __name__ == "__main__":
    main()
