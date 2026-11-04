"""TASK_DEPTH_CHANNEL D5 report: runs depth_v1 (multiplicative null), depth_v1_add (additive null) and depth_fold_v1,
against tri_veto4 and tri_veto4_fold_v2 (all from saved outputs).

- per image and run: share w < 0.5 / w < 0.9 (combined w_net; w_depth for the depth runs), candidate counts per stage,
  runtime;
- per evaluation class (fold / step / limb / texture; pinned ROIs scripts/depth_rois.py, fold line targets
  scripts/fold_null_rois.py except those relabelled in depth_rois.RELABELLED, texture areas scripts/fold_null_rois.py: rois + the trimmed bollard interior): share of
  w_depth < 0.5, per null;
- the spot-check targets of TASK_DEPTH_CHANNEL D5, per null;
- the decision rule for the additive null (proposed, not switched): additive becomes the default only if the share of
  w_depth < 0.5 on the 070 floor and both 079 grass ROIs is <= multiplicative + 0.02 and every step / limb target met
  under multiplicative is still met;
- overlays of the candidates rejected only by significance (reason signif with r > tau_low), per null, 079 075 070;
- synthetic test 7 (box silhouette recall) under the additive null;
- 090 (chain-link fence) as its own clutter case.
Writes baseline_output/diag/depth_v1/d5/{d5_report.json, signif_only_<stem>.png}.

Run from geometry/dedicated:  env -u PYTHONPATH uv run python scripts/depth_d5_report.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from depth_rois import BOXES, POLYLINES, RELABELLED, box_mask, sky_band_070  # noqa: E402
from fold_null_rois import TARGETS, _poly_samples, rois  # noqa: E402
from illum_baseline.config import load_config  # noqa: E402
from illum_baseline.stage_b import image_stems, load_upright  # noqa: E402

SB = ROOT / "baseline_output" / "stage_b"
OUT = ROOT / "baseline_output" / "diag" / "depth_v1" / "d5"
NULL_RUNS = {"multiplicative": "depth_v1", "additive": "depth_v1_add"}
RUNS = ("depth_v1", "depth_v1_add", "depth_fold_v1", "tri_veto4", "tri_veto4_fold_v2")
FENCE = (0, 150, 0, 280)  # 090 chain-link fence region (u0, u1, v0, v1), upright network px
STAGES = ("n_cand", "n_polarity", "n_height", "n_sharpness", "n_centring", "n_sign_ok", "n_signif_ok", "n_rel_ok",
          "n_accepted")


def load(run, stem):
    cfg = load_config(SB / run / "config_resolved.yaml")
    return load_upright(cfg, SB / run, stem)


def poly_w(w, poly):
    S = _poly_samples(poly)
    H, W = w.shape
    return w[np.clip(np.rint(S[:, 1]).astype(int), 0, H - 1), np.clip(np.rint(S[:, 0]).astype(int), 0, W - 1)], S


def class_samples(stem: str, w: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """{class: {roi name: w samples}} for this image."""
    s = stem[-3:]
    out: dict[str, dict[str, np.ndarray]] = {"fold": {}, "step": {}, "limb": {}, "texture": {}}
    for name, (st, cls, poly) in POLYLINES.items():
        if st == s:
            out[cls][name] = poly_w(w, poly)[0]
    for name, (st, kind, poly) in TARGETS.items():
        if st == s and name not in RELABELLED:  # relabelled targets are taken from POLYLINES with their new class
            out["fold"][f"{name} ({kind})"] = poly_w(w, poly)[0]
    for name, m in rois(stem, w.shape).items():
        if name != "083 bollard":
            out["texture"][name] = w[m]
    for name, (st, cls, box) in BOXES.items():
        if st == s and "legacy" not in name:
            out[cls][name] = w[box_mask(w.shape, box)]
    if s == "070":
        out["step"]["070 sky boundary"] = w[sky_band_070()]
    return out


def spot_checks(w_depth: dict, w_tri: dict, runtime: list[float]) -> list[dict]:
    """w_depth, w_tri: {stem3: upright map}."""
    rows = []

    def add(check, target, value, ok):
        rows.append({"check": check, "target": target, "value": value, "met": bool(ok)})

    floor = rois("taiwei_cui_070", w_depth["070"].shape)["070 floor"]
    v, t = float((w_depth["070"][floor] < 0.9).mean()), float((w_tri["070"][floor] < 0.9).mean())
    add("070 floor share w_depth < 0.9", f"<= tri_veto4 + 0.01 = {t + 0.01:.4f}", v, v <= t + 0.01)
    v = float(np.median(w_depth["070"][sky_band_070()]))
    add("070 sky boundary median w_depth", "<= 0.10", v, v <= 0.10)
    for name, m in rois("taiwei_cui_079", w_depth["079"].shape).items():
        v = float((w_depth["079"][m] < 0.5).mean())
        add(f"{name} share w_depth < 0.5", "<= 0.02", v, v <= 0.02)
    sil = [poly_w(w_depth["079"], POLYLINES[k][2])[0] for k in POLYLINES if k.startswith("079 box silhouette")]
    for k, x in zip([k for k in POLYLINES if k.startswith("079 box silhouette")], sil):
        add(f"{k} median w_depth", "<= 0.30", float(np.median(x)), np.median(x) <= 0.30)
    x = np.concatenate(sil)
    add("079 box silhouettes (pooled) median w_depth", "<= 0.30", float(np.median(x)), np.median(x) <= 0.30)
    wf, S = poly_w(w_depth["079"], POLYLINES["079 box lit/shaded fold"][2])
    sel = (S[:, 1] >= 225) & (S[:, 1] <= 260)
    add("079 box fold (lit/shaded, rows 225-260) median w_depth", ">= 0.90", float(np.median(wf[sel])),
        np.median(wf[sel]) >= 0.90)
    rows_w = w_depth["079"][[230, 254], 180:216].ravel()
    add("079 box fold rows 230/254 (u 180-215) median w_depth", ">= 0.90", float(np.median(rows_w)),
        np.median(rows_w) >= 0.90)
    limbs = []
    for k in ("083 bollard limb L", "083 bollard limb R"):
        x = poly_w(w_depth["083"], POLYLINES[k][2])[0]
        limbs.append(x)
        add(f"{k} median w_depth", "<= 0.50", float(np.median(x)), np.median(x) <= 0.50)
    x = poly_w(w_depth["083"], POLYLINES["083 bollard base rim"][2])[0]
    add("083 bollard base rim median w_depth", ">= 0.90", float(np.median(x)), np.median(x) >= 0.90)
    for k, st in (("088 rock ridge", "088"), ("076 facade ledge", "076")):
        x = poly_w(w_depth[st], TARGETS[k][2])[0]
        add(f"{k} median w_depth", ">= 0.90", float(np.median(x)), np.median(x) >= 0.90)
    add("depth runtime per image (CPU, excluding the context)", "< 3 s", float(max(runtime)), max(runtime) < 3)
    return rows


def signif_only_overlay(stem: str) -> Path:
    fig, axs = plt.subplots(1, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (model, run) in zip(axs, NULL_RUNS.items()):
        a, meta = load(run, stem)
        cand = pd.read_parquet(SB / run / f"{stem}_depth_candidates.parquet")
        prm = meta["params"]["geometry"]["depth"]
        sa_json = load_config(SB / run / "config_resolved.yaml").out_dir / meta["provenance"]["stage_a_json"]
        srgb = cv2.cvtColor(cv2.imread(str(sa_json.parent / "srgb" / f"{stem}.png")), cv2.COLOR_BGR2RGB)
        H, W = a["depth_w_net"].shape
        ax.imshow(srgb, extent=(-0.5, W - 0.5, H - 0.5, -0.5))
        acc = cand[cand["accepted"]]
        with np.errstate(invalid="ignore"):
            so = cand[(cand["reason"] == "signif") & (cand["r"] > prm["tau_low"])]
        ax.scatter(acc["u0"], acc["v0"], s=1.0, c="lime", linewidths=0, label=f"accepted ({len(acc)})")
        ax.scatter(so["u0"], so["v0"], s=3.0, c="red", linewidths=0, label=f"rejected only by significance ({len(so)})")
        ax.legend(loc="lower right", fontsize=9, markerscale=4)
        ax.set_title(f"{stem}  {model} null ({run})")
        ax.set_xticks([])
        ax.set_yticks([])
    out = OUT / f"signif_only_{stem[-3:]}.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def test7_additive() -> dict:
    from illum_baseline.detector import Params
    from illum_baseline.geom.context import GeometryContext
    from illum_baseline.geom.depth import DepthChannel, DepthParams
    from illum_baseline.geom.synthetic import scene_objects
    from test_depth_channel import CAN, K, NOISE, coverage, noisy

    Z, sid, _, _ = scene_objects()
    ctx = GeometryContext(noisy(Z * CAN, NOISE, 0), None, K, Params(prefilter="trilateral"))
    box = np.isin(sid, [3, 4, 5])
    pts = []
    for ax in (0, 1):
        a = [slice(None), slice(None)]
        b = [slice(None), slice(None)]
        a[ax], b[ax] = slice(None, -1), slice(1, None)
        za, zb = Z[tuple(a)], Z[tuple(b)]
        sil = (box[tuple(a)] != box[tuple(b)]) & (np.abs(za - zb) / np.minimum(za, zb) > 0.03)
        vv, uu = np.nonzero(sil)
        pts.append(np.stack([uu + 0.5 * (ax == 1), vv + 0.5 * (ax == 0)], -1).astype(float))
    P = np.concatenate(pts)
    H, W = Z.shape
    P = P[(P[:, 0] > 15) & (P[:, 0] < W - 16) & (P[:, 1] > 15) & (P[:, 1] < H - 16)]
    out = {}
    for model in ("multiplicative", "additive"):
        res = DepthChannel().run(ctx, dataclasses.replace(DepthParams(), null_model=model))
        out[model] = {"recall": coverage(res.candidates, P), "n_samples": int(len(P)),
                      "sigma_geo_rho": res.scalars["sigma_geo_rho"], "lambda_hat": res.scalars["lambda_hat"]}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stems = image_stems(SB / "depth_v1")
    per_image, cls_rows = [], []
    w_by = {run: {} for run in RUNS}
    runtime = {run: [] for run in NULL_RUNS.values()}
    for stem in stems:
        s = stem[-3:]
        row = {"stem": s}
        for run in RUNS:
            a, meta = load(run, stem)
            w = a["w_net"].astype(np.float64)
            row[f"{run} w<0.5"], row[f"{run} w<0.9"] = float((w < 0.5).mean()), float((w < 0.9).mean())
            if "depth_w_net" in a:
                wd = a["depth_w_net"].astype(np.float64)
                w_by[run][s] = wd
                sc = meta["channels"]["depth"]
                if run in runtime:
                    runtime[run].append(sc["runtime_s"])
                    row[f"{run} stages"] = ">".join(str(sc[k]) for k in STAGES)
                    row[f"{run} t_s"] = sc["runtime_s"]
                    row[f"{run} ctx_s"] = sc["context_s"]
            else:
                w_by[run][s] = w
        per_image.append(row)
        for model, run in NULL_RUNS.items():
            for cls, d in class_samples(stem, w_by[run][s]).items():
                for name, x in d.items():
                    cls_rows.append({"null_model": model, "class": cls, "roi": name, "stem": s, "n": int(x.size),
                                     "share_w<0.5": float((x < 0.5).mean()), "median_w": float(np.median(x))})
    cls_df = pd.DataFrame(cls_rows)
    cls_tab = (cls_df.assign(k=lambda d: d["share_w<0.5"] * d["n"]).groupby(["null_model", "class"])
               .agg(n=("n", "sum"), k=("k", "sum")).assign(share=lambda d: d["k"] / d["n"]).drop(columns="k"))
    spot = {model: spot_checks(w_by[run], w_by["tri_veto4"], runtime[run]) for model, run in NULL_RUNS.items()}

    # decision rule (proposed, not switched)
    def tex(model):
        d = cls_df[cls_df["null_model"] == model].set_index("roi")["share_w<0.5"]
        return {k: float(d[k]) for k in ("070 floor", "079 grass", "079 grass sunlit")}

    tm, ta = tex("multiplicative"), tex("additive")
    texture_ok = {k: ta[k] <= tm[k] + 0.02 for k in tm}
    sm = {r["check"]: r["met"] for r in spot["multiplicative"]}
    sa = {r["check"]: r["met"] for r in spot["additive"]}
    step_limb = [c for c in sm if ("silhouette" in c or "sky boundary" in c or "limb" in c)]
    kept = {c: (sa[c] or not sm[c]) for c in step_limb}
    rule = {"texture_mult": tm, "texture_add": ta, "texture_ok": texture_ok, "step_limb_kept": kept,
            "additive_qualifies": all(texture_ok.values()) and all(kept.values()),
            "fails_only_on_texture": (not all(texture_ok.values())) and all(kept.values())}

    # 090 fence clutter case
    fence = {}
    u0, u1, v0, v1 = FENCE
    for model, run in NULL_RUNS.items():
        cand = pd.read_parquet(SB / run / "taiwei_cui_090_depth_candidates.parquet")
        a = cand[cand["accepted"]]
        inside = (a["u0"] >= u0) & (a["u0"] <= u1) & (a["v0"] >= v0) & (a["v0"] <= v1)
        w = w_by[run]["090"]
        m = box_mask(w.shape, FENCE)
        fence[model] = {"accepted": int(len(a)), "accepted_in_fence": int(inside.sum()),
                        "thin_share_in_fence": float(a.loc[inside, "thin"].mean()) if inside.any() else float("nan"),
                        "w<0.5_fence": float((w[m] < 0.5).mean()), "w<0.5_rest": float((w[~m] < 0.5).mean()),
                        "w<0.5_image": float((w < 0.5).mean())}

    overlays = {s: str(signif_only_overlay(f"taiwei_cui_{s}")) for s in ("079", "075", "070")}
    t7 = test7_additive()
    sheets = {model: [str(SB / run / "viz_depth" / f"taiwei_cui_{s}.png") for s in ("070", "075", "079", "083", "090")]
              for model, run in NULL_RUNS.items()}
    rep = {"per_image": per_image, "per_class": cls_tab.reset_index().to_dict(orient="records"),
           "per_class_roi": cls_rows, "spot_checks": spot, "decision_rule": rule, "fence_090": fence,
           "test7_box_silhouette_recall": t7, "overlays": overlays, "depth_sheets": sheets}
    (OUT / "d5_report.json").write_text(json.dumps(rep, indent=1, default=float))

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pi = pd.DataFrame(per_image)
    print(pi[[c for c in pi.columns if "w<" in c or c == "stem"]].round(4).to_string(index=False))
    print(pi[["stem"] + [c for c in pi.columns if "stages" in c or "t_s" in c or "ctx" in c]].to_string(index=False))
    print("\nper class (share w_depth < 0.5):\n" + cls_tab.round(4).to_string())
    print("\nper ROI:\n" + cls_df.pivot_table(index=["class", "roi"], columns="null_model",
                                               values="share_w<0.5").round(4).to_string())
    for model, rows in spot.items():
        print(f"\nspot checks ({model}):")
        for r in rows:
            print(f"  {'OK ' if r['met'] else 'NO '} {r['check']:<58} {r['value']:.4f}  ({r['target']})")
    print("\ndecision rule:", json.dumps(rule, indent=1))
    print("090 fence:", json.dumps(fence, indent=1))
    print("test 7 (box silhouettes):", json.dumps(t7, indent=1))
    print("overlays:", overlays)


if __name__ == "__main__":
    main()
