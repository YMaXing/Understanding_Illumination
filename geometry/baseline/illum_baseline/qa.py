"""Dataset QA for one Stage B run (spec M5): histograms, focal pooling, flags, suggested s_hat clamps.

Writes out_dir/stage_b/<run>/qa_report.md and qa/*.png.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config, detector_params  # noqa: E402
from .detector import compute_weights  # noqa: E402
from .manifest import manifest_frame  # noqa: E402
from .stage_a import stage_a_dir  # noqa: E402
from .stage_b import run_dir  # noqa: E402

HISTS = [
    ("s_hat_raw", "s_hat (unclamped)"),
    ("sky_frac", "sky_frac"),
    ("f_per_image", "f per image [upright full-res px]"),
    ("focal_rel_uncertainty", "focal_uncertainty / f"),
    ("frac_w_lt_0.5", "frac(w < 0.5)"),
]


def _histograms(df: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, len(HISTS), figsize=(4.2 * len(HISTS), 3.4), constrained_layout=True)
    for ax, (col, label) in zip(axes, HISTS):
        v = df[col].to_numpy(dtype=float)
        ax.hist(v, bins=min(30, max(5, len(v))), color="0.35")
        ax.axvline(np.median(v), color="k", ls="--", lw=1, label=f"median {np.median(v):.3g}")
        if col == "s_hat_raw":
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            for pct in (2, 98):
                ax.axvline(np.percentile(v, pct), color="tab:red", ls=":", lw=1)
        if col == "focal_rel_uncertainty":
            ax.axvline(0.15, color="tab:red", ls=":", lw=1, label="flag 0.15")
        ax.set_title(f"{label} (n={len(v)})")
        ax.legend(fontsize=8)
    path = out / "qa_histograms.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _clamp_effect(cfg: Config, df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Re-run the detector from the Stage A cache with the suggested clamps (only clamped images change)."""
    p = dataclasses.replace(detector_params(cfg), s_hat_min=lo, s_hat_max=hi)
    rows = []
    for _, r in df.iterrows():
        if lo <= r["s_hat_raw"] <= hi:
            continue
        z = np.load(stage_a_dir(cfg) / f"{r['stem']}.npz")
        sky = z["sky_prob"].astype(np.float64) if "sky_prob" in z.files else None
        s = compute_weights(z["D"], sky, p)["scalars"]
        rows.append({"stem": r["stem"], "s_hat_raw": r["s_hat_raw"], "s_hat_clamped_to": s["s_hat"],
                     "frac_w_lt_0.5_now": r["frac_w_lt_0.5"], "frac_w_lt_0.5_clamped": s["frac_w_lt_0.5"]})
    return pd.DataFrame(rows)


def _table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> list[str]:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, fmt[c]) if c in fmt and isinstance(v, (int, float, np.number)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def run_qa(cfg: Config, run_name: str | None = None) -> Path:
    rd = run_dir(cfg, run_name)
    df = manifest_frame(rd)
    if df.empty:
        raise FileNotFoundError(f"no Stage B outputs in {rd}")
    qd = rd / "qa"
    qd.mkdir(exist_ok=True)
    hist = _histograms(df, qd)
    n = len(df)

    # Focal: pooled vs per-image, per group.
    pooled = json.loads((stage_a_dir(cfg) / "focal_pooled.json").read_text())
    fdf = df[["stem", "f_per_image", "f_pooled", "focal_uncertainty", "focal_rel_uncertainty"]].copy()
    fdf["dev_from_pooled_%"] = 100 * (fdf["f_per_image"] / fdf["f_pooled"] - 1)
    groups = []
    for g, info in pooled["groups"].items():
        groups.append({"group": g, "n": info["n"], "f_pooled (median)": info["f_pooled"],
                       "f_min": info["f_min"], "f_max": info["f_max"],
                       "spread_%": 100 * (info["f_max"] - info["f_min"]) / info["f_pooled"]})

    # Flags.
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    flagged = df[df["n_flags"] > 0]
    flag_rows = [{"stem": r["stem"], "flags": ", ".join(c.removeprefix("flag_") for c in flag_cols if r[c])}
                 for _, r in flagged.iterrows()]

    # Suggested clamps from the unclamped s_hat.
    lo, hi = (float(v) for v in np.percentile(df["s_hat_raw"], [2, 98]))
    effect = _clamp_effect(cfg, df, lo, hi)

    lines = [
        f"# QA report: run `{rd.name}`",
        "",
        f"- images: {n}; Stage A cache `{stage_a_dir(cfg)}`; manifest `{rd / 'manifest.parquet'}`",
        f"- detector params: k_tau={cfg.detector.k_tau}, k_eps={cfg.detector.k_eps}, w_min={cfg.detector.w_min}, "
        f"sigma={cfg.detector.sigma}, s_hat_min={cfg.detector.s_hat_min}, s_hat_max={cfg.detector.s_hat_max}",
        f"- focal mode: {cfg.focal.mode}; group_key: {cfg.focal.group_key}",
        "",
        "## Histograms",
        "",
        f"![histograms](qa/{hist.name})",
        "",
        "| quantity | min | P2 | median | P98 | max |",
        "|---|---|---|---|---|---|",
    ]
    for col, label in HISTS:
        v = df[col].to_numpy(dtype=float)
        q = np.percentile(v, [0, 2, 50, 98, 100])
        lines.append(f"| {label} | " + " | ".join(f"{x:.4g}" for x in q) + " |")
    lines += [
        "",
        "## Focal: pooled vs per-image",
        "",
        *_table(pd.DataFrame(groups), {"f_pooled (median)": ".1f", "f_min": ".1f", "f_max": ".1f", "spread_%": ".1f"}),
        "",
    ]
    if not pooled["pooled_meaningful"]:
        lines += [f"**Pooled f is not meaningful here:** {pooled['note']}.", ""]
    lines += [
        *_table(fdf, {"f_per_image": ".1f", "f_pooled": ".1f", "focal_uncertainty": ".1f",
                      "focal_rel_uncertainty": ".3f", "dev_from_pooled_%": "+.1f"}),
        "",
        "## Flagged images",
        "",
        f"Thresholds: sky_frac > 0.5; focal_uncertainty/f > 0.15; s_hat outside dataset P2-P98 "
        f"[{df.attrs['thresholds']['s_hat_p2']:.3e}, {df.attrs['thresholds']['s_hat_p98']:.3e}]; n_bad_depth > 0; "
        f"frac_w_lt_0.5 > dataset P98 ({df.attrs['thresholds']['frac_w_lt_0.5_p98']:.3f}).",
        "",
        *(_table(pd.DataFrame(flag_rows)) if flag_rows else ["none"]),
        "",
        f"{len(flagged)}/{n} images carry at least one flag. Flag counts: "
        + ", ".join(f"{c.removeprefix('flag_')}={int(df[c].sum())}" for c in flag_cols) + ".",
        "",
    ]
    if n < 50:
        lines += [f"Note: with n={n}, the percentile-based flags (P2/P98) mark essentially the single most extreme "
                  "image(s) by construction.", ""]
    lines += [
        "## Suggested dataset-level s_hat clamps",
        "",
        "From the unclamped per-image s_hat (P2 / P98):",
        "",
        "```yaml",
        "detector:",
        f"  s_hat_min: {lo:.4e}",
        f"  s_hat_max: {hi:.4e}",
        "```",
        "",
        "Not applied. Effect on the images the clamps would change (detector re-run from the Stage A cache):",
        "",
        *(_table(effect, {"s_hat_raw": ".3e", "s_hat_clamped_to": ".3e", "frac_w_lt_0.5_now": ".3f",
                          "frac_w_lt_0.5_clamped": ".3f"}) if not effect.empty else ["none"]),
        "",
        "## Per-image detector scalars",
        "",
        *_table(df[["stem", "s_hat", "tau", "rho_floor", "sky_frac", "n_bad_depth", "frac_w_lt_0.5",
                    "frac_w_lt_0.9", "frac_w_lt_1", "f_n"]],
                {"s_hat": ".3e", "tau": ".3e", "rho_floor": ".3e", "sky_frac": ".4f", "frac_w_lt_0.5": ".3f",
                 "frac_w_lt_0.9": ".3f", "frac_w_lt_1": ".3f", "f_n": ".1f"}),
        "",
    ]
    path = rd / "qa_report.md"
    path.write_text("\n".join(lines))
    return path
