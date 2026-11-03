"""Combined geometry sheet: stage_b/<run>/viz_geom/<stem>.png, upright, 3x3 panels + a text box (TASK_GEOM_INTEGRATION G5).

For runs with the depth, fold and Laplacian channels (geom_v1). From the saved outputs only (<stem>.npz, <stem>.json,
the depth and fold candidate tables) plus the Stage A sRGB for panel 1. Network pixel coordinates (the full-res sRGB is
drawn over the same extent).
  1 sRGB + accepted depth edges (log10 dZ; sky edges cyan), fold edges (dtheta), rescued creases (dashed)
  2 log10 Z   3 normal map   4 w_depth   5 w_fold   6 combined w
  7 tri_veto4 w (the Laplacian channel is tri_veto4's detector, unchanged)
  8 disagreement: Laplacian-only (red: w_lap < 0.5, min(w_depth, w_fold) >= 0.9) vs depth/fold-only (blue: the reverse)
  9 text box: channel settings, counts, shares, alignment shift (none)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from .config import Config  # noqa: E402

DZ_LOG_LIM = (-2.0, 1.5)     # log10 m
DTHETA_LIM = (0.0, 90.0)     # deg
GEOM_CHANNELS = {"depth", "fold", "laplacian"}


def wants_geom_sheet(channels) -> bool:
    return GEOM_CHANNELS <= set(channels or ())


def _segments(df: pd.DataFrame, half: float) -> np.ndarray:
    t = np.stack([-df["e1_v"].to_numpy(), df["e1_u"].to_numpy()], -1)
    c = df[["u0", "v0"]].to_numpy()
    return np.stack([c - half * t, c + half * t], 1)


def disagreement(w_lap: np.ndarray, w_df: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Laplacian-only, depth/fold-only) masks with the G1 audit thresholds (< 0.5 in one, >= 0.9 in the other)."""
    return (w_lap < 0.5) & (w_df >= 0.9), (w_df < 0.5) & (w_lap >= 0.9)


def render_geom(cfg: Config, rd: Path, stem: str) -> Path:
    from .stage_b import load_upright

    a, meta = load_upright(cfg, rd, stem)
    pg = meta["params"]["geometry"]
    foc, ch = meta["focal"], meta["channels"]
    dc = pd.read_parquet(rd / f"{stem}_depth_candidates.parquet")
    fc = pd.read_parquet(rd / f"{stem}_fold_candidates.parquet")
    sa_json = cfg.out_dir / meta["provenance"]["stage_a_json"]
    srgb = cv2.cvtColor(cv2.imread(str(sa_json.parent / "srgb" / f"{stem}.png")), cv2.COLOR_BGR2RGB)

    wd, wf, wl = (a[k].astype(np.float64) for k in ("depth_w_net", "fold_w_net", "laplacian_w_net"))
    w = a["w_net"].astype(np.float64)
    H_n, W_n = w.shape
    ext = (-0.5, W_n - 0.5, H_n - 0.5, -0.5)
    portrait = H_n > W_n
    fig = plt.figure(figsize=(18, 19) if portrait else (22, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(9)]
    axt = ax[8]
    fig.suptitle(f"{stem}   run={rd.name}   geometric edges {meta['combined']['channels']} (min)   upright frame",
                 fontsize=14)

    # 1. sRGB + accepted edges
    ax[0].imshow(srgb, extent=ext)
    dacc = dc[dc["accepted"]]
    fin, sky = dacc[np.isfinite(dacc["dZ"]) & ~dacc["sky_edge"]], dacc[dacc["sky_edge"]]
    s = ax[0].scatter(fin["u0"], fin["v0"], s=1.2, c=np.log10(fin["dZ"].clip(lower=1e-3)), cmap="plasma",
                      vmin=DZ_LOG_LIM[0], vmax=DZ_LOG_LIM[1], linewidths=0)
    ax[0].scatter(sky["u0"], sky["v0"], s=1.2, c="cyan", linewidths=0)
    facc = fc[fc["accepted"]]
    rescued = facc[facc["rescued"]] if "rescued" in facc else facc.iloc[:0]
    plain = facc[~facc.index.isin(rescued.index)]
    norm = plt.Normalize(*DTHETA_LIM)
    lc = LineCollection(_segments(plain, 0.7), cmap="viridis", norm=norm, linewidths=1.4)
    lc.set_array(np.degrees(plain["dtheta"].to_numpy()))
    ax[0].add_collection(lc)
    if len(rescued):
        lcr = LineCollection(_segments(rescued, 1.2), cmap="viridis", norm=norm, linewidths=1.4, linestyles="--")
        lcr.set_array(np.degrees(rescued["dtheta"].to_numpy()))
        ax[0].add_collection(lcr)
    fig.colorbar(s, ax=ax[0], fraction=0.046, pad=0.01, label="depth: log10 Delta Z [m]")
    fig.colorbar(lc, ax=ax[0], fraction=0.046, pad=0.01, label="fold: dtheta(k1) [deg]")
    ax[0].set_title(f"depth edges by log10 dZ ({len(dacc)}, cyan: sky {len(sky)})\n"
                    f"folds by dtheta ({len(facc)}, dashed: rescued creases {len(rescued)})")

    # 2. log10 Z
    Z = foc["f_n"] * a["D"].astype(np.float64) / pg["depth"]["canonical_focal"]
    im = ax[1].imshow(np.ma.masked_where(a["sky_mask"], np.log10(Z)), cmap="viridis", extent=ext)
    fig.colorbar(im, ax=ax[1], fraction=0.046, label="log10 Z [m]")
    ax[1].set_title("log10 Z = log10(f_n D / 300) (sky masked)")

    # 3. normal map
    n = a["n_hat"].astype(np.float64)
    nv = a["normal_valid"]
    ax[2].imshow(np.where(nv[..., None], np.clip((n + 1) / 2, 0, 1), 0.5), extent=ext)
    ax[2].set_title("normal map RGB = (n+1)/2; gray invalid")

    # 4-7. weights
    for i, (img, title) in enumerate([(wd, "w_depth"), (wf, "w_fold"), (w, "combined w = min(depth, fold, laplacian)"),
                                      (wl, "tri_veto4 w (= the Laplacian channel)")]):
        im = ax[3 + i].imshow(img, cmap="gray", vmin=0, vmax=1, extent=ext)
        ax[3 + i].set_title(f"{title}\nw < 0.5: {(img < 0.5).mean():.3f}   w < 0.9: {(img < 0.9).mean():.3f}")
    fig.colorbar(im, ax=ax[6], fraction=0.046, label="w")

    # 8. disagreement
    lap_only, df_only = disagreement(wl, np.minimum(wd, wf))
    gray = cv2.cvtColor(cv2.resize(srgb, (W_n, H_n), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    base = np.repeat((0.35 + 0.4 * gray[..., None] / 255.0), 3, axis=2)
    base[lap_only] = (0.85, 0.1, 0.1)
    base[df_only] = (0.1, 0.35, 0.95)
    ax[7].imshow(base, extent=ext)
    ax[7].set_title(f"red: Laplacian-only {lap_only.mean():.3f}   blue: depth/fold-only {df_only.mean():.3f}\n"
                    "(< 0.5 in one, >= 0.9 in the other)")

    for k in (0, 1, 2, 3, 4, 5, 6, 7):
        ax[k].set_xlim(ext[0], ext[1])
        ax[k].set_ylim(ext[2], ext[3])
        ax[k].set_xticks([])
        ax[k].set_yticks([])

    # text box
    d, f = ch["depth"], ch["fold"]
    lp = pg.get("laplacian", {}).get("pool_radius")
    lines = [
        f"run {rd.name}   stem {stem}",
        f"channels {meta['combined']['channels']}, combine {meta['combined']['combine']}",
        "alignment shift: none",
        "",
        "depth: null {}, dZ0 {} m, band_min {},\n  sky_band {}".format(
            pg["depth"]["null_model"], pg["depth"]["dZ0"], pg["depth"]["band_min"], pg["depth"].get("sky_band", "depth")),
        f"  accepted {d.get('n_accepted')} (sky {len(sky)}), sky ring px {d.get('n_sky_ring_px', 0)}",
        "fold: null {}, veto {}, band_min {},\n  crease_rescue {}".format(
            pg["fold"]["null_model"], pg["fold"]["veto_mode"], pg["fold"]["band_min"],
            pg["fold"].get("crease_rescue", "off")),
        (f"  rescue theta {pg['fold']['rescue_theta_deg']} deg, k_L {pg['fold']['rescue_k_L']}"
         if pg["fold"].get("crease_rescue", "off") != "off" else "  rescue off"),
        f"  accepted {f.get('n_accepted')}, rescued {f.get('n_rescued', 0)}",
        "laplacian: tri_veto4 detector,\n  pooling radius {}".format(lp if lp is not None else
                                                                   f"{meta['detector']['r']} (detector r)"),
        f"  s_hat {meta['detector']['s_hat']:.3e}, tau {meta['detector']['tau']:.3e}",
        "",
        "share of network px with w < 0.5 / < 0.9:",
        f"  depth     {(wd < 0.5).mean():.4f} / {(wd < 0.9).mean():.4f}",
        f"  fold      {(wf < 0.5).mean():.4f} / {(wf < 0.9).mean():.4f}",
        f"  laplacian {(wl < 0.5).mean():.4f} / {(wl < 0.9).mean():.4f}",
        f"  combined  {(w < 0.5).mean():.4f} / {(w < 0.9).mean():.4f}",
        f"  Laplacian-only {lap_only.mean():.4f}, depth/fold-only {df_only.mean():.4f}",
        "",
        f"f_n {foc['f_n']:.1f} px (network)",
        "deliverable: <stem>.npz 'w' (original res,\n  bilinear upsample, clip w_min)",
    ]
    axt.axis("off")
    axt.set_title("9. settings, counts, shares")
    axt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10, transform=axt.transAxes)
    out = rd / "viz_geom" / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out
