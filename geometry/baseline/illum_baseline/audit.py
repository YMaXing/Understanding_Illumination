"""Dataset audit (spec M1): per-file stats and tags, a proposed preprocess config, thumbnails."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .config import Config
from .io_tiff import NAMED_TAGS, load_tiff, orientation_of, to_upright, upright_size
from .preprocess import LUMA, linear_to_srgb, resolve_layout

THUMB_LONG_SIDE = 512


def list_files(cfg: Config) -> list[Path]:
    files = sorted(p for p in cfg.data_dir.glob(cfg.paths.glob) if p.is_file())
    if not files:
        raise FileNotFoundError(f"no files match {cfg.paths.glob!r} under {cfg.data_dir}")
    return files


def sample_files(files: list[Path], seed: int) -> list[Path]:
    if len(files) <= 500:
        return files
    return sorted(random.Random(seed).sample(files, 200))


def _trailing_zero_bits(a: np.ndarray) -> int:
    """Bits that are zero in every value: >0 means the data does not use the low bits."""
    acc = int(np.bitwise_or.reduce(a.ravel().astype(np.uint64)))
    if acc == 0:
        return 0
    return (acc & -acc).bit_length() - 1


def audit_file(path: Path, data_dir: Path) -> dict[str, Any]:
    arr, tags = load_tiff(path)
    h, w = arr.shape[:2]
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    orient = orientation_of(tags)
    w_up, h_up = upright_size(w, h, orient)
    flat = arr.reshape(-1, ch)
    dmax = np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else None

    rec: dict[str, Any] = {
        "file": str(path.relative_to(data_dir)),
        "stem": path.stem,
        "dtype": str(arr.dtype),
        "W": w, "H": h, "channels": ch,
        "orientation": orient,
        "orientation_tag_present": "Orientation" in tags,
        "W_up": w_up, "H_up": h_up,
        "WxH_up": f"{w_up}x{h_up}",
        "n_pages": tags["_n_pages"], "n_subifds": tags["_n_subifds"],
    }
    for c in range(ch):
        v = flat[:, c]
        lo, hi = np.percentile(v, [0.1, 99.9])
        rec.update({
            f"c{c}_min": int(v.min()) if dmax is not None else float(v.min()),
            f"c{c}_max": int(v.max()) if dmax is not None else float(v.max()),
            f"c{c}_p0.1": float(lo), f"c{c}_p99.9": float(hi),
            f"c{c}_mean": float(v.mean()),
            f"c{c}_frac_zero": float((v == 0).mean()),
            f"c{c}_n_unique": int(np.unique(v).size),
        })
        if dmax is not None:
            rec[f"c{c}_frac_at_dtype_max"] = float((v == dmax).mean())
            rec[f"c{c}_frac_mult_257"] = float((v.astype(np.int64) % 257 == 0).mean())
            rec[f"c{c}_unused_low_bits"] = _trailing_zero_bits(v)
    if dmax is not None:
        rec["dtype_max"] = int(dmax)
        rec["sat_frac_any_at_dtype_max"] = float((flat == dmax).any(axis=1).mean())
        rec["sat_frac_all_at_dtype_max"] = float((flat == dmax).all(axis=1).mean())
    fmax = flat.max()
    rec["file_max"] = float(fmax)
    rec["sat_frac_any_at_file_max"] = float((flat == fmax).any(axis=1).mean())

    if ch == 3 and dmax is not None:
        x = flat.astype(np.float64) / dmax
        unsat = ~(flat == dmax).any(axis=1)
        means = x[unsat].mean(axis=0)
        rec.update({f"grayworld_gain_c{c}": float(means[1] / max(means[c], 1e-12)) for c in range(3)})
        y = x[unsat] @ LUMA
        rec["Y_median"] = float(np.median(y))
        rec["Y_p99"] = float(np.percentile(y, 99))

    for name in NAMED_TAGS:
        v = tags.get(name)
        rec[f"tag_{name}"] = json.dumps(v) if isinstance(v, (list, dict)) else v
    rec["tags_all"] = json.dumps({k: v for k, v in tags.items() if not k.startswith("_")})
    return rec


def propose(df: pd.DataFrame) -> tuple[dict[str, Any], list[str], list[str]]:
    """Proposed preprocess/focal config, evidence lines, and flags."""
    ev: list[str] = []
    flags: list[str] = []
    prop: dict[str, Any] = {}
    n = len(df)

    photometric = Counter(df["tag_PhotometricInterpretation"].astype(str))
    spp = Counter(df["tag_SamplesPerPixel"].astype(str))
    has_cfa = df["tag_CFAPattern"].notna().any()
    if set(photometric) == {"RGB"} and set(df["channels"]) == {3} and not has_cfa:
        prop["layout"] = "rgb"
        prop["cfa_pattern"] = None
        ev.append(f"layout=rgb: all {n} files have PhotometricInterpretation=RGB, SamplesPerPixel={dict(spp)}, "
                  f"3 channels, no CFAPattern tag.")
    else:
        prop["layout"] = "auto"
        flags.append(f"Layout ambiguous: Photometric={dict(photometric)}, channels={dict(Counter(df['channels']))}, "
                     f"CFA tag present={has_cfa}.")

    dtypes = Counter(df["dtype"])
    bps = Counter(df["tag_BitsPerSample"].astype(str))
    ev.append(f"dtype {dict(dtypes)}, BitsPerSample {dict(bps)}.")

    # Black level: no DNG BlackLevel tag, so infer from the minima. A sensor pedestal P is camera-wide
    # and keeps every pixel near or above P; one file whose minimum is ~0 therefore rules out P >> noise.
    ch = int(df["channels"].iloc[0])
    min_cols = [f"c{c}_min" for c in range(ch)]
    file_min = df[min_cols].min(axis=1)
    near0 = 0.001 * (df["dtype_max"].iloc[0] if "dtype_max" in df else df["file_max"].max())
    n_near0 = int((file_min <= near0).sum())
    if df["tag_BlackLevel"].notna().any():
        prop["black_level"] = "from tag"
        ev.append("black_level: BlackLevel tag present (see audit.csv).")
    else:
        zero_frac = float(np.mean([df[f"c{c}_frac_zero"].mean() for c in range(ch)]))
        corr = float(np.corrcoef(file_min, df["Y_median"])[0, 1]) if "Y_median" in df and n > 2 else float("nan")
        detail = (f"{n_near0}/{n} files have a minimum <= {near0:.0f} (0.1% of full scale) in some channel; "
                  f"per-file minimum ranges {int(file_min.min())}..{int(file_min.max())}; exact zeros are rare "
                  f"(mean fraction {zero_frac:.1e}); corr(per-file min, Y median) = {corr:.2f}, i.e. higher minima "
                  f"come from brighter scenes, not from a pedestal")
        if n_near0 > 0:
            prop["black_level"] = 0
            ev.append(f"black_level=0: no BlackLevel tag; {detail}. Consistent with data that is already "
                      "black-subtracted.")
        else:
            prop["black_level"] = file_min.min().item()
            flags.append(f"Black level uncertain: no tag and no file reaches ~0 ({detail}); proposing the dataset "
                         "minimum.")

    dmax = int(df["dtype_max"].iloc[0]) if "dtype_max" in df else None
    sat = df["sat_frac_any_at_dtype_max"] if dmax is not None else None
    if df["tag_WhiteLevel"].notna().any():
        prop["white_level"] = "from tag"
    elif dmax is not None and (df["file_max"] == dmax).any():
        prop["white_level"] = dmax
        ev.append(f"white_level={dmax}: no WhiteLevel tag; {(df['file_max'] == dmax).sum()}/{n} files reach the dtype "
                  f"max (any-channel saturation fraction median {sat.median():.2e}, max {sat.max():.2e}).")
    else:
        prop["white_level"] = float(df["file_max"].max())
        flags.append(f"White level uncertain: no tag and no file reaches the dtype max; proposing the dataset max "
                     f"{prop['white_level']}.")

    low_bits = [int(df[f"c{c}_unused_low_bits"].max()) for c in range(ch)] if dmax else []
    m257 = float(np.mean([df[f"c{c}_frac_mult_257"].mean() for c in range(ch)])) if dmax else float("nan")
    uniq = int(np.median([df[f"c{c}_n_unique"].median() for c in range(ch)]))
    ev.append(f"Bit usage: unused low bits per channel (max over files) {low_bits}; fraction of values that are "
              f"multiples of 257 {m257:.4f} (an 8-bit image expanded to 16 bits would give 1.0); median unique "
              f"values per channel {uniq}.")

    if df["tag_AsShotNeutral"].notna().any():
        prop["white_balance"] = "metadata"
        ev.append("white_balance=metadata: AsShotNeutral present.")
    else:
        # Raw camera RGB has strongly unequal channel means (gray-world gains far from 1, typically R, B ~ 1.5-2.5).
        # Median gains near 1 indicate the data were already white-balanced; outliers are then scene color.
        g = df[[f"grayworld_gain_c{c}" for c in range(3)]]
        med = g.median()
        outliers = df.loc[((g - 1).abs() > 0.3).any(axis=1), "stem"].tolist()
        gains_txt = (f"Gray-world gains (R,G,B) median {med.round(3).tolist()}, range R [{g.iloc[:, 0].min():.2f}, "
                     f"{g.iloc[:, 0].max():.2f}], B [{g.iloc[:, 2].min():.2f}, {g.iloc[:, 2].max():.2f}]; files with a "
                     f"gain off by > 30%: {outliers or 'none'}")
        if ((med - 1).abs() < 0.15).all():
            prop["white_balance"] = "none"
            ev.append(f"white_balance=none: no AsShotNeutral/WB tags. {gains_txt}. Median gains within 15% of 1 "
                      "mean the data look already white-balanced, so gray-world would only push scene color "
                      "(the outliers) toward gray. Compare the two thumbnail grids.")
        else:
            prop["white_balance"] = "grayworld"
            ev.append(f"white_balance=grayworld: no AsShotNeutral/WB tags and channel means are unbalanced. {gains_txt}.")
    ev.append(f"Exposure: median over files of Y_median={df['Y_median'].median():.4f}, Y_p99={df['Y_p99'].median():.4f} "
              f"(fraction of full scale). Default exposure_target 0.9 / P99 kept.")
    ev.append(f"Linearity (heuristic, no tag can confirm it): median luminance {df['Y_median'].median():.3f} of full "
              "scale with P99 near 0.6 is typical of linear data; gamma-encoded images usually sit around 0.3-0.5.")

    orient = Counter(df["orientation"])
    present = int(df["orientation_tag_present"].sum())
    prop["respect_orientation_tag"] = True
    ev.append(f"Orientation: tag present in {present}/{n} files; values {dict(orient)}.")
    if present == 0:
        flags.append("No Orientation tag in any file, so stored == upright by definition. Whether the stored "
                     "arrays are visually upright can only be checked on the thumbnails.")

    sizes = Counter(df["WxH_up"])
    ev.append(f"Upright sizes: {dict(sizes)}.")
    n_portrait = int((df["H_up"] > df["W_up"]).sum())
    if n_portrait == 0:
        flags.append("No portrait images in the dataset; portrait paths are covered only by unit tests.")

    for k in ("Make", "Model", "FocalLength"):
        if df[f"tag_{k}"].notna().any():
            ev.append(f"{k}: {dict(Counter(df[f'tag_{k}'].astype(str)))}.")
    if any(df[f"tag_{k}"].notna().any() for k in ("Make", "Model", "FocalLength")):
        prop["group_key"] = [k for k in ("Make", "Model", "FocalLength") if df[f"tag_{k}"].notna().any()]
    else:
        prop["group_key"] = None
        ev.append("focal.group_key=null: no Make/Model/FocalLength tags (no EXIF), so all files form one group.")
        flags.append("No camera metadata: one focal group is an assumption. If the images come from different "
                     "cameras or zoom settings, the pooled f is meaningless; per_image (the default) is unaffected.")
    return prop, ev, flags


def _thumb(srgb: np.ndarray) -> np.ndarray:
    h, w = srgb.shape[:2]
    s = THUMB_LONG_SIDE / max(h, w)
    return cv2.resize(srgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _grid(thumbs: list[tuple[str, np.ndarray]], cols: int = 4) -> np.ndarray:
    cell = THUMB_LONG_SIDE + 8
    rows = (len(thumbs) + cols - 1) // cols
    canvas = np.full((rows * (cell + 24), cols * cell, 3), 255, np.uint8)
    for i, (name, t) in enumerate(thumbs):
        r, c = divmod(i, cols)
        y0, x0 = r * (cell + 24), c * cell
        canvas[y0 + 24:y0 + 24 + t.shape[0], x0:x0 + t.shape[1]] = t
        cv2.putText(canvas, name, (x0 + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def make_thumbnails(cfg: Config, df: pd.DataFrame, prop: dict[str, Any], out: Path) -> tuple[list[str], list[str]]:
    """8 thumbnails (4 landscape + 4 portrait, filled from the other orientation if short)."""
    rng = random.Random(cfg.seed)
    land = df[df["W_up"] >= df["H_up"]]["file"].tolist()
    port = df[df["H_up"] > df["W_up"]]["file"].tolist()
    pick_p = rng.sample(port, min(4, len(port)))
    pick_l = rng.sample(land, min(8 - len(pick_p), len(land)))
    picks = pick_l + pick_p
    notes = [f"{len(pick_l)} landscape + {len(pick_p)} portrait (dataset has {len(land)} landscape, {len(port)} portrait)."]

    black = prop["black_level"] if isinstance(prop["black_level"], (int, float, list)) else 0
    white = prop["white_level"] if isinstance(prop["white_level"], (int, float)) else 65535
    tdir = out / "thumbs"
    tdir.mkdir(parents=True, exist_ok=True)
    grids: dict[str, list] = {wb: [] for wb in dict.fromkeys([prop["white_balance"], "grayworld", "none"])}
    records = []
    for f in picks:
        arr, tags = load_tiff(cfg.data_dir / f)
        resolve_layout(arr, tags, prop["layout"])
        for wb in grids:
            srgb, rec = linear_to_srgb(
                arr, black_level=black, white_level=white, white_balance=wb,
                exposure_percentile=cfg.preprocess.exposure_percentile,
                exposure_target=cfg.preprocess.exposure_target,
                exposure_gain_max=cfg.preprocess.exposure_gain_max,
            )
            up = to_upright(srgb, orientation_of(tags))
            t = _thumb(up)
            stem = Path(f).stem
            if wb == prop["white_balance"]:
                cv2.imwrite(str(tdir / f"{stem}.png"), cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
                records.append({"file": f, **rec})
            grids[wb].append((stem, t))
    names = []
    for wb, items in grids.items():
        name = f"thumbs_grid_wb-{wb}.png"
        cv2.imwrite(str(out / name), cv2.cvtColor(_grid(items), cv2.COLOR_RGB2BGR))
        names.append(name)
    pd.DataFrame(records).to_csv(out / "thumbs_preprocess.csv", index=False)
    return names, notes


def run_audit(cfg: Config) -> Path:
    out = cfg.out_dir / "audit"
    out.mkdir(parents=True, exist_ok=True)
    files = list_files(cfg)
    sample = sample_files(files, cfg.seed)
    rows = [audit_file(p, cfg.data_dir) for p in sample]
    df = pd.DataFrame(rows)
    df.to_csv(out / "audit.csv", index=False)

    prop, ev, flags = propose(df)
    grids, thumb_notes = make_thumbnails(cfg, df, prop, out)
    write_report(out / "audit_report.md", cfg, df, len(files), prop, ev, flags, grids, thumb_notes)
    return out


def _yaml_val(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v).lower()
    return json.dumps(v) if isinstance(v, (list, str)) else str(v)


def write_report(path: Path, cfg: Config, df: pd.DataFrame, n_total: int, prop: dict[str, Any],
                 ev: list[str], flags: list[str], grids: list[str], thumb_notes: list[str]) -> None:
    p = cfg.preprocess
    ch = int(df["channels"].iloc[0])
    lines = [
        "# Dataset audit",
        "",
        f"- data_dir: `{cfg.data_dir}`",
        f"- files matching `{cfg.paths.glob}`: {n_total}; audited: {len(df)}"
        + (" (all)" if len(df) == n_total else f" (seeded sample, seed={cfg.seed})"),
        "- per-file details: `audit.csv` (all tags in column `tags_all`)",
        "",
        "## Proposed config",
        "",
        "```yaml",
        "preprocess:",
        f"  layout: {_yaml_val(prop['layout'])}",
        f"  cfa_pattern: {_yaml_val(prop.get('cfa_pattern'))}",
        f"  black_level: {_yaml_val(prop['black_level'])}",
        f"  white_level: {_yaml_val(prop['white_level'])}",
        f"  white_balance: {_yaml_val(prop['white_balance'])}",
        f"  exposure_percentile: {p.exposure_percentile}",
        f"  exposure_target: {p.exposure_target}",
        f"  exposure_gain_max: {p.exposure_gain_max}",
        f"  respect_orientation_tag: {_yaml_val(prop['respect_orientation_tag'])}",
        "focal:",
        f"  group_key: {_yaml_val(prop['group_key'])}",
        "```",
        "",
        "## Evidence",
        "",
        *[f"- {e}" for e in ev],
        "",
        "## Flags",
        "",
        *([f"- {f}" for f in flags] or ["- none"]),
        "",
        "## Counts",
        "",
        "| upright W×H | count |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in Counter(df["WxH_up"]).items()],
        "",
        "| Orientation | count |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in Counter(df["orientation"]).items()],
        "",
        "## Per-file summary",
        "",
        "| file | W×H | " + " | ".join(f"c{c} min / p0.1 / p99.9 / max" for c in range(ch))
        + " | sat (any ch @ max) | gray-world gains R,G,B | Y median |",
        "|---|---|" + "---|" * ch + "---|---|---|",
    ]
    for _, r in df.iterrows():
        cells = " | ".join(f"{r[f'c{c}_min']} / {r[f'c{c}_p0.1']:.0f} / {r[f'c{c}_p99.9']:.0f} / {r[f'c{c}_max']}"
                           for c in range(ch))
        gw = ", ".join(f"{r.get(f'grayworld_gain_c{c}', float('nan')):.2f}" for c in range(3))
        lines.append(f"| {r['file']} | {r['W']}×{r['H']} | {cells} | {r.get('sat_frac_any_at_dtype_max', float('nan')):.2e} "
                     f"| {gw} | {r.get('Y_median', float('nan')):.4f} |")
    lines += [
        "",
        "## Thumbnails",
        "",
        f"Provisional sRGB with the proposed levels, exposure gain to P99 = 0.9, upright frame. "
        f"Grids with gray-world and with no WB, for judging whether WB is needed (proposed: {prop['white_balance']}).",
        "",
        *[f"- {n}" for n in thumb_notes],
        *[f"- `{g}`" for g in grids],
        f"- individual (WB {prop['white_balance']}): `thumbs/<stem>.png`; preprocessing records: `thumbs_preprocess.csv`",
        "",
    ]
    notes = path.parent / "visual_notes.md"
    if notes.exists():
        lines += ["## Visual check (hand-written, `visual_notes.md`)", "", notes.read_text().strip(), ""]
    path.write_text("\n".join(lines))
