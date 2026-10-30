"""Stage A: preprocessing, GeoCalib and DA3 inference with a resumable per-image cache (spec M2).

Cache, all in the upright frame:
  out_dir/stage_a/<stem>.npz        D float32 (H_n, W_n), sky_prob float16 (H_n, W_n)
  out_dir/stage_a/<stem>.json       A1-A4 metadata and provenance
  out_dir/stage_a/srgb/<stem>.png   upright full-resolution sRGB
  out_dir/stage_a/focal_pooled.json per-group pooled focal (rewritten after every run)
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import zlib
import warnings
from importlib import metadata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from . import PROJECT_ROOT
from .audit import list_files
from .config import Config
from .geometry import focal_at_network, network_size, scale_factors
from .io_tiff import load_tiff, orientation_of, to_upright, upright_size
from .preprocess import linear_to_srgb_float, quantize, resolve_layout

INTERP = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
STAGE_A_VERSION = 2  # 2: GeoCalib reseeded per image


def stage_a_dir(cfg: Config) -> Path:
    v = cfg.paths.stage_a_variant
    return cfg.out_dir / "stage_a" / v if v else cfg.out_dir / "stage_a"


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance() -> dict[str, Any]:
    pkgs = {}
    for name in ("torch", "numpy", "opencv-python", "opencv-python-headless", "tifffile", "imagecodecs",
                 "depth-anything-3", "geocalib", "kornia", "xformers"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    try:
        geocalib_src = json.loads(metadata.distribution("geocalib").read_text("direct_url.json") or "{}")
        geocalib_commit = geocalib_src.get("vcs_info", {}).get("commit_id")
    except metadata.PackageNotFoundError:
        geocalib_commit = None
    da3_dir = PROJECT_ROOT.parent / "DA3" / "Depth-Anything-3"
    return {
        "packages": pkgs,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "da3_git": _git(["rev-parse", "HEAD"], da3_dir),
        "geocalib_commit": geocalib_commit,
        "repo_git": _git(["rev-parse", "HEAD"], PROJECT_ROOT),
        "repo_dirty": bool(_git(["status", "--porcelain", "--", "."], PROJECT_ROOT)),
        "stage_a_version": STAGE_A_VERSION,
    }


def prepare(cfg: Config, path: Path) -> dict[str, Any]:
    """A1 + A2: linear TIFF -> upright full-res sRGB and network-resolution uint8 input."""
    p = cfg.preprocess
    raw, tags = load_tiff(path)
    layout = resolve_layout(raw, tags, p.layout)
    if layout != "rgb":
        raise NotImplementedError("mosaic input: demosaic is not implemented (the M1 audit found RGB data)")
    srgb_f, rec = linear_to_srgb_float(
        raw, black_level=p.black_level, white_level=p.white_level, white_balance=p.white_balance,
        exposure_percentile=p.exposure_percentile, exposure_target=p.exposure_target,
        exposure_gain_max=p.exposure_gain_max, exposure_mode=p.exposure_mode,
    )
    orient = orientation_of(tags) if p.respect_orientation_tag else 1
    h_s, w_s = raw.shape[:2]
    w_up, h_up = upright_size(w_s, h_s, orient)
    w_n, h_n = network_size(w_up, h_up, cfg.network.long_side, cfg.network.multiple)
    interp = INTERP[cfg.network.resize_interp]
    if p.dither == "none":
        # Baseline path: quantize at full res, resize the uint8 image.
        up = to_upright(quantize(srgb_f), orient)
        net = cv2.resize(up, (w_n, h_n), interpolation=interp)
        rec["dither"] = "none"
    else:
        # Dither at each final quantization: full res (GeoCalib, viz) and network res (DA3). The network
        # input is resized in float so the second quantization does not undo the first dither.
        seed = [cfg.seed, zlib.crc32(path.stem.encode())]
        rng_full, rng_net = (np.random.default_rng(seed + [k]) for k in (0, 1))
        up = to_upright(quantize(srgb_f, p.dither, rng_full), orient)
        net_f = cv2.resize(to_upright(srgb_f, orient).astype(np.float32), (w_n, h_n), interpolation=interp)
        net = quantize(net_f.astype(np.float64), p.dither, rng_net)
        rec.update({"dither": p.dither, "dither_seed": seed, "dither_applied_at": ["full_res", "network_res"]})
    assert up.shape[:2] == (h_up, w_up)
    s_x, s_y = scale_factors(w_up, h_up, w_n, h_n)
    meta = {
        "frames": {
            "stored_WxH": [w_s, h_s],
            "orientation_tag": orientation_of(tags),
            "orientation_applied": orient,
            "upright_WxH": [w_up, h_up],
            "note": "cache arrays are upright; deliverables are converted to the stored frame in Stage B",
        },
        "preprocess": {"layout": layout, **rec},
        "network": {
            "W_n": w_n, "H_n": h_n, "s_x": s_x, "s_y": s_y,
            "resize_interp": cfg.network.resize_interp,
        },
        "tags": {k: tags.get(k) for k in ("Make", "Model", "FocalLength") if k in tags},
    }
    return {"srgb_up": up, "net": net, "meta": meta}


def group_of(meta: dict[str, Any], group_key: list[str] | None) -> str:
    if not group_key:
        return "all"
    return "|".join(f"{k}={meta['tags'].get(k)}" for k in group_key)


def update_focal(cfg: Config) -> dict[str, Any]:
    """Pool f per group over every cached image, then rewrite each JSON's focal block."""
    d = stage_a_dir(cfg)
    metas = {p: json.loads(p.read_text()) for p in sorted(d.glob("*.json")) if p.name != "focal_pooled.json"}
    groups: dict[str, list[float]] = {}
    for m in metas.values():
        groups.setdefault(m["focal"]["group"], []).append(m["focal"]["f_per_image"])
    pooled = {g: float(np.median(v)) for g, v in groups.items()}
    meaningful = cfg.focal.group_key is not None
    summary = {
        "group_key": cfg.focal.group_key,
        "pooled_meaningful": meaningful,
        "note": None if meaningful else (
            "group_key is null and the images may come from different cameras/zoom settings, "
            "so the pooled f is recorded for QA only and must not be used as the active focal"),
        "groups": {g: {"n": len(groups[g]), "f_pooled": pooled[g], "f_min": min(groups[g]),
                       "f_max": max(groups[g])} for g in groups},
    }
    (d / "focal_pooled.json").write_text(json.dumps(summary, indent=2))
    for p, m in metas.items():
        fo, net, gc = m["focal"], m["network"], m["geocalib"]
        fo["f_pooled"] = pooled[fo["group"]]
        fo["pooled_n"] = len(groups[fo["group"]])
        fo["pooled_meaningful"] = meaningful
        fo["mode"] = cfg.focal.mode
        if cfg.focal.mode == "per_image":
            fo["f_active"] = fo["f_per_image"]
            fo["f_n"] = focal_at_network(gc["fx"], gc["fy"], net["s_x"], net["s_y"])
        else:
            fo["f_active"] = fo["f_pooled"]
            fo["f_n"] = focal_at_network(fo["f_pooled"], fo["f_pooled"], net["s_x"], net["s_y"])
        p.write_text(json.dumps(m, indent=2))
    return summary


def run_stage_a(cfg: Config, force: bool = False, limit: int | None = None,
                stems: list[str] | None = None) -> dict[str, Any]:
    d = stage_a_dir(cfg)
    (d / "srgb").mkdir(parents=True, exist_ok=True)
    cfg.snapshot(d / "config_resolved.yaml")
    files = list_files(cfg)
    if stems:
        missing = set(stems) - {f.stem for f in files}
        if missing:
            raise FileNotFoundError(f"stems not found under data_dir: {sorted(missing)}")
        files = [f for f in files if f.stem in set(stems)]
    if limit is not None:
        files = files[:limit]
    todo = [f for f in files if force or not ((d / f"{f.stem}.npz").exists() and (d / f"{f.stem}.json").exists())]
    stems = [f.stem for f in files]
    if len(set(stems)) != len(stems):
        raise ValueError("duplicate file stems under data_dir; cache keys would collide")

    status = {"n_files": len(files), "n_skipped": len(files) - len(todo), "n_done": 0}
    if todo:
        device = resolve_device(cfg.device)
        from .calib import Calibrator
        from .depth import DepthEstimator

        calib = Calibrator(cfg.geocalib.weights, cfg.geocalib.camera_model, device, cfg.seed)
        depth = DepthEstimator(cfg.da3.model_id, cfg.da3.process_res, cfg.da3.process_res_method, device, cfg.seed)
        prov = provenance()

        # Batch by network shape so DA3 never center-crops mixed sizes; flush when a batch is full.
        pending: dict[tuple, list[tuple[Path, dict]]] = {}

        def flush(shape: tuple) -> None:
            chunk = pending.pop(shape)
            if cfg.da3.tta == "none":
                preds = depth.predict([item["net"] for _, item in chunk])
            else:
                from .depth import tta_predict

                preds = [tta_predict(depth, item["net"], cfg.da3.tta_shift_set, cfg.da3.tta_feather_px)
                         for _, item in chunk]
            for (f, item), pred in zip(chunk, preds):
                _save(cfg, f, item, pred, prov)
                status["n_done"] += 1

        for f in tqdm(todo, desc="stage-a"):
            item = prepare(cfg, f)
            item["meta"]["geocalib"] = calib.calibrate(item["srgb_up"])
            pending.setdefault(item["net"].shape, []).append((f, item))
            if len(pending[item["net"].shape]) >= cfg.da3.batch_size:
                flush(item["net"].shape)
        for shape in list(pending):
            flush(shape)
    status["focal"] = update_focal(cfg)
    return status


def _save(cfg: Config, f: Path, item: dict[str, Any], pred: dict[str, Any], prov: dict[str, Any]) -> None:
    d = stage_a_dir(cfg)
    meta = item["meta"]
    gc = meta["geocalib"]
    meta = {
        "stem": f.stem,
        "file": str(f.relative_to(cfg.data_dir)),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        **meta,
        "da3": pred["record"],
        "focal": {
            "group": group_of(meta, cfg.focal.group_key),
            "f_per_image": (gc["fx"] + gc["fy"]) / 2.0,
            "units": "upright full-resolution pixels",
            "canonical_focal": cfg.da3.canonical_focal,
            "metric_depth": "Z = f_n * D / canonical_focal (f_n at network resolution)",
        },
        "config": {k: cfg.to_dict()[k] for k in ("preprocess", "network", "geocalib", "focal", "da3", "seed")},
        "provenance": prov,
    }
    arrays = {"D": pred["D"]}
    if pred["sky_prob"] is not None:
        arrays["sky_prob"] = pred["sky_prob"]
    if "passes" in pred:  # TTA: keep every pass, unshifted/unflipped, NaN where not covered
        (d / "tta").mkdir(exist_ok=True)
        np.savez_compressed(
            d / "tta" / f"{f.stem}_passes.npz",
            D=np.stack([p["D"] for p in pred["passes"]]).astype(np.float32),
            sky_prob=np.stack([p["sky"] for p in pred["passes"]]).astype(np.float16)
            if pred["passes"][0]["sky"] is not None else np.zeros(0),
            dx=np.array([p["dx"] for p in pred["passes"]]), dy=np.array([p["dy"] for p in pred["passes"]]),
            flip=np.array([p["flip"] for p in pred["passes"]]), scale=np.array([p["scale"] for p in pred["passes"]]))
    tmp = d / f"{f.stem}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(d / f"{f.stem}.npz")
    cv2.imwrite(str(d / "srgb" / f"{f.stem}.png"), cv2.cvtColor(item["srgb_up"], cv2.COLOR_RGB2BGR))
    (d / f"{f.stem}.json").write_text(json.dumps(meta, indent=2))
