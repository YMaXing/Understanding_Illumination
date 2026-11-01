"""Stage B: detector + deliverables per image (spec M3/M4). Reads only the Stage A cache.

Per run (out_dir/stage_b/<run>/), all arrays in the STORED frame:
  <stem>.npz   w (original res, float16), w_net, q, L_over_rho (network res, float16),
               D (network res, float32), sky_mask (network res, bool)
  <stem>.json  detector scalars, focal, geometry, provenance
  manifest.parquet (+ manifest.csv), config_resolved.yaml, viz/<stem>.png (upright)

With geometry.enabled (docs/geometry_channels.md section 3.5) the geometric-edge channels replace the detector call:
  <stem>.npz   w, w_net (combined, elementwise min), <name>_w_net, <name>_strength, <name>_<array> per channel,
               q, L_over_rho (when the laplacian channel runs), n_hat (H, W, 3; components in the UPRIGHT camera
               frame), sigma_theta, normal_valid (when a channel computed the normals), D, sky_mask
  <stem>_<name>_candidates.parquet, <stem>_<name>_<table>.parquet   upright network coordinates
  <stem>.json  + channels (per-channel scalars), combined, params.geometry, provenance.geometry_version
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import Config, dense_fit_params, detector_params
from .detector import compute_weights
from .io_tiff import to_upright
from .outputs import deliverable_w, net_to_stored
from .stage_a import stage_a_dir

STAGE_B_VERSION = 1
GEOMETRY_VERSION = 1  # recorded only for geometry.enabled runs


def run_dir(cfg: Config, run_name: str | None = None) -> Path:
    return cfg.out_dir / "stage_b" / (run_name or cfg.run.name)


def cached_stems(cfg: Config) -> list[str]:
    d = stage_a_dir(cfg)
    return sorted(p.stem for p in d.glob("*.npz") if (d / f"{p.stem}.json").exists() and not p.stem.endswith(".tmp"))


def select_stems(cfg: Config, sample: int | None, limit: int | None, only: list[str] | None = None) -> list[str]:
    stems = cached_stems(cfg)
    if not stems:
        raise FileNotFoundError(f"no Stage A cache under {stage_a_dir(cfg)}; run stage-a first")
    if only:
        missing = sorted(set(only) - set(stems))
        if missing:
            raise FileNotFoundError(f"no Stage A cache for {missing} under {stage_a_dir(cfg)}")
        return sorted(only)
    from .audit import list_files

    missing = sorted({f.stem for f in list_files(cfg)} - set(stems))
    if missing:
        warnings.warn(f"{len(missing)} data files have no Stage A cache (e.g. {missing[:3]}); run stage-a first")
    if sample is not None:
        if sample >= len(stems):
            if sample > len(stems):
                warnings.warn(f"--sample {sample} > {len(stems)} cached images; using all")
        else:
            stems = sorted(random.Random(cfg.seed).sample(stems, sample))
    if limit is not None:
        stems = stems[:limit]
    return stems


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _params_record(cfg: Config) -> dict[str, Any]:
    det = dataclasses.asdict(cfg.detector)
    if det["a_min"] is None:  # threshold floor off: keep the record identical to pre-T4 runs
        det.pop("a_min")
        det.pop("lambda_ref")
    if not det.get("eps_floor"):  # eps floor off: identical to pre-eps-floor runs
        det.pop("eps_floor")
    if not det.get("scale_band"):  # scale-ratio test off: identical to earlier runs
        det.pop("scale_band", None)
        det.pop("band_factor", None)
        det.pop("scale_band_mode", None)
    elif det.get("scale_band_mode") == "replace":  # identical to runs made before the veto mode existed
        det.pop("scale_band_mode", None)
    if det.get("prefilter", "none") == "none":  # prefilter off: identical to pre-prefilter runs
        for k in ("prefilter", "tri_sigma_s", "tri_sigma_r", "tri_sigma_g", "tri_shat"):
            det.pop(k, None)
    return {"detector": det, "sky": dataclasses.asdict(cfg.sky)}


def _inherit_stage_a_settings(cfg: Config) -> None:
    """For a Stage A variant, take preprocess/network/geocalib/focal/da3 from the variant's own snapshot,
    so the Stage B run config records what actually produced its inputs."""
    if not cfg.paths.stage_a_variant:
        return
    from .config import load_config

    snap = stage_a_dir(cfg) / "config_resolved.yaml"
    if not snap.exists():
        raise FileNotFoundError(f"{snap} missing; run stage-a for variant {cfg.paths.stage_a_variant!r} first")
    sa_cfg = load_config(snap)
    for section in ("preprocess", "network", "geocalib", "focal", "da3"):
        setattr(cfg, section, getattr(sa_cfg, section))


def _check_run_config(cfg: Config, rd: Path, force: bool) -> None:
    """A run directory holds exactly one parameter set; refuse to mix unless --force."""
    from .config import load_config

    v = cfg.paths.stage_a_variant
    if v and v not in rd.name:
        raise ValueError(f"run name {rd.name!r} must include the Stage A variant {v!r}")
    snap = rd / "config_resolved.yaml"
    if snap.exists() and not force:
        old, new = load_config(snap).to_dict(), cfg.to_dict()  # load_config fills keys added since

        def key(d):
            return ({k: d[k] for k in ("detector", "sky", "preprocess", "geometry")}
                    | {"variant": d["paths"]["stage_a_variant"]})

        if key(old) != key(new):
            raise ValueError(f"run {rd.name!r} exists with different detector/sky/preprocess/geometry/variant settings; "
                             "use another --run-name or --force")
    cfg.snapshot(snap)


def _detector_record(sc: dict[str, Any]) -> dict[str, Any]:
    """The JSON "detector" block from compute_weights scalars."""
    return ({k: sc[k] for k in (
        "s_hat", "s_hat_raw", "s_hat_clamped", "tau", "eps", "rho_floor", "r", "R", "sky_frac",
        "n_bad_depth", "n_noise_valid", "frac_w_lt_0.5", "frac_w_lt_0.9", "frac_w_lt_1")}
        | {k: sc[k] for k in ("tau_noise", "tau_min", "tau_min_over_tau_noise", "a_min", "lambda_ref",
                              "eps_noise", "eps_min", "eps_floor", "prefilter", "tri_shat",
                              "s_hat_filtered_response", "scale_band", "band_factor", "scale_band_mode",
                              "band_gain") if k in sc})


def _meta_common(cfg: Config, stem: str, meta_a: dict[str, Any]) -> dict[str, Any]:
    """The JSON "focal", "geometry", "params" and "provenance" blocks."""
    sa = stage_a_dir(cfg)
    fr, net = meta_a["frames"], meta_a["network"]
    gc, fo = meta_a["geocalib"], meta_a["focal"]
    return {
        "focal": {
            "f_active": fo["f_active"], "f_n": fo["f_n"], "mode": fo["mode"],
            "f_per_image": fo["f_per_image"], "f_pooled": fo["f_pooled"],
            "pooled_meaningful": fo["pooled_meaningful"],
            "fx": gc["fx"], "fy": gc["fy"],
            "focal_uncertainty": gc["focal_uncertainty"],
            "focal_rel_uncertainty": gc["focal_uncertainty"] / fo["f_per_image"],
            "vfov_deg": gc["vfov_deg"], "vfov_uncertainty_deg": gc["vfov_uncertainty_deg"],
            "roll_deg": gc["roll_deg"], "pitch_deg": gc["pitch_deg"],
            "roll_uncertainty_deg": gc["roll_uncertainty_deg"],
            "pitch_uncertainty_deg": gc["pitch_uncertainty_deg"],
            "units": fo["units"], "canonical_focal": fo["canonical_focal"],
        },
        "geometry": {
            "s_x": net["s_x"], "s_y": net["s_y"], "W_n": net["W_n"], "H_n": net["H_n"],
            "orientation_tag": fr["orientation_tag"], "orientation_applied": fr["orientation_applied"],
            "stored_WxH": fr["stored_WxH"], "upright_WxH": fr["upright_WxH"],
            "frames": ("npz arrays are in the stored frame (aligned with the TIFF); w is at the original "
                       "resolution, all other arrays at network resolution; viz/ is upright"),
        },
        "params": _params_record(cfg),
        "provenance": {
            "stage_a_json": str((sa / f"{stem}.json").relative_to(cfg.out_dir)),
            "stage_a_npz_sha256": _sha256(sa / f"{stem}.npz"),
            "stage_a_created": meta_a["created"],
            "stage_a_version": meta_a["provenance"]["stage_a_version"],
            "stage_b_version": STAGE_B_VERSION,
            "repo_git": meta_a["provenance"].get("repo_git"),
        },
    }


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _channel_params(cfg: Config, name: str) -> dict[str, Any]:
    if name == "laplacian":
        return dataclasses.asdict(detector_params(cfg))
    return dataclasses.asdict(getattr(cfg.geometry, name))


def run_geometry(cfg: Config, D_up: np.ndarray, sky_up: np.ndarray | None, meta_a: dict[str, Any]):
    """(context, channel results, combined w_net) at network resolution, upright."""
    from .geom.camera import Intrinsics
    from .geom.channels import CHANNELS, combine
    from .geom.context import GeometryContext

    ctx = GeometryContext(D_up, sky_up, Intrinsics.from_stage_a(meta_a), detector_params(cfg), dense_fit_params(cfg))
    results = [CHANNELS[name]().run(ctx, _channel_params(cfg, name)) for name in cfg.geometry.channels]
    return ctx, results, combine(results, cfg.geometry.combine)


def _process_one_geometry(cfg: Config, stem: str, rd: Path, meta_a: dict[str, Any], D_up: np.ndarray,
                          sky_up: np.ndarray | None) -> dict[str, Any]:
    ctx, results, w_net = run_geometry(cfg, D_up, sky_up, meta_a)
    fr = meta_a["frames"]
    orient = fr["orientation_applied"]
    w_clip = min(r.w_min for r in results)
    w_full = deliverable_w(w_net, tuple(fr["upright_WxH"]), orient, w_clip)
    if w_full.shape[:2] != (fr["stored_WxH"][1], fr["stored_WxH"][0]):
        raise RuntimeError(f"{stem}: deliverable shape {w_full.shape} != stored {fr['stored_WxH']}")

    def st(a: np.ndarray, dtype=np.float16) -> np.ndarray:
        a = net_to_stored(a, orient)
        return a if a.dtype == bool else a.astype(dtype)

    arrays = {"w": w_full.astype(np.float16), "w_net": st(w_net)}
    by_name = {r.name: r for r in results}
    if "laplacian" in by_name:  # unprefixed as in detector runs, so viz.render works unchanged
        arrays["q"] = st(by_name["laplacian"].arrays["q"])
        arrays["L_over_rho"] = st(by_name["laplacian"].arrays["L_over_rho"])
    for r in results:
        arrays[f"{r.name}_w_net"] = st(r.w_net)
        if r.strength is not None:
            arrays[f"{r.name}_strength"] = st(r.strength)
        for k, a in r.arrays.items():
            arrays[f"{r.name}_{k}"] = st(a)
    if "_normals" in ctx.__dict__:  # a channel computed the normals: save them (vector components stay upright)
        arrays["n_hat"] = st(ctx.n_hat)
        arrays["sigma_theta"] = st(ctx.sigma_theta)
        arrays["normal_valid"] = st(ctx.normal_valid)
    arrays["D"] = st(D_up, np.float32)
    arrays["sky_mask"] = st(ctx.S)
    tmp = rd / f"{stem}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(rd / f"{stem}.npz")
    for r in results:
        if r.candidates is not None:
            r.candidates.to_parquet(rd / f"{stem}_{r.name}_candidates.parquet", index=False)
        for tname, t in r.tables.items():
            t.to_parquet(rd / f"{stem}_{r.name}_{tname}.parquet", index=False)

    if "laplacian" in by_name:
        det = _detector_record(by_name["laplacian"].scalars)
    else:
        dp = ctx.disparity
        det = {"rho_floor": dp.rho_floor, "sky_frac": dp.sky_frac, "n_bad_depth": int(dp.bad.sum())}
    combined = {"frac_w_lt_0.5": float((w_net < 0.5).mean()), "frac_w_lt_0.9": float((w_net < 0.9).mean()),
                "frac_w_lt_1": float((w_net < 1).mean()), "w_min_clip": w_clip,
                "channels": list(cfg.geometry.channels), "combine": cfg.geometry.combine}
    common = _meta_common(cfg, stem, meta_a)
    common["geometry"]["frames"] += ("; n_hat vector components are in the upright camera frame (x right, y down, "
                                     "z forward); candidate tables use upright network-resolution pixel coordinates")
    from .geom.fold import params_record

    geo = dataclasses.asdict(cfg.geometry)
    geo["fold"] = params_record(geo["fold"])
    common["params"]["geometry"] = geo
    common["provenance"]["geometry_version"] = GEOMETRY_VERSION
    meta = {
        "stem": stem,
        "file": meta_a["file"],
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "run": rd.name,
        "detector": det,
        "normals_computed": "_normals" in ctx.__dict__,
        "combined": combined,
        "channels": _jsonable({r.name: r.scalars for r in results}),
        **common,
    }
    (rd / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return meta


def process_one(cfg: Config, stem: str, rd: Path) -> dict[str, Any]:
    sa = stage_a_dir(cfg)
    meta_a = json.loads((sa / f"{stem}.json").read_text())
    cache = np.load(sa / f"{stem}.npz")
    D_up = cache["D"]
    sky_up = cache["sky_prob"].astype(np.float64) if "sky_prob" in cache else None
    if cfg.geometry.enabled:
        return _process_one_geometry(cfg, stem, rd, meta_a, D_up, sky_up)

    params = detector_params(cfg)
    out = compute_weights(D_up, sky_up, params)
    sc = out["scalars"]

    fr, net = meta_a["frames"], meta_a["network"]
    orient = fr["orientation_applied"]
    w_up_size = tuple(fr["upright_WxH"])
    w_full = deliverable_w(out["w"], w_up_size, orient, cfg.detector.w_min)
    if w_full.shape[:2] != (fr["stored_WxH"][1], fr["stored_WxH"][0]):
        raise RuntimeError(f"{stem}: deliverable shape {w_full.shape} != stored {fr['stored_WxH']}")

    arrays = {
        "w": w_full.astype(np.float16),
        "w_net": net_to_stored(out["w"], orient).astype(np.float16),
        "q": net_to_stored(out["q"], orient).astype(np.float16),
        "L_over_rho": net_to_stored(out["L_over_rho"], orient).astype(np.float16),
        "D": net_to_stored(D_up, orient).astype(np.float32),
        "sky_mask": net_to_stored(out["sky_mask"], orient),
    }
    tmp = rd / f"{stem}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(rd / f"{stem}.npz")

    meta = {
        "stem": stem,
        "file": meta_a["file"],
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "run": rd.name,
        "detector": _detector_record(sc),
        **_meta_common(cfg, stem, meta_a),
    }
    (rd / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return meta


def run_stage_b(cfg: Config, run_name: str | None = None, force: bool = False,
                sample: int | None = None, limit: int | None = None, viz: bool = True,
                stems: list[str] | None = None) -> dict[str, Any]:
    from .manifest import write_manifest
    from .viz import render
    from .viz_fold import render_fold

    rd = run_dir(cfg, run_name)
    _inherit_stage_a_settings(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    _check_run_config(cfg, rd, force)
    stems = select_stems(cfg, sample, limit, stems)
    status = {"run": rd.name, "n_images": len(stems), "n_done": 0, "n_skipped": 0}
    for stem in tqdm(stems, desc=f"stage-b {rd.name}"):
        if not force and (rd / f"{stem}.npz").exists() and (rd / f"{stem}.json").exists():
            status["n_skipped"] += 1
        else:
            process_one(cfg, stem, rd)
            status["n_done"] += 1
        g = cfg.geometry
        if viz and (not g.enabled or "laplacian" in g.channels) and (force or not (rd / "viz" / f"{stem}.png").exists()):
            render(cfg, rd, stem)  # detector sheet (with the combined w in geometry runs)
        if viz and g.enabled and "fold" in g.channels and (force or not (rd / "viz_fold" / f"{stem}.png").exists()):
            render_fold(cfg, rd, stem)
    status["manifest"] = str(write_manifest(rd))
    return status


def image_stems(rd: Path) -> list[str]:
    """Stems with both <stem>.npz and <stem>.json (excludes manifest files and temporaries)."""
    return sorted(p.stem for p in rd.glob("*.npz") if not p.stem.endswith(".tmp") and (rd / f"{p.stem}.json").exists())


def load_upright(cfg: Config, rd: Path, stem: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Saved Stage B arrays converted back to the upright frame, plus the image JSON."""
    meta = json.loads((rd / f"{stem}.json").read_text())
    o = meta["geometry"]["orientation_applied"]
    z = np.load(rd / f"{stem}.npz")
    return {k: to_upright(z[k], o) for k in z.files}, meta
