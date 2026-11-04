"""M0 smoke test: run DA3METRIC-LARGE and GeoCalib on one 8-bit image and check API facts.

Usage:
    env -u PYTHONPATH uv run python scripts/smoke_m0.py [IMAGE] [--out DIR]

Each check prints PASS/FAIL/INFO; a JSON summary is written to DIR/smoke_m0.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = HERE.parent / "DA3/Depth-Anything-3/assets/examples/SOH/000.png"
DEFAULT_OUT = HERE / "baseline_output/m0_smoke"
DA3_MODEL_ID = "depth-anything/DA3METRIC-LARGE"

results: dict = {}


def report(key: str, ok: bool | None, msg: str, **data) -> None:
    tag = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
    print(f"[{tag}] {key}: {msg}")
    results[key] = {"status": tag, "msg": msg, **data}


def network_size(w: int, h: int, long_side: int = 504, multiple: int = 14) -> tuple[int, int]:
    """Spec A2: long side -> 504, short side -> 14*round(short*504/long/14)."""
    if w >= h:
        return long_side, multiple * round(h * long_side / w / multiple)
    return multiple * round(w * long_side / h / multiple), long_side


def stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {
        "min": float(a.min()),
        "p1": float(np.percentile(a, 1)),
        "median": float(np.median(a)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def check_da3(rgb_net: np.ndarray, device: torch.device) -> None:
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.io.input_processor import InputProcessor

    t0 = time.time()
    model = DepthAnything3.from_pretrained(DA3_MODEL_ID).to(device).eval()
    report("da3.load", True, f"from_pretrained({DA3_MODEL_ID!r}) in {time.time() - t0:.1f}s",
           model_name=model.model_name)

    H, W = rgb_net.shape[:2]
    kw = dict(process_res=504, process_res_method="upper_bound_resize")

    # 1) InputProcessor must not resize: compare against our own ImageNet normalization.
    imgs_cpu, _, _ = model._preprocess_inputs([rgb_net], None, None, **kw)
    ref = InputProcessor.NORMALIZE(torch.from_numpy(rgb_net).permute(2, 0, 1).float() / 255.0)
    shape_ok = tuple(imgs_cpu.shape) == (1, 3, H, W)
    diff = float((imgs_cpu[0] - ref).abs().max()) if shape_ok else float("nan")
    report("da3.upper_bound_resize_noop", shape_ok and diff < 1e-6,
           f"processed {tuple(imgs_cpu.shape)} for input {H}x{W}; max|diff| vs own normalize = {diff:.2e}")
    # Same check on the dataset's network shapes (1023x768 -> 504x378, 768x1023 -> 378x504).
    rng = np.random.default_rng(0)
    for w, h in [network_size(1023, 768), network_size(768, 1023)]:
        x = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        t, _, _ = model._preprocess_inputs([x], None, None, **kw)
        r = InputProcessor.NORMALIZE(torch.from_numpy(x).permute(2, 0, 1).float() / 255.0)
        ok = tuple(t.shape) == (1, 3, h, w) and float((t[0] - r).abs().max()) < 1e-6
        report(f"da3.upper_bound_resize_noop_{w}x{h}", ok, f"processed {tuple(t.shape)} for input {w}x{h}")

    # 2) Public inference() path.
    pred = model.inference([rgb_net], **kw)
    report("da3.prediction_fields", None,
           ", ".join(f"{k}={None if v is None else (getattr(v, 'shape', v), getattr(v, 'dtype', type(v).__name__))}"
                     for k, v in vars(pred).items() if k not in ("aux", "gaussians")))
    report("da3.output_resolution", pred.depth.shape == (1, H, W),
           f"depth {pred.depth.shape} {pred.depth.dtype}", depth_stats=stats(pred.depth[0]))
    report("da3.conf", None, f"conf is {'None' if pred.conf is None else pred.conf.shape}")
    report("da3.sky_public", None,
           f"Prediction.sky dtype={None if pred.sky is None else pred.sky.dtype}, "
           f"frac={None if pred.sky is None else float(pred.sky.mean()):.4f}")

    # 3) Raw forward: the sky head output before OutputProcessor thresholds it.
    imgs, _, _ = model._prepare_model_inputs(imgs_cpu, None, None)
    raw = model._run_model_forward(imgs, None, None, [], False, False, "saddle_balanced")
    sky_raw = raw["sky"].float().cpu().numpy()
    depth_raw = raw["depth"].float().cpu().numpy()
    report("da3.raw_keys", None, f"raw output keys: {sorted(raw.keys())}; sky {sky_raw.shape}, depth {depth_raw.shape}")
    s = sky_raw[0, 0]
    report("da3.sky_raw_range", None,
           f"raw sky min={s.min():.4f} max={s.max():.4f}; frac>1: {(s > 1).mean():.4f}; "
           f"frac in [0.3,0.5): {((s >= 0.3) & (s < 0.5)).mean():.4f}", sky_stats=stats(s))
    same = np.array_equal(pred.sky[0], s >= 0.5) if pred.sky is not None else None
    report("da3.sky_public_is_raw_ge_0p5", same, "Prediction.sky == (raw sky >= 0.5)")
    sky03 = s >= 0.3
    if sky03.any():
        fill = depth_raw[0, 0][sky03]
        report("da3.sky_depth_fill", None,
               f"depth on raw sky>=0.3: min={fill.min():.4f} max={fill.max():.4f} (constant fill expected)")

    # 4) Batch independence: two different images (original + horizontal flip) in one call.
    flip = np.ascontiguousarray(rgb_net[:, ::-1])
    pb = model.inference([rgb_net, flip], **kw)
    nonsky = s < 0.3
    d_single, d_batch = pred.depth[0], pb.depth[0]
    rel = np.abs(d_batch - d_single) / d_single
    report("da3.batch_independence", None,
           f"N=2 call vs N=1, non-sky rel diff: median {np.median(rel[nonsky]):.2e}, "
           f"max {rel[nonsky].max():.2e}; view0 vs view1 of [img,img] is exact if only bf16 "
           f"kernel numerics differ (see docs/api_notes.md)")
    pp = model.inference([rgb_net, rgb_net], **kw)
    report("da3.identical_views_identical", bool(np.array_equal(pp.depth[0], pp.depth[1])),
           "N=2 [img,img]: view0 depth == view1 depth (no cross-view coupling)")

    del model
    torch.cuda.empty_cache()


def check_geocalib(rgb_full: np.ndarray, device: torch.device) -> None:
    from geocalib import GeoCalib

    t0 = time.time()
    model = GeoCalib(weights="pinhole").to(device)
    report("geocalib.load", True, f"GeoCalib(weights='pinhole') in {time.time() - t0:.1f}s")

    H, W = rgb_full.shape[:2]
    img = torch.from_numpy(rgb_full).permute(2, 0, 1).float().div(255.0).to(device)  # [3,H,W] in [0,1]
    out = model.calibrate(img, camera_model="pinhole")
    cam, grav = out["camera"], out["gravity"]
    size = cam.size[0].tolist()
    fx, fy = cam.f[0].tolist()
    report("geocalib.camera_size_is_input", bool(np.allclose(size, [W, H], atol=1e-2)),
           f"camera.size={size} for input W,H=({W},{H})")
    roll, pitch = [float(np.degrees(v)) for v in grav.rp[0].tolist()]
    unc = {k: float(v.reshape(-1)[0]) for k, v in out.items() if "uncertainty" in k}
    report("geocalib.single", None,
           f"fx={fx:.2f} fy={fy:.2f} cx,cy={cam.c[0].tolist()} vfov={np.degrees(float(cam.vfov[0])):.2f}deg "
           f"roll={roll:.2f}deg pitch={pitch:.2f}deg unc={unc}",
           fx=fx, fy=fy, uncertainties=unc, keys=sorted(out.keys()))

    # Empirical check of input-pixel coordinates: half-size input should give ~half focal.
    half = cv2.resize(rgb_full, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    out_h = model.calibrate(torch.from_numpy(half).permute(2, 0, 1).float().div(255.0).to(device))
    fh = float(out_h["camera"].f[0, 1])
    ratio = fh / fy
    report("geocalib.focal_scales_with_input", abs(ratio - 0.5) < 0.05,
           f"f(half-size)={fh:.2f} / f(full)={fy:.2f} = {ratio:.3f} (expect ~0.5)")

    # shared_intrinsics=True on a batch of same-size images.
    flip = torch.flip(img, dims=[-1])
    out_s = model.calibrate(torch.stack([img, flip]), camera_model="pinhole", shared_intrinsics=True)
    fs = out_s["camera"].f.tolist()
    unc_s = {k: list(np.atleast_1d(v.cpu().numpy()).round(4)) for k, v in out_s.items() if "uncertainty" in k}
    report("geocalib.shared_intrinsics", None,
           f"f per image={fs}; covariance {tuple(out_s['covariance'].shape)}; uncertainties={unc_s}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        report("env.gpu", True, f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
                                f"CUDA {torch.version.cuda}, bf16={torch.cuda.is_bf16_supported()}")
    else:
        device = torch.device("cpu")
        warnings.warn("CUDA not available; running on CPU")
        report("env.gpu", False, "CUDA not available, using CPU")

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return 2
    rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb_full.shape[:2]
    Wn, Hn = network_size(W, H)
    rgb_net = cv2.resize(rgb_full, (Wn, Hn), interpolation=cv2.INTER_AREA)
    report("input", None, f"{args.image.name}: {W}x{H} -> network {Wn}x{Hn}")

    torch.manual_seed(0)
    check_da3(rgb_net, device)
    check_geocalib(rgb_full, device)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "smoke_m0.json").write_text(json.dumps(results, indent=2, default=str))
    n_fail = sum(r["status"] == "FAIL" for r in results.values())
    print(f"\n{n_fail} FAIL; summary -> {args.out / 'smoke_m0.json'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
