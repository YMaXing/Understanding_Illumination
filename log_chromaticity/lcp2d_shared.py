"""Linear RGB TIFF -> mean-ISD shared plane -> C1/C2 float coordinates.

PyCharm: data/ and model/ stay in the project root above clustering/.
No command-line arguments required. Relative paths resolve beside this script.
Requires numpy, tifffile, imagecodecs, torch, opencv-python and the original
model modules plus vit_12_linear.pth (weights not bundled).

Input: demosaiced THREE-channel LINEAR RGB TIFF, uint16 or float [0,1].
This does not decode a Bayer mosaic or apply inverse gamma. For 12/14-bit
values stored in uint16, set WHITE_LEVEL to the actual sensor white level.

Normalize each predicted ISD, take its arithmetic mean over valid pixels,
and normalize that mean. All pixels in one image share this direction n and
one orthonormal 3x2 basis B. This approximates the old per-pixel projection.
With A=(10.4,10.4,10.4), p=log(65535*RGB):
    q = A + (p-A) - dot(p-A,n)*n
    [C1,C2] = (q-A) @ B
    q = A + B @ [C1,C2]
A shared frame is defined within each image, not across different images.
Outputs: *_lcp.npz, *_C1.tiff, *_C2.tiff. Coordinates are signed float32;
no exponentiation, channel-sum normalization, or display quantization.
Invalid pixels are NaN; use valid_mask for clustering.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ANCHOR = 10.4
BASIS_CONVENTION = "single_frame_from_shared_ISD_v1"

# Local Run configuration. Paths are relative to THIS script, not PyCharm's cwd.
SCRIPT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = "data"
OUTPUT_DIR = "results"
MODEL_ROOT = "model"   # Model .py files may also sit beside this script.
MODEL_PATH = None      # None: find vit_12_linear.pth automatically.
DEVICE = "auto"        # "auto", "cpu", or "cuda"
BLACK_LEVEL = 0.0
WHITE_LEVEL = None     # uint16 default: 65535; float default: 1.0
OVERWRITE = False      # Set True to replace previous outputs intentionally.


def local_path(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path



def read_linear_tiff(path, black_level=0.0, white_level=None):
    """Read original TIFF precision. Return RGB float32, masks, and metadata."""
    import tifffile

    if not np.isfinite(black_level):
        raise ValueError("black-level must be finite")
    with tifffile.TiffFile(path) as tif:
        if len(tif.series) != 1:
            raise ValueError("Expected one image series, not a multi-image TIFF")
        series = tif.series[0]
        raw = series.asarray()
        axes = series.axes
        photo = tif.pages[0].photometric
        orientation_tag = tif.pages[0].tags.get("Orientation")
        orientation = int(orientation_tag.value) if orientation_tag else 1
    if orientation != 1:
        raise ValueError("TIFF Orientation must be 1; explicitly orient image and ISD together")
    if raw.ndim != 3:
        raise ValueError("Expected demosaiced HxWx3 linear RGB; Bayer/grayscale/stack TIFF is unsupported")
    if axes in ("SYX", "CYX") and raw.shape[0] == 3:
        raw = np.moveaxis(raw, 0, -1)
    elif axes not in ("YXS", "YXC") or raw.shape[-1] != 3:
        raise ValueError(f"Expected three RGB samples per pixel; got axes={axes}, shape={raw.shape}")
    # PhotometricInterpretation RGB=2. No silent YCbCr/CMYK/CFA conversion.
    if photo != 2:
        raise ValueError(f"Expected RGB TIFF photometric interpretation, got {photo}")
    if raw.dtype == np.uint16:
        default_white = 65535.0
    elif np.issubdtype(raw.dtype, np.floating):
        default_white = 1.0
    else:
        raise ValueError(f"Expected uint16 or floating RGB TIFF, got {raw.dtype}")
    white = default_white if white_level is None else float(white_level)
    if not np.isfinite(white) or white <= black_level:
        raise ValueError("white-level must be finite and greater than black-level")
    values = raw.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Input TIFF contains NaN/Inf")
    if np.issubdtype(raw.dtype, np.floating) and white_level is None:
        if values.min() < 0 or values.max() > 1:
            raise ValueError("Float TIFF outside [0,1]: specify its actual --white-level")
    valid = np.all((values > black_level) & (values < white), axis=-1)
    saturated = np.any(values >= white, axis=-1)
    rgb = np.clip((values - black_level) / (white - black_level), 0, 1).astype(np.float32)
    metadata = dict(source=str(Path(path).resolve()), original_dtype=str(raw.dtype),
                    original_axes=axes, black_level=float(black_level), white_level=white,
                    input_assumption="demosaiced linear RGB in estimator-compatible RGB space",
                    orientation=orientation, shape=list(rgb.shape))
    return rgb, valid, saturated, metadata


def normalize_isd(isd):
    n = np.asarray(isd, dtype=np.float64)
    if n.ndim != 3 or n.shape[-1] != 3:
        raise ValueError("ISD must have shape HxWx3 in RGB-component order")
    finite = np.isfinite(n).all(axis=-1)
    safe = np.where(finite[..., None], n, 0.0)
    norms = np.linalg.norm(safe, axis=-1)
    valid = finite & (norms > 1e-8)
    neutral = np.ones(3) / np.sqrt(3)
    unit = np.where(valid[..., None], safe / np.maximum(norms[..., None], 1e-8), neutral)
    return unit, valid


def plane_basis(unit_isd):
    """Shortest rotation of a neutral reference frame into each local plane.

    Plane normals are unoriented: n and -n generate the same basis. The original
    ISD sign is retained in the output isd_map; only the basis normal is flipped.
    """
    u = np.ones(3) / np.sqrt(3)
    a = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    dot = np.einsum("...c,c->...", unit_isd, u)
    dominant = np.take_along_axis(unit_isd, np.argmax(np.abs(unit_isd), axis=-1)[..., None], axis=-1)[..., 0]
    sign = np.where(dot < 0, -1.0, 1.0)
    sign = np.where(np.abs(dot) < 1e-12, np.where(dominant < 0, -1.0, 1.0), sign)
    n = unit_isd * sign[..., None]
    cosine = np.einsum("...c,c->...", n, u)
    v = np.cross(u, n)
    va = np.cross(v, a)
    e1 = a + va + np.cross(v, va) / (1 + cosine[..., None])
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2, axis=-1, keepdims=True)
    return np.stack((e1, e2), axis=-1)


def fit_shared_isd(unit_isd, valid_mask):
    directions = unit_isd[valid_mask]
    if len(directions) == 0:
        raise ValueError("No valid pixels for mean ISD; check TIFF black/white levels")
    mean = directions.mean(axis=0)
    length = float(np.linalg.norm(mean))
    if not np.isfinite(length) or length < 1e-6:
        raise ValueError("Mean ISD is near zero; check ISD signs/data or supply --shared-isd")
    return mean / length, length


def project_lcp2d(linear_rgb, isd_map, anchor=ANCHOR, valid_image=None, shared_isd=None):
    """Project onto ONE plane per image using the normalized mean unit ISD."""
    rgb = np.asarray(linear_rgb, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB must have shape HxWx3")
    if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 1):
        raise ValueError("RGB must be finite linear data in [0,1]")
    if not np.isfinite(anchor):
        raise ValueError("anchor must be finite")
    unit, valid_isd = normalize_isd(isd_map)
    if unit.shape != rgb.shape:
        raise ValueError("ISD and image must have the same HxWx3 shape")
    positive = np.all(rgb > 0, axis=-1)
    valid = valid_isd & positive
    if valid_image is not None:
        image_mask = np.asarray(valid_image, dtype=bool)
        if image_mask.shape != valid.shape:
            raise ValueError("valid_image must have shape HxW")
        valid &= image_mask
    if shared_isd is None:
        n, mean_length = fit_shared_isd(unit, valid)
    else:
        n = np.asarray(shared_isd, dtype=np.float64)
        if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-8:
            raise ValueError("shared ISD must be a finite nonzero three-vector")
        n = n / np.linalg.norm(n)
        mean_length = float("nan")
    basis = plane_basis(n[None, None, :])[0, 0]
    p = np.zeros_like(rgb)
    np.log(rgb * 65535.0, out=p, where=rgb > 0)
    shifted = p - anchor
    q = shifted - np.sum(shifted * n, axis=-1, keepdims=True) * n + anchor
    coords = np.einsum("...c,ck->...k", q - anchor, basis)
    angles = np.degrees(np.arccos(np.clip(np.abs(unit @ n), 0, 1)))
    coords[~valid] = np.nan
    q[~valid] = np.nan
    angles[~valid] = np.nan
    unit[~valid_isd] = np.nan
    return dict(lcp=coords.astype(np.float32), projected_log_rgb=q.astype(np.float32),
                basis=basis.astype(np.float32), isd_map=unit.astype(np.float32),
                shared_isd=n.astype(np.float32), isd_angle_degrees=angles.astype(np.float32),
                mean_isd_resultant_length=np.asarray(mean_length, dtype=np.float32),
                valid_isd=valid_isd, valid_positive_rgb=positive, valid_mask=valid,
                anchor=np.full(3, anchor, dtype=np.float32))


def load_estimator(model_root, model_path, device):
    root = Path(model_root).resolve()
    sys.path.insert(0, str(SCRIPT_DIR))
    sys.path.insert(0, str(root))
    for name in ("map_ViT_models.py", "model_utils.py"):
        if not any((folder / name).is_file() for folder in (root, SCRIPT_DIR)):
            raise FileNotFoundError(
                f"Missing {name}: place it in {root} or {SCRIPT_DIR}, without download suffixes")
    if root == SCRIPT_DIR / "model" and all(
        (root / name).is_file() for name in ("map_ViT_models.py", "model_utils.py")
    ):
        from model.map_ViT_models import ViT_Patch2Patch_ver3
        from model.model_utils import ISDMapEstimator
    else:
        # Custom folders and legacy files beside this script use the paths above.
        from importlib import import_module

        ViT_Patch2Patch_ver3 = import_module("map_ViT_models").ViT_Patch2Patch_ver3
        ISDMapEstimator = import_module("model_utils").ISDMapEstimator

    if model_path:
        weights = Path(model_path)
    else:
        candidates = (root / "model/vit_12_linear.pth", root / "vit_12_linear.pth",
                      root.parent / "model/vit_12_linear.pth", SCRIPT_DIR / "vit_12_linear.pth")
        weights = next((p for p in candidates if p.is_file()), None)
    if weights is None or not weights.is_file():
        raise FileNotFoundError("vit_12_linear.pth not found; supply --model-path")
    model = ViT_Patch2Patch_ver3(img_size=512, patch_size=16, in_ch=3, out_ch=3,
                               embed_dim=768, depth=12, heads=8, dropout=0)
    return ISDMapEstimator(model=model, model_path=str(weights), device=device)


def estimate_isd(estimator, rgb):
    import cv2

    height, width = rgb.shape[:2]
    small = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    # The supplied ISDMapEstimator converts BGR back to RGB internally.
    predicted, _ = estimator.predict(np.ascontiguousarray(small[..., ::-1]))
    if predicted.shape != (512, 512, 3):
        raise ValueError(f"Unexpected estimator output: {predicted.shape}")
    if (height, width) != (512, 512):
        predicted = cv2.resize(predicted, (width, height), interpolation=cv2.INTER_LINEAR)
    return predicted  # Unit-normalized by project_lcp2d after interpolation.


def read_isd(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as saved:
            return saved["isd_map"].copy()
    raise ValueError("Precomputed ISD must be .npy or .npz (key isd_map), RGB-component order")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--single", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--isd-map", type=Path, help="Precomputed map for --single; bypasses ViT")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    parser.add_argument("--black-level", type=float, default=BLACK_LEVEL)
    parser.add_argument("--white-level", type=float, default=WHITE_LEVEL)
    parser.add_argument("--anchor", type=float, default=ANCHOR)
    parser.add_argument("--shared-isd", type=float, nargs=3, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    args = parser.parse_args()
    if args.isd_map and not args.single:
        parser.error("--isd-map is only supported with --single")
    if args.isd_map and (args.model_root or args.model_path):
        parser.error("Use either --isd-map or ViT model arguments")
    if not args.single and args.input_dir is None:
        args.input_dir = INPUT_DIR
    if not args.isd_map:
        args.model_root = args.model_root or MODEL_ROOT
        args.model_path = args.model_path or MODEL_PATH
    for name in ("single", "input_dir", "output_dir", "model_root", "model_path", "isd_map"):
        setattr(args, name, local_path(getattr(args, name)))
    if args.single and not args.single.is_file():
        parser.error(f"Input file does not exist: {args.single}")
    if not args.single and not args.input_dir.is_dir():
        parser.error(f"Create {args.input_dir} and put your .tif/.tiff images there")
    paths = [args.single] if args.single else sorted(p for p in args.input_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
    if not paths:
        parser.error("No TIFF images found")
    if any(p.suffix.lower() not in (".tif", ".tiff") for p in paths):
        parser.error("Input must be .tif or .tiff")
    if len({p.stem.casefold() for p in paths}) != len(paths):
        parser.error("Input names have duplicate stems; use unique names")
    suffixes = ("_lcp.npz", "_C1.tiff", "_C2.tiff")
    targets = [args.output_dir / (p.stem + s) for p in paths for s in suffixes]
    if not args.overwrite and any(p.exists() for p in targets):
        parser.error("An output already exists; choose a new folder or use --overwrite")
    if not args.isd_map:
        import torch
        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA unavailable in this interpreter; set DEVICE='cpu' or 'auto'")
    print(f"Input: {args.single or args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"ISD: {'precomputed' if args.isd_map else args.device}; projection: mean ISD, shared plane")
    estimator = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import tifffile

    for path in paths:
        rgb, valid_image, saturated, metadata = read_linear_tiff(path, args.black_level, args.white_level)
        if args.isd_map:
            isd = read_isd(args.isd_map)
            metadata["isd_source"] = str(args.isd_map.resolve())
        else:
            if estimator is None:
                estimator = load_estimator(args.model_root, args.model_path, args.device)
            isd = estimate_isd(estimator, rgb)
            metadata["isd_source"] = "ViT_Patch2Patch_ver3, resize to 512 then resize back"
            metadata["model_root"] = str(args.model_root.resolve())
            metadata["model_path_override"] = str(args.model_path) if args.model_path else None
        result = project_lcp2d(rgb, isd, anchor=args.anchor, valid_image=valid_image, shared_isd=args.shared_isd)
        result["saturated_mask"] = saturated
        metadata.update(basis_convention=BASIS_CONVENTION, coordinate_space="one shared ISD plane per image",
                        coordinate_order=["C1", "C2"], spatial_axes=["row", "column"],
                        coordinates_are_global_plane=True, shared_scope="within_image",
                        replaces_per_pixel_projection=True,
                        shared_isd_method="explicit" if args.shared_isd is not None else "normalized_mean_of_valid_unit_isd",
                        reconstruction="anchor + einsum('ck,...k->...c', basis, lcp)")
        result["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        np.savez_compressed(args.output_dir / f"{path.stem}_lcp.npz", **result)
        for channel in range(2):
            pixels = result["lcp"][..., channel].copy()
            pixels[~result["valid_mask"]] = np.nan
            tifffile.imwrite(args.output_dir / f"{path.stem}_C{channel+1}.tiff", pixels,
                             photometric="minisblack", metadata={"axes": "YX", "channel": f"C{channel+1}"})
        print(f"{path.name}: lcp={result['lcp'].shape}, float32, valid={result['valid_mask'].mean():.2%}")


if __name__ == "__main__":
    main()
