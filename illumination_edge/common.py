"""Shared paths, I/O and helpers for illumination_edge/.

Run everything through the wrapper, from any directory:
    illumination_edge/run.sh <script>.py ...
which is  PYTHONDONTWRITEBYTECODE=1 env -u PYTHONPATH uv run --project <abs>/log_chromaticity python <abs>/<script>.py

`material_edges` (and the rest of ../log_chromaticity) is imported read-only via sys.path; bytecode writing is
disabled before that import so nothing is written into that folder.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
CODEBASE = ROOT.parent
LOGCHROMA = CODEBASE / "log_chromaticity"
IMAGE_DIR = CODEBASE / "image_data" / "depth_and_surface_normal_edges"
ISD_DIR = LOGCHROMA / "data" / "batch1" / "output"          # resize-512 ViT maps (default)
ISD_DIR_FULLRES = ROOT / "outputs" / "isd_fullres"           # --full-res ViT maps
GEOM_DIR = CODEBASE / "geometry" / "dedicated" / "baseline_output" / "stage_b" / "geom_v1_1"
OUT = ROOT / "outputs"

if str(LOGCHROMA) not in sys.path:
    sys.path.insert(0, str(LOGCHROMA))
import material_edges as ME  # noqa: E402  (estimate_noise, boundary_strength, detect, thin_edges, ...)

SAT = 65000


def list_images(img_dir=IMAGE_DIR):
    img_dir = Path(img_dir)
    return sorted({p for pat in ("*.tif", "*.tiff") for p in img_dir.glob(pat)})


def load_rgb(path):
    """uint16 linear RGB (H, W, 3). cv2 returns BGR; reversed here."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint16:
        raise ValueError(f"{path}: need a 3-channel uint16 TIFF")
    return np.ascontiguousarray(img[..., ::-1])


def save_rgb(path, img):
    """uint16 RGB -> LZW TIFF (BGR on disk, like the originals)."""
    ok = cv2.imwrite(str(path), np.ascontiguousarray(img[..., ::-1]), [cv2.IMWRITE_TIFF_COMPRESSION, 5])
    if not ok:
        raise IOError(f"could not write {path}")


def load_isd(stem, isd_dir=ISD_DIR, shape=None):
    """Unit ISD (H, W, 3) float32, R,G,B order, oriented to a positive component sum. Always via cv2."""
    p = Path(isd_dir) / f"{stem}_isd.tiff"
    isd = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if isd is None:
        raise FileNotFoundError(p)
    if shape is not None and isd.shape[:2] != tuple(shape[:2]):
        raise ValueError(f"{p}: ISD {isd.shape[:2]} vs image {shape[:2]}")
    return unit(isd.astype(np.float64)).astype(np.float32)


def unit(v):
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    return v * np.where(v.sum(-1, keepdims=True) < 0, -1.0, 1.0)


def valid_mask(img):
    return np.all((img > 0) & (img < SAT), axis=-1)


def log_rgb(img):
    return np.log(np.maximum(img.astype(np.float64), 1.0))


def load_geom(stem, shape):
    """geom_v1_1 npz: w (H, W) float32 in the stored frame, sky mask upsampled (nearest) to (H, W)."""
    z = np.load(GEOM_DIR / f"{stem}.npz")
    w = z["w"].astype(np.float32)
    if w.shape != tuple(shape[:2]):
        raise ValueError(f"{stem}: geom w {w.shape} vs image {shape[:2]}")
    sky = cv2.resize(z["sky_mask"].astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return w, sky


def noise_floor_bins(img, sat=SAT, nbins=30, min_px=2000):
    """Per-channel (intensity, noise-floor) points, computed exactly as in material_edges.estimate_noise:
    local variance of the 3x3 high-pass residual (7x7 window, /0.889) and its 5th percentile / 0.75 per
    log-spaced intensity bin."""
    from scipy import ndimage as ndi
    pts = []
    for c in range(3):
        x = img[..., c].astype(np.float64)
        resid = x - ndi.uniform_filter(x, 3)
        lv = ndi.uniform_filter(resid * resid, 7) / 0.889
        mu = ndi.uniform_filter(x, 7)
        ok = (mu > 5) & (mu < sat)
        edges = np.geomspace(max(mu[ok].min(), 5), mu[ok].max(), nbins)
        idx = np.digitize(mu, edges)
        xs, ys = [], []
        for b in range(1, len(edges)):
            m = ok & (idx == b)
            if m.sum() > min_px:
                xs.append(np.median(mu[m]))
                ys.append(np.percentile(lv[m], 5) / 0.75)
        pts.append((np.array(xs), np.array(ys)))
    return pts


def estimate_noise_robust(img, trim=np.log(1.5), iters=4):
    """var(I_c) = g*I + r2 fitted to the same floor points as material_edges.estimate_noise, but in log space with
    one-sided trimming: texture only ever inflates a bin's floor, so bins more than `trim` (log) above the fit are
    dropped and the fit repeated. ME.estimate_noise's linear NNLS is dominated by texture-contaminated bright bins
    and overestimates low-intensity noise 10-50x on these images (DECISIONS.md, Phase 1)."""
    from scipy.optimize import least_squares
    params = []
    for xs, ys in noise_floor_bins(img):
        keep = np.ones(len(xs), bool)
        p = np.array([max(np.median(ys / xs), 1e-3), max(ys.min() * 0.5, 1.0)])
        for _ in range(iters):
            f = lambda q: np.log(q[0] * xs[keep] + q[1]) - np.log(ys[keep])
            p = least_squares(f, p, bounds=([0, 1.0], [np.inf, np.inf]), loss="soft_l1", f_scale=0.1).x
            res = np.log(ys) - np.log(p[0] * xs + p[1])
            new = res < trim
            if new.sum() < 3 or (new == keep).all():
                break
            keep = new
        params.append((float(p[0]), float(p[1])))
    return params


def noise_fit_residual(img, params, imax=4000.0):
    """Per channel: median log(measured floor / model) over floor bins with I < imax. > 0 = model under-predicts."""
    out = []
    for (g, r2), (xs, ys) in zip(params, noise_floor_bins(img)):
        sel = xs < imax
        out.append(float(np.median(np.log(ys[sel]) - np.log(g * xs[sel] + r2))) if sel.any() else float("nan"))
    return out


def tone(img, pct=99.5, gain=1.0):
    """Tonemap linear RGB -> uint8 RGB: scale so the pct-th percentile maps to 1, then gamma 2.2."""
    x = img.astype(np.float32)
    ref = max(np.percentile(x, pct), 1.0)
    return (np.clip(gain * x / ref, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def isoline(f, level=0.5):
    """1-px-thin (4-connected) isoline of f at level: of each sign-changing 4-neighbour pair, the pixel nearer level."""
    g = f.astype(np.float32) - level
    out = np.zeros(f.shape, bool)
    for ax in (0, 1):
        a = g[:-1, :] if ax == 0 else g[:, :-1]
        b = g[1:, :] if ax == 0 else g[:, 1:]
        cross = (a * b < 0) | ((a == 0) ^ (b == 0))
        pick_a = cross & (np.abs(a) <= np.abs(b))
        pick_b = cross & ~pick_a
        if ax == 0:
            out[:-1, :] |= pick_a
            out[1:, :] |= pick_b
        else:
            out[:, :-1] |= pick_a
            out[:, 1:] |= pick_b
    return out


def dist_to(mask):
    """Euclidean distance from every pixel to the nearest True pixel of mask (inf if mask is empty)."""
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
