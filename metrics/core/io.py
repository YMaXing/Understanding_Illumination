"""Image and prediction I/O (DESIGN.md §2; Deltas B1, C1, C2).

    img = load_image(cfg, stem)        # dict: I (H, W, 3) uint16, lI float64, valid bool, H, W, path, sat_band count
    lR, lL = load_prediction(cfg, pred_dir, stem, img)   # float64; lL = lI - lR (checked against a log_L key if present)

ℓI comes only from the configured loader hook (config loader.read / loader.log_i / loader.mask); nothing here
re-implements preprocessing. Validity: invalid if any channel <= eps_dn or >= sat_dn, or the loader masks it.
"""

from pathlib import Path

import numpy as np

from . import config as K


class PredictionError(ValueError):
    pass


def pixel_sha256(I):
    """Identity of the decoded image: SHA-256 of '<shape>|<dtype>|' + the C-contiguous bytes of loader.read's array
    (uint16 RGB). Unchanged by metadata edits (e.g. a Windows Explorer rating), unlike the file hash."""
    import hashlib
    a = np.ascontiguousarray(I)
    return hashlib.sha256(f"{a.shape}|{a.dtype}|".encode() + a.tobytes()).hexdigest()


def read_raw(cfg, path):
    return K.import_hook(cfg["loader"]["read"], cfg["loader"].get("optional_imports"))(path)


def validity_mask(cfg, I, loader_mask=None):
    v = cfg["validity"]
    ok = np.all(I > v["eps_dn"], axis=-1) & np.all(I < v["sat_dn"], axis=-1)
    if loader_mask is not None:
        ok &= loader_mask.astype(bool)
    return ok


def load_image(cfg, stem, path=None):
    L = cfg["loader"]
    path = Path(path) if path else K.image_path(cfg, stem)
    I = K.import_hook(L["read"])(path)
    if I.ndim != 3 or I.shape[2] != 3:
        raise ValueError(f"{path}: loader returned shape {I.shape}, need (H, W, 3)")
    lI = np.asarray(K.import_hook(L["log_i"])(I), dtype=np.float64)
    lm = K.import_hook(L["mask"])(I) if L.get("mask") else None
    valid = validity_mask(cfg, I, lm)
    v = cfg["validity"]
    mx = I.max(axis=-1)
    sat_band = int(((mx >= v["sat_report_lo_dn"]) & (mx < v["sat_dn"])).sum())
    return dict(stem=stem, path=str(path), I=I, lI=lI, valid=valid, H=I.shape[0], W=I.shape[1], sat_band=sat_band)


def noise_params(cfg, img):
    """Sensor-noise model [(g, r2)] per channel from the configured hook (Delta Q1); cached on img."""
    if "noise" not in img:
        img["noise"] = [tuple(map(float, p)) for p in K.import_hook(cfg["noise"]["model"])(img["I"])]
    return img["noise"]


def geom_edge(cfg, stem, shape, set_name=None):
    """geom_v1_1 edge mask (w < edge_thr), full res; None if the set has no geom directory configured."""
    G = cfg["maps"]["geom_v1_1"]
    set_name = set_name or K.image_set(cfg, stem)
    d = G["dirs"].get(set_name)
    if not d:
        return None
    with np.load(K.repo_path(d) / f"{stem}.npz") as z:
        w = z[G["key"]].astype(np.float32)
    if w.shape != tuple(shape[:2]):
        raise ValueError(f"{stem}: geom {G['key']} {w.shape} vs image {shape[:2]}")
    return w < G["edge_thr"]


def context(cfg, stem, lR, img=None, set_name=None):
    """Everything pair_metrics.measure needs for one image and one ℓR."""
    img = img or load_image(cfg, stem)
    set_name = set_name or K.image_set(cfg, stem)
    if "geom_edge" not in img:
        img["geom_edge"] = geom_edge(cfg, stem, img["lI"].shape, set_name)
    return dict(lI=img["lI"], lR=lR, valid=img["valid"], I=img["I"], noise=noise_params(cfg, img),
                geom_edge=img["geom_edge"], regions=None, set=set_name)


def derive_log_L(lI, lR):
    """ℓL = ℓI − ℓR in float64 (§2: one degree of freedom per pixel)."""
    return np.asarray(lI, np.float64) - np.asarray(lR, np.float64)


def load_prediction(cfg, pred_dir, stem, img):
    """<pred_dir>/<stem>.npz: log_R float32 (H, W, 3), natural log, full res, stored frame; optional log_L."""
    P = cfg["predictions"]
    p = Path(pred_dir) / f"{stem}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as z:
        if P["key_log_R"] not in z.files:
            raise PredictionError(f"{p}: no key {P['key_log_R']!r} (keys {z.files})")
        lR = z[P["key_log_R"]]
        lL_pred = z[P["key_log_L"]] if P["key_log_L"] in z.files else None
    want = (img["H"], img["W"], 3)
    if lR.shape != want:
        raise PredictionError(f"{p}: log_R shape {lR.shape}, need {want} (full res, stored frame)")
    if lR.dtype != np.dtype(P["dtype"]):
        raise PredictionError(f"{p}: log_R dtype {lR.dtype}, need {P['dtype']}")
    bad = ~np.isfinite(lR)
    if bad.any():
        raise PredictionError(f"{p}: {int(bad.any(-1).sum())} pixels with non-finite log_R")
    lR = lR.astype(np.float64)
    lL = derive_log_L(img["lI"], lR)
    if lL_pred is not None:
        if lL_pred.shape != want:
            raise PredictionError(f"{p}: log_L shape {lL_pred.shape}, need {want}")
        dev = float(np.max(np.abs(lL_pred.astype(np.float64) - lL)[img["valid"]], initial=0.0))
        if not dev <= P["log_L_tol"]:
            raise PredictionError(f"{p}: identity log_L = ℓI − log_R fails on valid pixels: max |dev| {dev:.3g} "
                                  f"> {P['log_L_tol']:g} (is the network's ℓI the metrics loader's?)")
    return lR, lL
