"""Config loading and path resolution for metrics/ (config.yaml + the untracked config.local.yaml)."""

import copy
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import yaml

METRICS = Path(__file__).resolve().parent.parent
REPO = METRICS.parent
DEFAULT_CONFIG = METRICS / "config.yaml"
LOCAL_CONFIG = METRICS / "config.local.yaml"


def load_config(path=None, overrides=None, local=True):
    """config.yaml, then config.local.yaml (machine-specific; untracked) merged over it, then `overrides`.
    Adds cfg['_path'], cfg['_local'] (the local file used, or None) and cfg['_repo']."""
    path = Path(path) if path else DEFAULT_CONFIG
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    lp = LOCAL_CONFIG if local else None
    if lp is not None and lp.exists():
        cfg = _merge(cfg, yaml.safe_load(lp.read_text(encoding="utf-8")) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    cfg["_path"] = str(path.resolve())
    cfg["_local"] = str(lp) if lp is not None and lp.exists() else None
    cfg["_repo"] = str(REPO)
    return cfg


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def repo_path(p):
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_set(cfg, stem):
    """The provenance set of an image (config image_sets); KeyError if it is not listed exactly once."""
    hits = [s for s, stems in cfg["image_sets"].items() if stem in stems]
    if len(hits) != 1:
        raise KeyError(f"{stem}: listed in {len(hits)} image sets ({hits}); expected exactly 1")
    return hits[0]


def group_of(cfg, set_name):
    """Provenance group (dev, glossy_dev, glossy_confirm, draw54) of an image set: a stratum, never a report block."""
    for g, sets in cfg["aggregate"]["provenance_groups"].items():
        if set_name in sets:
            return g
    raise KeyError(f"set {set_name!r} is in no provenance group")


def annotator_group(cfg, set_name):
    for g, sets in cfg["aggregate"]["annotator_groups"].items():
        if set_name in sets:
            return g
    raise KeyError(f"set {set_name!r} is in no annotator group")


def glossy_pool(cfg):
    p = repo_path(cfg["paths"]["glossy_pool"])
    return {x["stem"] for x in json.loads(p.read_text())["images"]}


def dataset_dir(cfg, set_name):
    key = cfg["sets"][set_name]["images"]
    return repo_path(cfg["dataset"][key])


def image_path(cfg, stem):
    p = dataset_dir(cfg, image_set(cfg, stem)) / f"{stem}.tif"
    if not p.exists():
        raise FileNotFoundError(f"{p} (set dataset.{cfg['sets'][image_set(cfg, stem)]['images']} in metrics/config.local.yaml)")
    return p


def all_stems(cfg, sets=None):
    ev = cfg["image_sets"]
    return [s for k in (sets or ev) for s in ev[k]]


eval_stems = all_stems          # older name


def split_of(cfg, stem):
    """The data split (train / val / test) from the image's annotation file; None if it has none."""
    p = repo_path(cfg["paths"]["annotations"]) / f"{stem}.json"
    return json.loads(p.read_text()).get("split") if p.exists() else None


_hooks = {}
STUBBED = []


def _stub(name):
    """An inert placeholder for an optional import that is absent from this clone: any use raises a clear error."""
    m = types.ModuleType(name)

    def __getattr__(attr):
        raise ImportError(f"{name}.{attr} was used, but {name} is a placeholder: the file log_chromaticity/{name}.py is "
                          f"missing from this clone (git-ignored). The metrics hooks never use it; whatever called "
                          f"{name}.{attr} needs the real log_chromaticity/ folder.")
    m.__getattr__ = __getattr__
    m.__metrics_stub__ = True
    return m


def import_hook(spec, optional_imports=()):
    """'file.py:function' (relative to the repo root) -> the callable, imported from that file (never copied).
    The module is registered under its file stem if that name is free, so its own sibling imports keep working.
    Names in optional_imports that cannot be imported get an inert placeholder (recorded in STUBBED)."""
    if spec in _hooks:
        return _hooks[spec]
    fname, func = spec.rsplit(":", 1)
    path = repo_path(fname).resolve()
    name = path.stem
    mod = sys.modules.get(name)
    if mod is None or Path(getattr(mod, "__file__", "")).resolve() != path:
        sys.dont_write_bytecode = True            # never write __pycache__ into folders outside metrics/
        for opt in optional_imports or _default_optional():
            if opt in sys.modules:
                continue
            found = importlib.util.find_spec(opt) is not None if _on_path(opt) else False
            if not found and not (REPO / "log_chromaticity" / f"{opt}.py").exists():
                sys.modules[opt] = _stub(opt)
                STUBBED.append(opt)
        modname = name if name not in sys.modules else f"_metrics_hook_{name}_{abs(hash(str(path)))}"
        ms = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(ms)
        sys.modules[modname] = mod
        ms.loader.exec_module(mod)
    fn = getattr(mod, func)
    _hooks[spec] = fn
    return fn


def _on_path(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _default_optional():
    try:
        return load_config().get("loader", {}).get("optional_imports", [])
    except Exception:
        return []
