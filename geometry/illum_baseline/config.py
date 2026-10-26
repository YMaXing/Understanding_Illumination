"""Config dataclasses, YAML load/validate, and a resolved snapshot."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CFA_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")


@dataclass
class PathsCfg:
    data_dir: str | None = None
    out_dir: str | None = None
    glob: str = "**/*.tif*"


@dataclass
class PreprocessCfg:
    layout: str = "auto"
    cfa_pattern: str | None = None
    black_level: float | list[float] | None = None
    white_level: float | None = None
    white_balance: str = "grayworld"
    exposure_percentile: float = 99.0
    exposure_target: float = 0.9
    exposure_gain_max: float = 64.0
    respect_orientation_tag: bool = True


@dataclass
class NetworkCfg:
    long_side: int = 504
    multiple: int = 14
    resize_interp: str = "area"


@dataclass
class GeoCalibCfg:
    weights: str = "pinhole"
    camera_model: str = "pinhole"


@dataclass
class FocalCfg:
    mode: str = "per_image"
    group_key: list[str] | None = None


@dataclass
class DA3Cfg:
    model_id: str = "depth-anything/DA3METRIC-LARGE"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    batch_size: int = 1
    canonical_focal: float = 300.0


@dataclass
class SkyCfg:
    threshold: float = 0.5
    open_close_radius: int = 1
    min_area_px: int = 50


@dataclass
class DetectorCfg:
    sigma: float = 1.25
    sigma_w_prior: float = 1.0
    truncate: float = 4.0
    rho_floor_frac: float = 0.5
    rho_floor_percentile: float = 1.0
    k_tau: float = 3.0
    k_eps: float = 3.0
    w_min: float = 0.05
    s_hat_min: float | None = 1.0e-6
    s_hat_max: float | None = None


@dataclass
class RunCfg:
    name: str = "baseline_v0"
    limit: int | None = None
    force: bool = False


@dataclass
class Config:
    paths: PathsCfg = field(default_factory=PathsCfg)
    device: str = "cuda"
    seed: int = 0
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    network: NetworkCfg = field(default_factory=NetworkCfg)
    geocalib: GeoCalibCfg = field(default_factory=GeoCalibCfg)
    focal: FocalCfg = field(default_factory=FocalCfg)
    da3: DA3Cfg = field(default_factory=DA3Cfg)
    sky: SkyCfg = field(default_factory=SkyCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    run: RunCfg = field(default_factory=RunCfg)

    @property
    def data_dir(self) -> Path:
        if self.paths.data_dir is None:
            raise ValueError("paths.data_dir is null; set it in the config")
        return Path(self.paths.data_dir)

    @property
    def out_dir(self) -> Path:
        if self.paths.out_dir is None:
            raise ValueError("paths.out_dir is null; set it in the config")
        return Path(self.paths.out_dir)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def snapshot(self, path: Path) -> None:
        """Write the resolved config as YAML."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _build(cls, data: dict[str, Any] | None, where: str):
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for name, f in fields.items():
        if name not in data:
            continue
        sub = f.default_factory if f.default_factory is not dataclasses.MISSING else None
        if sub is not None and dataclasses.is_dataclass(sub):
            kwargs[name] = _build(sub, data[name], f"{where}.{name}" if where else name)
        else:
            kwargs[name] = data[name]
    return cls(**kwargs)


def validate(cfg: Config) -> None:
    p = cfg.preprocess
    if p.layout not in ("rgb", "mosaic", "auto"):
        raise ValueError(f"preprocess.layout must be rgb|mosaic|auto, got {p.layout!r}")
    if p.cfa_pattern is not None and p.cfa_pattern not in CFA_PATTERNS:
        raise ValueError(f"preprocess.cfa_pattern must be one of {CFA_PATTERNS} or null")
    if p.layout == "mosaic" and p.cfa_pattern is None:
        raise ValueError("preprocess.layout=mosaic requires cfa_pattern")
    if p.white_balance not in ("metadata", "grayworld", "none"):
        raise ValueError(f"preprocess.white_balance must be metadata|grayworld|none, got {p.white_balance!r}")
    if not 0 < p.exposure_percentile <= 100:
        raise ValueError("preprocess.exposure_percentile must be in (0, 100]")
    if cfg.focal.mode not in ("per_image", "pooled"):
        raise ValueError(f"focal.mode must be per_image|pooled, got {cfg.focal.mode!r}")
    if cfg.network.long_side % cfg.network.multiple:
        raise ValueError("network.long_side must be a multiple of network.multiple")
    if cfg.da3.batch_size < 1:
        raise ValueError("da3.batch_size must be >= 1")
    d = cfg.detector
    if not 0 < d.w_min < 1:
        raise ValueError("detector.w_min must be in (0, 1)")
    if d.s_hat_min is not None and d.s_hat_max is not None and d.s_hat_min > d.s_hat_max:
        raise ValueError("detector.s_hat_min > s_hat_max")
    if cfg.paths.data_dir is not None and cfg.paths.out_dir is not None:
        data, out = cfg.data_dir.resolve(), cfg.out_dir.resolve()
        if out == data or data in out.parents:
            raise ValueError(f"out_dir {out} must not be inside data_dir {data}")


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = _build(Config, raw, "")
    validate(cfg)
    return cfg


def detector_params(cfg: Config):
    """Detector Params (B1-B7) from the sky and detector config sections."""
    from .detector import Params

    d, s = cfg.detector, cfg.sky
    return Params(
        sky_threshold=s.threshold, open_close_radius=s.open_close_radius, min_area_px=s.min_area_px,
        sigma=d.sigma, sigma_w_prior=d.sigma_w_prior, truncate=d.truncate,
        rho_floor_frac=d.rho_floor_frac, rho_floor_percentile=d.rho_floor_percentile,
        k_tau=d.k_tau, k_eps=d.k_eps, w_min=d.w_min, s_hat_min=d.s_hat_min, s_hat_max=d.s_hat_max,
    )
