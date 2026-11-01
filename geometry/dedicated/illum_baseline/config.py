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
    stage_a_variant: str | None = None  # None: out_dir/stage_a; else out_dir/stage_a/<variant>


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
    exposure_mode: str = "p99"          # p99 | median_rolloff
    dither: str = "none"                # none | tpdf
    da3_float_input: bool = False       # not supported (DA3 re-quantizes array input to 8-bit RGB)


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
    tta: str = "none"                   # none | shift_flip
    tta_shift_set: str = "spec"         # spec: (0,0),(7,0),(0,7),(7,7); latin14: 14 shifts covering all phases mod 14
    tta_feather_px: int = 0             # 0: plain average over covering passes; >0: smoothstep feathering width


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
    a_min: float | None = None          # threshold floor (TASK_070_STRIPES T4); None = pure noise gate
    lambda_ref: float = 15.0            # px, reference ripple wavelength for a_min
    eps_floor: bool = False             # with a_min: eps = max(k_eps s_hat, (k_eps/k_tau) tau_min); user-approved extension
    prefilter: str = "none"             # none | trilateral: tangent-plane bilateral on rho before the Laplacian
    tri_sigma_s: float = 3.0            # px
    tri_sigma_r: float = 0.01           # relative deviation from the local tangent plane
    tri_sigma_g: float = 3.0            # px, scale of the tangent-plane gradient
    tri_shat: str = "unfiltered"        # s_hat from the unfiltered | filtered response
    scale_band: bool = False            # scale-ratio test: q from L_sigma - L_{band_factor*sigma} (cancels smooth curvature)
    band_factor: float = 2.0
    scale_band_mode: str = "replace"    # replace: q = |band response|; veto: q = min(q, g |band|), g = 1/(1 - 1/band_factor)


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
    if p.exposure_mode not in ("p99", "median_rolloff"):
        raise ValueError(f"preprocess.exposure_mode must be p99|median_rolloff, got {p.exposure_mode!r}")
    if p.dither not in ("none", "tpdf"):
        raise ValueError(f"preprocess.dither must be none|tpdf, got {p.dither!r}")
    if p.da3_float_input:
        raise ValueError("preprocess.da3_float_input is not supported: DA3's InputProcessor._load_image converts "
                         "array input with Image.fromarray(...).convert('RGB'), i.e. 8-bit (docs/api_notes.md)")
    v = cfg.paths.stage_a_variant
    if v is not None and (not str(v) or not str(v).replace("_", "").replace("-", "").isalnum()):
        raise ValueError(f"paths.stage_a_variant must be a simple name, got {v!r}")
    if not 0 < p.exposure_percentile <= 100:
        raise ValueError("preprocess.exposure_percentile must be in (0, 100]")
    if cfg.focal.mode not in ("per_image", "pooled"):
        raise ValueError(f"focal.mode must be per_image|pooled, got {cfg.focal.mode!r}")
    if cfg.network.long_side % cfg.network.multiple:
        raise ValueError("network.long_side must be a multiple of network.multiple")
    if cfg.da3.tta not in ("none", "shift_flip"):
        raise ValueError(f"da3.tta must be none|shift_flip, got {cfg.da3.tta!r}")
    if cfg.da3.tta_shift_set not in ("spec", "latin14"):
        raise ValueError(f"da3.tta_shift_set must be spec|latin14, got {cfg.da3.tta_shift_set!r}")
    if cfg.da3.tta_feather_px < 0:
        raise ValueError("da3.tta_feather_px must be >= 0")
    if cfg.detector.scale_band_mode not in ("replace", "veto"):
        raise ValueError(f"detector.scale_band_mode must be replace|veto, got {cfg.detector.scale_band_mode!r}")
    if cfg.detector.band_factor <= 1:
        raise ValueError("detector.band_factor must be > 1")
    if cfg.detector.prefilter not in ("none", "trilateral"):
        raise ValueError(f"detector.prefilter must be none|trilateral, got {cfg.detector.prefilter!r}")
    if cfg.detector.tri_shat not in ("unfiltered", "filtered"):
        raise ValueError(f"detector.tri_shat must be unfiltered|filtered, got {cfg.detector.tri_shat!r}")
    if min(cfg.detector.tri_sigma_s, cfg.detector.tri_sigma_r, cfg.detector.tri_sigma_g) <= 0:
        raise ValueError("detector.tri_sigma_* must be > 0")
    if cfg.detector.eps_floor and cfg.detector.a_min is None:
        raise ValueError("detector.eps_floor requires detector.a_min")
    if cfg.detector.a_min is not None and not cfg.detector.a_min > 0:
        raise ValueError("detector.a_min must be > 0 or null")
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


PATH_KEYS = ("data_dir", "out_dir")


def _resolve_path(value: str | None, base: Path) -> str | None:
    """Absolute path string: ~ expanded, relative paths taken relative to `base`."""
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def apply_sets(cfg: Config, sets: list[str] | None) -> Config:
    """Apply --set overrides, then validate. Keys are section.key=value or a top-level key=value (device, seed);
    values are parsed as YAML. Relative paths.data_dir / paths.out_dir are taken relative to the working directory."""
    for item in sets or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects section.key=value or key=value, got {item!r}")
        value = yaml.safe_load(val)
        if "." not in key:
            top = {f.name: f for f in dataclasses.fields(Config)}
            if key not in top or dataclasses.is_dataclass(getattr(cfg, key)):
                raise ValueError(f"--set: unknown top-level key {key!r}")
            setattr(cfg, key, value)
            continue
        section, name = key.split(".", 1)
        obj = getattr(cfg, section, None)
        if obj is None or not dataclasses.is_dataclass(obj) or name not in {f.name for f in dataclasses.fields(obj)}:
            raise ValueError(f"--set: unknown key {key!r}")
        if section == "paths" and name in PATH_KEYS:
            value = _resolve_path(value, Path.cwd())
        setattr(obj, name, value)
    validate(cfg)
    return cfg


def apply_paths(cfg: Config, input_dir: str | Path | None = None, output_dir: str | Path | None = None) -> Config:
    """Command-line --input/--output (relative to the working directory); they take precedence over the YAML."""
    if input_dir is not None:
        cfg.paths.data_dir = _resolve_path(input_dir, Path.cwd())
    if output_dir is not None:
        cfg.paths.out_dir = _resolve_path(output_dir, Path.cwd())
    validate(cfg)
    return cfg


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config. Relative paths.data_dir / paths.out_dir are resolved against the YAML
    file's own directory, so configs stay valid when the repository moves; the resolved (absolute) paths are what
    run snapshots record."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = _build(Config, raw, "")
    for key in PATH_KEYS:
        setattr(cfg.paths, key, _resolve_path(getattr(cfg.paths, key), path.resolve().parent))
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
        a_min=d.a_min, lambda_ref=d.lambda_ref, eps_floor=d.eps_floor,
        prefilter=d.prefilter, tri_sigma_s=d.tri_sigma_s, tri_sigma_r=d.tri_sigma_r, tri_sigma_g=d.tri_sigma_g,
        tri_shat=d.tri_shat, scale_band=d.scale_band, band_factor=d.band_factor, scale_band_mode=d.scale_band_mode,
    )
