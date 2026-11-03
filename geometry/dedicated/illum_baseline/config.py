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
class DenseFitCfg:
    """N1 dense edge-aware plane fit (illum_baseline/geom/planefit.py: DenseFitParams)."""
    sigma_n: float = 3.0
    sigma_r: float = 0.01
    sigma_g: float = 3.0
    cond_max: float = 1.0e6
    min_weight_sum: float = 5.0
    g_init: str = "side_window"         # side_window | gaussian
    refit: int = 1


@dataclass
class FoldCfg:
    """Fold channel (illum_baseline/geom/fold.py: FoldParams; docs/fold_channel.md)."""
    cand_sigma_d: float = 1.0
    cand_sigma_T: float = 1.5
    cand_rate_factor: float = 3.0
    profile_half_len: int = 12
    width_min: float = 2.0
    width_max: float = 12.0
    k_min: int = 6
    k_margin: int = 2
    m: int = 8
    m_min: int = 4
    w_t: int = 4
    max_excluded: float = 0.2
    planarity_factor: float = 3.0
    b_tri_margin: float = 1.25
    null_rms_quantile: float = 0.95
    p_fa: float = 1.0e-3
    null_stride: int = 4
    lambda_clamp: list[float] = field(default_factory=lambda: [0.5, 1.0e4])
    null_model: str = "multiplicative"  # multiplicative | additive | additive_local
    lambda_n: float = 93.2              # additive nulls, k_min = 6 (illum_baseline/geom/data/lambda_n.json)
    local_tile: int = 48                # additive_local: tile size (px)
    local_min_samples: int = 50         # additive_local: fewer valid null samples -> the global sigma_geo
    cand_radius: float = 32.0           # additive_cand: null samples within this radius of x0 (px)
    cand_radius_wide: float = 48.0      # additive_cand: second radius when fewer than cand_min_samples
    cand_min_samples: int = 50          # additive_cand: else the global sigma_geo
    veto_k_factor: float = 2.0
    C_min: float = 0.85
    veto_mode: str = "fixed"            # fixed | capped (k2 capped by the next obstacle; TASK_FOLD_V2 F7)
    veto_curv_coef: float = 0.4         # capped: C_min_eff = 1 - coef (1 - C_curv)
    veto_curv_max: float = 0.8          # capped: decidable iff C_curv <= this (task 0.7; calibrated 0.8, F7)
    J_min: float = 0.0872
    J0: float = 0.25
    w_min: float = 0.05
    band_min: int = 2
    min_component_px: int = 5
    depth_jump_rel: float = 0.02
    crease_rescue: str = "off"          # off | laplacian_corroborated (TASK_GEOM_INTEGRATION G3)
    rescue_theta_deg: float = 8.0
    rescue_k_L: float = 3.0
    rescue_min_component_px: int = 10
    rescue_support: float = 0.7
    site: str = "nms"                   # nms | plane_intersection (TASK_GEOM_V2 V2)
    site_max_t: float = 3.0
    site_smooth_n: int = 5
    site_window_px: float = 4.0


@dataclass
class DepthCfg:
    """Depth-edge channel (illum_baseline/geom/depth.py: DepthParams; docs/depth_channel.md). Field for field."""
    disparity: str = "rho_f"            # rho_f | rho (ablation)
    sigma: float = 1.25
    truncate: float = 4.0
    sigma_w_prior: float = 1.0
    sigma_L_factor: float = 4.0
    k_g: float = 3.0
    lobe_half_len: float = 8.0
    lobe_step: float = 0.25
    k_h: float = 3.0
    s_lo_factor: float = 0.8
    sigma_w_max: float = 1.02           # P99 of sigma_w_c at unambiguous occlusions (geom/data/depth_calibration.json)
    centre_tol: float = 0.4
    symmetry: bool = False
    symmetry_ratio: float = 4.0
    k_min: int = 11
    m: int = 8
    w_t: int = 4
    m_min: int = 4
    max_excluded: float = 0.2
    planarity_factor: float = 3.0
    b_tri_margin: float = 1.25
    null_rms_quantile: float = 0.95
    gap_mode: str = "at_x0"
    doublet_coef: float = 1.03
    tau_low: float = 0.02
    tau_high: float = 0.04
    min_component_px: int = 5
    link_radius: float = 3.0
    z: float = 3.29
    null_stride: int = 4
    lambda_clamp: list[float] = field(default_factory=lambda: [0.5, 1.0e4])
    null_model: str = "additive"        # additive (default since D5) | multiplicative; both significances recorded
    lambda_n: float = 104.6             # additive null, k = 11 (geom/data/depth_calibration.json)
    smooth_n: int = 5
    sky_far_frac: float = 0.5
    canonical_focal: float = 300.0
    dZ0: float = 0.10                   # m
    w_min: float = 0.05
    band_min: int = 2
    sky_band: str = "depth"             # depth | depth_and_mask (G2: also paint the sky-mask boundary ring)
    sky_ring_px: int = 2
    sky_ring_near_px: float = 4.0


@dataclass
class LaplacianCfg:
    """Laplacian channel options (the detector section supplies everything else)."""
    pool_radius: int | None = None      # B6 pooling radius in px; null = the detector's r = ceil(hypot(sigma,
                                        # sigma_w_prior)) + 1 (3 at the defaults). Recorded only when set (G5)


@dataclass
class GeometryCfg:
    """Geometric-edge channels (docs/geometry_channels.md). Off by default: Stage B then runs the detector path."""
    enabled: bool = False
    channels: list[str] = field(default_factory=lambda: ["laplacian"])
    combine: str = "min"
    dense_fit: DenseFitCfg = field(default_factory=DenseFitCfg)
    fold: FoldCfg = field(default_factory=FoldCfg)
    depth: DepthCfg = field(default_factory=DepthCfg)
    laplacian: LaplacianCfg = field(default_factory=LaplacianCfg)


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
    geometry: GeometryCfg = field(default_factory=GeometryCfg)
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


def _coerce(annotation: str, value: Any, key: str) -> Any:
    """Float fields: YAML 1.1 reads "1e-4" (no decimal point) as a string; turn float-looking strings into floats
    (also inside lists) and fail clearly on anything else. Other types pass through unchanged."""
    if "float" not in str(annotation):
        return value

    def one(v: Any) -> Any:
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"{key}: expected a number, got {v!r}") from None
        return v

    return [one(v) for v in value] if isinstance(value, list) else one(value)


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
            kwargs[name] = _coerce(f.type, data[name], f"{where}.{name}" if where else name)
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
    _validate_geometry(cfg.geometry)
    if cfg.paths.data_dir is not None and cfg.paths.out_dir is not None:
        data, out = cfg.data_dir.resolve(), cfg.out_dir.resolve()
        if out == data or data in out.parents:
            raise ValueError(f"out_dir {out} must not be inside data_dir {data}")


def _validate_geometry(g: GeometryCfg) -> None:
    from .geom.channels import CHANNELS

    if not isinstance(g.channels, list) or not g.channels:
        raise ValueError("geometry.channels must be a non-empty list")
    unknown = [c for c in g.channels if c not in CHANNELS]
    if unknown:
        raise ValueError(f"geometry.channels: unknown {unknown}; available {sorted(CHANNELS)}")
    if len(set(g.channels)) != len(g.channels):
        raise ValueError("geometry.channels has duplicates")
    if g.combine != "min":
        raise ValueError(f"geometry.combine must be 'min' (the only mode implemented), got {g.combine!r}")
    d, f = g.dense_fit, g.fold
    if d.g_init not in ("side_window", "gaussian"):
        raise ValueError(f"geometry.dense_fit.g_init must be side_window|gaussian, got {d.g_init!r}")
    if min(d.sigma_n, d.sigma_r, d.sigma_g, d.cond_max) <= 0 or d.refit < 0:
        raise ValueError("geometry.dense_fit: scales and cond_max must be > 0, refit >= 0")
    if f.null_model not in ("multiplicative", "additive", "additive_local", "additive_cand"):
        raise ValueError("geometry.fold.null_model must be multiplicative|additive|additive_local|additive_cand, "
                         f"got {f.null_model!r}")
    if f.veto_mode not in ("fixed", "capped"):
        raise ValueError(f"geometry.fold.veto_mode must be fixed|capped, got {f.veto_mode!r}")
    if not 0 < f.veto_curv_coef < 1 or not 0 < f.veto_curv_max < 1:
        raise ValueError("geometry.fold: veto_curv_coef and veto_curv_max must be in (0, 1)")
    if f.local_tile < 4 or f.local_min_samples < 1:
        raise ValueError("geometry.fold: local_tile >= 4, local_min_samples >= 1")
    if not 0 < f.p_fa < 1:
        raise ValueError("geometry.fold.p_fa must be in (0, 1)")
    if len(f.lambda_clamp) != 2 or not 0 < f.lambda_clamp[0] <= f.lambda_clamp[1]:
        raise ValueError("geometry.fold.lambda_clamp must be [lo, hi] with 0 < lo <= hi")
    if not 1 <= f.m_min <= f.m:
        raise ValueError("geometry.fold: need 1 <= m_min <= m")
    if f.crease_rescue not in ("off", "laplacian_corroborated"):
        raise ValueError(f"geometry.fold.crease_rescue must be off|laplacian_corroborated, got {f.crease_rescue!r}")
    if (f.site not in ("nms", "plane_intersection", "hessian_peak") or f.site_max_t <= 0 or f.site_smooth_n < 1
            or f.site_window_px <= 0):
        raise ValueError(f"geometry.fold.site must be nms|plane_intersection|hessian_peak (site_max_t, site_window_px "
                         f"> 0, site_smooth_n >= 1), "
                         f"got {f.site!r}")
    if f.rescue_theta_deg < 0 or f.rescue_k_L < 0 or f.rescue_min_component_px < 1 or not 0 < f.rescue_support <= 1:
        raise ValueError("geometry.fold: rescue_theta_deg, rescue_k_L >= 0, rescue_min_component_px >= 1, "
                         "rescue_support in (0, 1]")
    if not 0 < f.C_min <= 1:
        raise ValueError("geometry.fold.C_min must be in (0, 1]")
    if not 0 < f.w_min < 1:
        raise ValueError("geometry.fold.w_min must be in (0, 1)")
    if min(f.cand_sigma_d, f.cand_sigma_T, f.cand_rate_factor, f.J0, f.b_tri_margin, f.veto_k_factor) <= 0:
        raise ValueError("geometry.fold: scales, factors and J0 must be > 0")
    if f.k_min < 1 or f.w_t < 0 or f.null_stride < 1 or not 0 <= f.max_excluded < 1:
        raise ValueError("geometry.fold: k_min >= 1, w_t >= 0, null_stride >= 1, 0 <= max_excluded < 1")
    if g.laplacian.pool_radius is not None and (not isinstance(g.laplacian.pool_radius, int)
                                                 or g.laplacian.pool_radius < 1):
        raise ValueError(f"geometry.laplacian.pool_radius must be null or an int >= 1, got {g.laplacian.pool_radius!r}")
    e = g.depth
    if e.disparity not in ("rho_f", "rho"):
        raise ValueError(f"geometry.depth.disparity must be rho_f|rho, got {e.disparity!r}")
    if e.null_model not in ("multiplicative", "additive"):
        raise ValueError(f"geometry.depth.null_model must be multiplicative|additive, got {e.null_model!r}")
    if e.sky_band not in ("depth", "depth_and_mask") or e.sky_ring_px < 0 or e.sky_ring_near_px < 0:
        raise ValueError(f"geometry.depth.sky_band must be depth|depth_and_mask (ring >= 0), got {e.sky_band!r}")
    if e.gap_mode != "at_x0":
        raise ValueError("geometry.depth.gap_mode must be at_x0 (min_span is recorded only)")
    if len(e.lambda_clamp) != 2 or not 0 < e.lambda_clamp[0] <= e.lambda_clamp[1]:
        raise ValueError("geometry.depth.lambda_clamp must be [lo, hi] with 0 < lo <= hi")
    if min(e.sigma, e.truncate, e.sigma_L_factor, e.lobe_half_len, e.lobe_step, e.sigma_w_max, e.centre_tol, e.z,
           e.dZ0, e.link_radius, e.canonical_focal, e.lambda_n, e.doublet_coef) <= 0 or min(e.k_g, e.k_h,
                                                                                            e.sigma_w_prior) < 0:
        raise ValueError("geometry.depth: scales, tolerances, z, dZ0, link_radius must be > 0; k_g, k_h >= 0")
    if not 0 <= e.tau_low <= e.tau_high:
        raise ValueError("geometry.depth: need 0 <= tau_low <= tau_high")
    if not 1 <= e.m_min <= e.m or e.k_min < 1 or e.w_t < 0 or e.null_stride < 1 or not 0 <= e.max_excluded < 1:
        raise ValueError("geometry.depth: 1 <= m_min <= m, k_min >= 1, w_t >= 0, null_stride >= 1, "
                         "0 <= max_excluded < 1")
    if not 0 < e.w_min < 1 or not 0 < e.sky_far_frac <= 1 or e.smooth_n < 1 or e.band_min < 0:
        raise ValueError("geometry.depth: w_min in (0, 1), sky_far_frac in (0, 1], smooth_n >= 1, band_min >= 0")


PATH_KEYS = ("data_dir", "out_dir")


def _resolve_path(value: str | None, base: Path) -> str | None:
    """Absolute path string: ~ expanded, relative paths taken relative to `base`."""
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def apply_sets(cfg: Config, sets: list[str] | None) -> Config:
    """Apply --set overrides, then validate. Keys are section[.subsection].key=value (e.g. geometry.fold.p_fa=1e-4)
    or a top-level key=value (device, seed);
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
        *parents, name = key.split(".")
        obj = cfg
        for part in parents:  # nested sections, e.g. geometry.fold.p_fa
            obj = getattr(obj, part, None)
            if obj is None or not dataclasses.is_dataclass(obj):
                raise ValueError(f"--set: unknown key {key!r}")
        if name not in {f.name for f in dataclasses.fields(obj)} or dataclasses.is_dataclass(getattr(obj, name)):
            raise ValueError(f"--set: unknown key {key!r}")
        if parents == ["paths"] and name in PATH_KEYS:
            value = _resolve_path(value, Path.cwd())
        ftype = {f.name: f.type for f in dataclasses.fields(obj)}[name]
        setattr(obj, name, _coerce(ftype, value, f"--set {key}"))
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


def dense_fit_params(cfg: Config):
    """geom DenseFitParams from geometry.dense_fit."""
    from .geom.planefit import DenseFitParams

    return DenseFitParams(**dataclasses.asdict(cfg.geometry.dense_fit))


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
