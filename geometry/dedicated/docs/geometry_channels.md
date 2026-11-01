# Geometric-edge channels: architecture

Status as of 2026-10-31: **as built** (TASK_FOLD_CHANNEL F0–F5). Project root `geometry/dedicated/`. The fold channel's
math, parameters and results are in `docs/fold_channel.md`; this document covers the architecture shared by all
channels, the Stage B integration, the bit-for-bit guarantees, and how to add the planned depth channel.

Conventions, as in `docs/pipeline_tri_veto4.md`:

- All geometry runs at **network resolution** (504×378 or 378×504), in the **upright frame**, in **float64**, in pure
  numpy/scipy. Nothing in `illum_baseline/geom/` imports torch, the networks or the RGB image.
- Pixel `x = (u, v)`: u is the column, v the row (downward). Camera frame is OpenCV (X right, Y down, Z forward).
- `ρ = 1/D` on `Ω = ¬S`, `ρ = 0` on the sky `S`. `ρ_f` is the trilateral-filtered disparity (`detector.trilateral`).

## 1. Module map

`illum_baseline/geom/`:

| module | contents |
|---|---|
| `camera.py` | `Intrinsics(f_x, f_y, c_x, c_y, W, H, sigma_f_x, sigma_f_y)`; `from_stage_a(meta_a)` (f_x = fx·s_x, f_y = fy·s_y, active focal; centred principal point); `ray`, `rays`, `pixel_grid`; `scaled(κ)` |
| `context.py` | `GeometryContext`: lazily cached, read-only per-image fields shared by all channels (section 3.2) |
| `planefit.py` | `DenseFitParams`, `dense_fit` (N1, side-window gradient init + refit), `side_window_gradient`; `window_fit` (masked OLS with covariance, for the two-sided windows) |
| `normals.py` | `plane_vector`, `normals` (n̂, Σ_n, σ_θ, μ, ‖μ‖), `plane_distance`, `focal_bracket` |
| `sampling.py` | `bilinear(arr, valid, u, v) -> (values, ok)` (ok: all four corners inside and valid) |
| `twosided.py` | **shared by the fold and the future depth channel.** `two_sided_fit` → `TwoSidedResult`; `null_projection`, `quad_form`, `chi2_statistic`, `fold_angle`, `tangent_basis`; `window_rms`, `planarity_scale`; `b_tri` (contamination envelope from `data/b_tri.json`). Decision-free (section 3.4) |
| `candidates.py` | `normal_structure_tensor`, `fold_candidates` (F2.1), `rounding_width` (F2.2), `Candidates`; stub `depth_candidates` |
| `fold.py` | `FoldParams`, `FoldChannel` (F2.2–F2.8, three null models) |
| `laplacian.py` | `LaplacianChannel`: adapter around `detector.compute_weights`, unchanged |
| `channels.py` | `ChannelResult`, `Channel` protocol, `combine`, `CHANNELS` (built on first access) |
| `evaluate.py` | fold precision/recall against `annotations/folds/<stem>.json` |
| `synthetic.py` | ray-cast test scenes with exact surface IDs and normals (folds, steps, box + cylinders, chamfers) |
| `data/` | `b_tri.json`, `lambda_n.json` (from `scripts/calibrate_fold.py`) |

Outside the subpackage:

| file | change | old behaviour preserved by |
|---|---|---|
| `detector.py` | code motion: `tangent_plane_weights` (with an optional `grad=`), `prepare_disparity` (section 5) | bit-for-bit tests |
| `config.py` | `GeometryCfg` (+ `DenseFitCfg`, `FoldCfg`), validation, nested `--set`, float coercion | `enabled: false` built-in default |
| `stage_b.py` | `run_geometry`, `_process_one_geometry` (branch only when enabled); JSON blocks moved into `_detector_record` / `_meta_common` unchanged | repro tests |
| `manifest.py` | NaN for missing detector scalars (fold-only runs); `combined_*`, `normals_computed` columns | unchanged values for detector runs |
| `viz_fold.py`, `diag_fold.py`, `cli.py` | fold sheet, profile tool (`diag-fold`), `eval-fold`; `viz` renders both sheets | – |
| `configs/candidate_fold_only.yaml`, `configs/candidate_lap_fold.yaml` | runs `fold_only`, `tri_veto4_fold` | `configs/default.yaml` unchanged |

## 2. Data flow (one image, `geometry.enabled: true`)

```
Stage A cache (upright): D, sky_prob        Stage A JSON: geocalib fx, fy, focal_uncertainty; network s_x, s_y; focal.mode
        │                                          │
        │                               Intrinsics.from_stage_a
        ▼                                          ▼
GeometryContext(D, sky_prob, K, detector Params, DenseFitParams)        fields computed on first access, read-only
  disparity: S, Ω, ρ, ρ_floor      detector.prepare_disparity (the compute_weights code)
  rho_f                            detector.trilateral with tri_sigma_s/r/g (whatever detector.prefilter says)
  dense                            N1: side-window g → weighted fit → refit with g = β
  normals: n̂, Σ_n, σ_θ, μ          N2, valid where the fit is valid
  L_over_rho_f                     un-vetoed L/ρ on ρ_f (convexity record only)
        │
        ├── LaplacianChannel.run(ctx, asdict(detector_params(cfg)))  → compute_weights(ctx.D, ctx.sky_prob, Params)
        ├── FoldChannel.run(ctx, asdict(cfg.geometry.fold))           → candidates → two-sided fits → gate → veto → w_fold
        │        channels run in the order of geometry.channels; each only reads ctx
        ▼
combine(results, "min") → w_net (upright, network res)
        ▼
deliverable_w (upsample upright, clip to the smallest channel w_min, to_stored), net_to_stored, savez, parquet, JSON,
viz/<stem>.png (detector sheet, when laplacian runs), viz_fold/<stem>.png (fold sheet, when fold runs)
```

With `geometry.enabled: false`, `process_one` runs the detector path, byte for byte.

## 3. Interfaces

### 3.1 Channels

```python
@dataclass
class ChannelResult:
    name: str
    w_net: np.ndarray                  # (H, W) in [w_min, 1], network res, upright
    strength: np.ndarray | None        # dimensionless per-pixel strength
    candidates: pd.DataFrame | None    # one row per candidate (all, with accept/reject reasons)
    arrays: dict[str, np.ndarray]      # extra network-res arrays, saved as "<name>_<key>"
    scalars: dict[str, float]          # calibrations, stage counts, runtimes
    w_min: float                       # the channel's floor (the deliverable is clipped at the smallest)
    tables: dict[str, pd.DataFrame]    # other per-image tables, saved as <stem>_<name>_<table>.parquet

class Channel(Protocol):
    name: str
    def run(self, ctx: GeometryContext, params: dict) -> ChannelResult: ...

CHANNELS                               # {"laplacian": LaplacianChannel, "fold": FoldChannel}; future "depth"
def combine(results, mode="min")       # elementwise min; one channel: its w_net unchanged; other modes: ValueError
```

`CHANNELS` is a module attribute built on first access (`channels.__getattr__`), because the channel modules import
`ChannelResult` from `channels.py`.

Strengths (dimensionless, so a later fusion rule can act on strengths rather than weights):

| channel | `strength` | meaning |
|---|---|---|
| `laplacian` | `q_pool / ε` | w = max(w_min, 1/(1 + strength)) where q_pool > 0 |
| `fold` | J = sin(min(Δθ(k₁), π/2)), max over accepted candidates whose band covers the pixel, else 0 | worst-case shading jump in units of the incident radiance Φ (`fold_channel.md` §4.7) |
| `depth` (planned) | ΔZ/Z_near (metric ΔZ as an extra array) | relative depth jump |

### 3.2 GeometryContext

```python
GeometryContext(D, sky_prob, K: Intrinsics, det: detector.Params, fit: DenseFitParams = DenseFitParams())
  # inputs: D, sky_prob, K, det, fit, shape, u, v (pixel grids)
  # cached, read-only (flags.writeable = False):
  disparity (S, ground, bad, rho, rho_floor, sky_frac)  → properties S, omega, rho, rho_floor
  rho_f                 # trilateral(ρ, Ω, ρ_floor, det.tri_sigma_s/r/g)
  g                     # Gaussian-derivative gradient of ρ_f at fit.sigma_g (not used by the dense fit since F2)
  dense: DenseFit       # alpha, beta, cov (H,W,3,3), m0 (Σ W), n_eff, s2, cond, valid
  normals: Normals      # n_hat, Sigma_n, sigma_theta, mu, mu_norm  → properties n_hat, Sigma_n, sigma_theta
  normal_valid
  L_over_rho_f
```

Channels only read the context; an accidental write raises. The per-offset trilateral weights are not cached
(253 × H × W × 8 B ≈ 385 MB); the weight generator is shared code instead.

### 3.3 Dense fit (N1)

One pass over the 253 offsets of ‖d‖ ≤ R_n = 9 with `detector.tangent_plane_weights(ρ_f, Ω, ρ_floor, σ_n, σ_r, σ_g,
grad=g)`. The regressand is centred on the tangent plane (same estimator, no cancellation in Σ W r²). Gradient g:
side-window initialisation (8 half-disks every 45°, the one with the smallest relative residual) and one refit with
g = β; `DenseFitParams(g_init="gaussian", refit=0)` is the F1 fit. Validity: sky, R_n from the sky or the border,
Σ W < 5, cond > 10⁶. Sandwich covariance. Details and evidence: `fold_channel.md` §3.1.

### 3.4 Two-sided fit (shared)

```python
two_sided_fit(ctx, x0, e1, k, m, w_t, *, rms_max=None, m_min=4, max_excluded=0.2, fallback=True) -> TwoSidedResult
    # x0 (N, 2) (u, v), e1 (N, 2) unit, k (N,) or scalar; side "+" is +e1
    # rms_max(k_rows, m) -> max rms/alpha per row; None skips the contamination check

@dataclass
class SideResult:              # one per side: result.p (+e1), result.m (−e1); every field (N,) or (N, ...)
    alpha, beta, c, cov        # OLS plane about the sample centroid c; Cov(alpha, beta) (NaN for fallback)
    rms_rel, n, excl, m_used   # rms(residual)/alpha, samples used, excluded fraction, accepted window length (0: none)
    planar, fallback           # planar window fit accepted; dense normal used instead
    nhat, Sigma_n, sigma_theta # N2 at c (or the dense values for fallback)
    usable                     # property: planar | fallback

@dataclass
class TwoSidedResult:
    p, m                       # SideResult
    d_rho, d_rho_se            # Δρ*: signed plane gap at s_gap (0 when the planes cross), its standard error
    s_gap, planes_cross        # where on [−k, k] the gap is smallest; the planes intersect inside the span
    rho_p_gap, rho_m_gap       # each side's plane at x0 + s_gap e1
    dZ_rel                     # |Δρ*| / min(rho_p_gap, rho_m_gap) = ΔZ / Z_near
    d_rho_x0                   # the gap at x0 itself (record)
    dtheta, T                  # angle between the normals; δᵀ(Σ₊ + Σ₋)⁻¹δ in the tangent basis at n̄
    valid                      # both sides usable
    side_fields()              # flat scalar columns <field>_p / <field>_m for tables
```

Helpers: `null_projection(n_p, n_m, S_p, S_m) -> (δ, S)` (2-D tangent-basis difference and covariance),
`quad_form(δ, S, scale, add)` = δᵀ(scale S + add I)⁻¹δ, `window_rms(result)`, `b_tri(k, m, tri)`.

`two_sided_fit` makes no decisions: no thresholds, no gate, no weight. Each channel builds its own `rms_max`, gate,
veto and weight.

## 4. Stage B integration and saving

`<stem>.npz` (stored frame; network res unless noted):

| key | dtype | notes |
|---|---|---|
| `w` | float16 | combined, original resolution, clipped to [smallest channel w_min, 1] |
| `w_net` | float16 | combined |
| `q`, `L_over_rho` | float16 | unprefixed, when `laplacian` runs (the detector sheet reads them) |
| `<name>_w_net`, `<name>_strength`, `<name>_<array>` | float16 / bool | per channel (fold: `fold_r_n`) |
| `n_hat` (H, W, 3), `sigma_theta`, `normal_valid` | float16, float16, bool | only when a channel computed the normals; **n_hat components are in the upright camera frame** (all images here have orientation 1) |
| `D`, `sky_mask` | float32, bool | as in detector runs |

Tables (upright network-resolution coordinates): `<stem>_<name>_candidates.parquet`, `<stem>_<name>_<table>.parquet`
(fold: `null`). Fold candidate columns, in order: `u v u0 v0 e1_u e1_v r_n width profile_ok k1 k2`, the side fields
`alpha rms_rel n excl m_used planar fallback sigma_theta` with suffix `_p` then `_m`, `n_p_x … n_m_z`, `valid dtheta T
T_over_lambda T_add T_add_local sigma_geo_local gate_mult gate_add gate_add_local gate dtheta_gate dtheta_gate_add
dtheta_gate_add_local dtheta_k2 C no_veto veto_pass k2_clean component_px accepted reason J chord J_lo J_hi w_c band
d_rho d_rho_se s_gap planes_cross d_rho_x0 rho_p_gap rho_m_gap dZ_rel depth_jump convexity L_over_rho_x0` (angles in
rad; `reason` ∈ accepted, component, veto, gate, invalid_window; tested in `tests/test_geom_stage_b.py`).

`<stem>.json`: the detector-run blocks, with `detector` from the Laplacian channel's scalars (fold-only runs:
`rho_floor`, `sky_frac`, `n_bad_depth`); plus `normals_computed`, `combined` (fractions of w_net below 0.5 / 0.9 / 1,
the clip, channels, combine), `channels` (each channel's scalars, NaN → null), `params.geometry`,
`provenance.geometry_version` = 1. The top-level `geometry` block (s_x, s_y, orientation) is unrelated to the config
section of the same name.

Run directories refuse a different `geometry` section unless `--force`. The manifest has NaN detector columns in
fold-only runs and `combined_frac_w_lt_*`, `normals_computed` for geometry runs.

**Config.** `geometry: {enabled, channels, combine, dense_fit: DenseFitCfg, fold: FoldCfg}`, built-in default
`enabled: false`; `DenseFitCfg` / `FoldCfg` are field-for-field `DenseFitParams` / `FoldParams` (tested). Validation:
known channels, `combine: min`, null models, ranges. `--set` takes nested keys (`--set geometry.fold.p_fa=1e-4`);
float fields coerce float-looking strings (YAML reads `1e-4` as a string) and fail with "expected a number"
otherwise.

## 5. What moved out of `detector.py`, and why it stays bit-for-bit

1. **`tangent_plane_weights(rho, ground, rho_floor, sigma_s, sigma_r, sigma_g, grad=None)`**: a generator yielding
   `(dx, dy, nb, e, w)` per offset of the disk, in the original loop order, containing verbatim the former body of
   `trilateral` (gradient, padding, `inv`, `ws`, `e`, `w`). `trilateral` accumulates `num += w*e; wsum += w` over it.
   `grad` (default None: the Gaussian gradient, as before) lets the dense fit supply its own gradient.
2. **`prepare_disparity(D, sky_prob, p) -> Disparity`**: steps B1, B2 and the ρ_floor line of `compute_weights`,
   verbatim; `compute_weights` and the context both call it.

The argument: statements moved without changing any expression, operand order, dtype or loop order; numpy ufuncs and
`math.exp` are deterministic for identical inputs; the only reductions (`num`, `wsum`) keep their order. `ctx.rho_f`
is the same `trilateral` call as inside `compute_weights`. `LaplacianChannel` calls `compute_weights` with the same
`Params`, and `combine` over one channel returns its `w_net` unchanged.

| test | reference |
|---|---|
| `test_baseline_repro.py` | `baseline_v0` and `tri_veto4` (10 images) and batch2 `tri_veto4` (10 images): every array and the JSON blocks |
| `test_laplacian_channel_bit_identical_to_tri_veto4` | `geometry.enabled`, `[laplacian]`: w, w_net, q, L_over_rho, D, sky_mask, JSON detector block |
| `test_trilateral_bitwise_equals_pre_refactor_*` | a frozen copy of the pre-refactor `trilateral` (synthetic with sky, real 070) |
| `test_context_disparity_bitwise_equals_detector` | `ctx.S, rho, rho_floor, rho_f` vs `compute_weights` |

With `[laplacian, fold]` the trilateral filter runs twice (inside `compute_weights` and in the context), about
0.4 s; the adapter calls the detector unchanged by design.

## 6. Adding the depth channel

The planned depth channel (a Canny-like detector on the detrended disparity, strength ΔZ/Z_near) plugs in without
touching `fold.py`.

**Planned defaults (F2 decisions; not implemented):**

| parameter | value | why |
|---|---|---|
| window skip k | ≥ 11 (= R_s + 2) | windows clear the trilateral bias zone (9 px) and the bilinear / subpixel margin; F2 arm B (k_min = 11): `depth_jump` recall 0.995 on single synthetic steps, vs 0.04 at k_min = 6 |
| noise inflation λ_n (additive null) | 121.2 | Monte-Carlo E[T]/2 at k = 11 (`geom/data/lambda_n.json`); 93.2 at k = 6 |
| jump measure | Δρ* = smallest plane gap over s ∈ [−k, k], with `planes_cross` | zero at fold-like plane intersections |

**Steps.**

1. **Candidates** (`geom/candidates.py`): implement the stub `depth_candidates(ctx, params) -> Candidates`, e.g.
   Canny on the detrended disparity gradient (ρ_f minus the dense plane prediction, or |∇ρ_f|/ρ_f with NMS along the
   gradient). Return the same container as the fold candidates: `ij` (NMS pixel), `x0` (subpixel), `e1` (unit, across
   the edge; keep the (−90°, 90°] orientation), `rate`, `rate_map`, `e1_map`, `valid_map`, `threshold`,
   `median_rate`.
2. **Channel** (`geom/depth.py`, new): `DepthChannel` with `name = "depth"` and `run(ctx, params) -> ChannelResult`.
   - Window fits: `r = two_sided_fit(ctx, x0, e1, k, m, w_t, rms_max=..., fallback=False)` with k ≥ 11. Build
     `rms_max` as the fold channel does (null-sample ε̂ and Q_0.95, `b_tri(k, m, tri)`; at k ≥ 11 b_tri ≈ 6e-6). No
     fallback: Δρ* needs planar fits on both sides.
   - Read from `TwoSidedResult`: `r.p.planar`, `r.m.planar`, `r.valid` (usable sides), `r.d_rho`, `r.d_rho_se`
     (jump and its SE), `r.planes_cross` (fold-like intersection: not a depth edge), `r.s_gap`, `r.rho_p_gap`,
     `r.rho_m_gap`, `r.dZ_rel` (ΔZ/Z_near, the strength), and for diagnostics `r.p.rms_rel`, `r.m.rms_rel`,
     `r.p.excl`, `r.m.excl`, `r.d_rho_x0`. `r.dtheta` and `r.T` are available if the depth channel wants to
     separate occlusions with and without a normal change.
   - Metric ΔZ: Z = f_n·D/300 with f_n from the Stage A JSON (or `(ctx.K.f_x + ctx.K.f_y)/2`), Z_near = 1/max(ρ̂±)
     scaled the same way, ΔZ = Z_near·dZ_rel; save it as an extra array.
   - Gate: on |Δρ*|/SE with a per-image null calibration of its own (as λ̂ or σ_geo for the fold channel; the null
     sample can reuse `two_sided_fit` on a stride grid), plus `dZ_rel > depth_jump_rel`; never on dZ_rel alone.
   - Weight, rasterization, component filter: as `fold.py` does with w_c and bands, from dZ_rel.
3. **Register**: add `"depth": DepthChannel` to the dict returned by `channels.__getattr__`.
4. **Config**: `DepthCfg` in `config.py` (field-for-field with a `DepthParams` in `depth.py`; extend
   `test_config_params_field_parity`), `geometry.depth: DepthCfg`, validation in `_validate_geometry`, and
   `_channel_params` in `stage_b.py` already returns `asdict(getattr(cfg.geometry, name))`.
5. **Saving and viz**: automatic from `ChannelResult` (`depth_w_net`, `depth_strength`, `depth_<array>`,
   `<stem>_depth_candidates.parquet`, JSON `channels.depth`); add a depth sheet like `viz_fold.py` if needed and
   call it from `run_stage_b` and `cli.cmd_viz` as the fold sheet is.
6. **Tests** (test plan): Δρ* exact on noise-free synthetic steps at k ≥ 11 (`test_gap_folds_cross_steps_exact`
   shows the two-sided part already is); strict silhouette recall: the depth decision on ≥ 90% of valid candidates at
   synthetic occlusion silhouettes (box scene and single steps) at k ≥ 11 (the fold channel only reports its 0.75);
   no depth edge at folds (`planes_cross`); scale invariance; and `channels: [laplacian]` still bit-identical to
   tri_veto4.

The design keeps channels independent: the context owns every shared field (ρ_f, the dense fit, n̂),
`two_sided_fit` is decision-free and takes k per call, each channel calibrates its own null sample, and channels
communicate only through `ChannelResult`.

## 7. Decisions log

Settled in F0 (leans agreed) and revised in F1–F4; current state:

| # | decision | state |
|---|---|---|
| 1 | location: self-contained project `geometry/dedicated/` (code, configs, tests, docs, data, outputs, environment), seeded from `geometry/baseline/`, which stays frozen | as built |
| 2 | run from `geometry/dedicated/` with `env -u PYTHONPATH uv run`; `pyproject.toml`/`uv.lock` differ from baseline only in the `depth-anything-3` path (`../../DA3/Depth-Anything-3`) | as built |
| 3 | bit-for-bit references: `baseline_v0`, `tri_veto4` (10 images), batch2 `tri_veto4` (10 images), copied with their Stage A caches | as built |
| 4 | trilateral weights not cached; the generator is shared | as built |
| 5 | the rounding width is `width`; `w_fold` is the weight | as built |
| 6 | the width includes the dense fit's own transition (k₁ ≤ 8) | as built |
| 7 | width profile: s = −12…12, 10%/90% of the running maximum, 12 and `profile_ok = False` when invalid or flat | as built |
| 8 | a bilinear sample is excluded if any corner is sky, invalid or outside | as built |
| 9 | null sample at k = k_min with each pixel's own e₁ | as built |
| 10 | ε̂ from the null windows without a check | extended (F2): contamination threshold max(3ε̂, 1.25 b_tri(k, m), Q_0.95 of the null rms) |
| 11 | k₂ sides never fall back (`no_veto` when invalid) | revised (F2): k₂ windows have no contamination check either, only the exclusion rule; curved surfaces are vetoed, folds next to narrow faces too |
| 12 | fallback normal from the nearest pixel to the nominal window centre | as built |
| 13 | structure tensor on valid normals, normalized convolution | as built |
| 14 | e₁ angle in (−90°, 90°], side "+" is +e₁ | as built |
| 15 | silhouettes with a normal change are fold candidates; `depth_jump` is a record only | as built |
| 16 | deliverable clipped at the smallest channel w_min | as built |
| 17 | nested `--set` keys | as built, plus float coercion (F3) |
| 18 | Σ_ĝ = Cov(β)/α² | as built |
| F2 | arm A (k_min = 6) for the fold channel; k ≥ 11 for the depth channel | as built |
| F2 | N1 tangent-plane gradient from side windows + one refit (fixes the invalid band at steps) | as built |
| F2 | λ clamp [0.5, 10⁴]; Δρ* as the smallest plane gap | as built |
| F2–F4 | null model: multiplicative default; additive and additive_local recorded | as built (neither additive model met the switching rule) |
| F3 | normals saved only when computed (`normals_computed` recorded); detector sheet only with `laplacian`, fold sheet with `fold` | as built |

## 8. Runtime (CPU, measured in F4)

| step | time per image |
|---|---|
| trilateral ρ_f | 0.3–0.4 s |
| dense fit (side windows + 2 weighted passes) and normals | about 4.5–6 s |
| fold channel after the context (candidates, null sample, two-sided fits, veto, rasterization) | 0.6–0.9 s |
| Laplacian channel | about 0.5 s |

The task's target (fold channel < 5 s on the CPU) is missed by the context. Inference is expected to run on CUDA; the
torch backend follows after F5, checked against this numpy implementation.
