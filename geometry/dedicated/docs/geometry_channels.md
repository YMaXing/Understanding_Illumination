# Geometric-edge channels: architecture

Status as of 2026-11-01: **as built** (TASK_FOLD_CHANNEL F0–F5, TASK_FOLD_V2 F6–F9, TASK_DEPTH_CHANNEL D0–D6). Project
root `geometry/dedicated/`. The fold channel's math, parameters and results are in `docs/fold_channel.md`, the depth
channel's in `docs/depth_channel.md`; this document covers the architecture shared by all channels, the Stage B
integration, the bit-for-bit guarantees, the depth channel's place in it, and notes for the integration task.

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
| `twosided.py` | **shared by the fold and depth channels.** `two_sided_fit(..., gap_mode, usable)` → `TwoSidedResult`; `one_sided_constant` (the depth channel's thin-structure fit); `null_projection`, `quad_form`, `chi2_statistic`, `fold_angle`, `tangent_basis`; `window_rms`, `planarity_scale`; `b_tri` (contamination envelope from `data/b_tri.json`). Decision-free (section 3.4) |
| `candidates.py` | `normal_structure_tensor`, `fold_candidates` (F2.1), `rounding_width` (F2.2), `Candidates`; `depth_kernels`, `depth_field`, `depth_candidates` (D-K, D-1–D-3) |
| `fold.py` | `FoldParams`, `FoldChannel` (F2.2–F2.8, three null models) |
| `depth.py` | `DepthParams`, `DepthChannel` (D-4–D-7: jump with retries, gates, ΔZ, weights; two null models), `depth_null` |
| `laplacian.py` | `LaplacianChannel`: adapter around `detector.compute_weights`, unchanged |
| `channels.py` | `ChannelResult`, `Channel` protocol, `combine`, `CHANNELS` (built on first access) |
| `evaluate.py` | fold precision/recall against `annotations/folds/<stem>.json` |
| `synthetic.py` | ray-cast test scenes with exact surface IDs and normals (folds, steps, box + cylinders, chamfers) |
| `data/` | `b_tri.json`, `lambda_n.json` (from `scripts/calibrate_fold.py`); `depth_calibration.json` (from `scripts/calibrate_depth.py`) |

Outside the subpackage:

| file | change | old behaviour preserved by |
|---|---|---|
| `detector.py` | code motion: `tangent_plane_weights` (with an optional `grad=`), `prepare_disparity` (section 5) | bit-for-bit tests |
| `config.py` | `GeometryCfg` (+ `DenseFitCfg`, `FoldCfg`), validation, nested `--set`, float coercion | `enabled: false` built-in default |
| `stage_b.py` | `run_geometry`, `_process_one_geometry` (branch only when enabled); JSON blocks moved into `_detector_record` / `_meta_common` unchanged | repro tests |
| `manifest.py` | NaN for missing detector scalars (fold-only runs); `combined_*`, `normals_computed` columns | unchanged values for detector runs |
| `viz_fold.py`, `diag_fold.py`, `cli.py` | fold sheet, profile tool (`diag-fold`), `eval-fold`; `viz` renders every sheet of the run's channels | – |
| `viz_depth.py`, `diag_depth.py` | depth sheet, profile tool (`diag-depth`) | – |
| `configs/candidate_fold_only.yaml`, `configs/candidate_lap_fold.yaml`, `configs/candidate_fold_v2.yaml`, `configs/candidate_lap_fold_v2.yaml`, `configs/candidate_depth_only.yaml`, `configs/candidate_depth_fold.yaml` | runs `fold_only`, `tri_veto4_fold`, `fold_v2`, `tri_veto4_fold_v2`, `depth_v1`, `depth_fold_v1` | `configs/default.yaml` unchanged |

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
| `<name>_w_net`, `<name>_strength`, `<name>_<array>` | float16 / float32 / bool | per channel (fold: `fold_r_n`; depth: `depth_dZ` float32, `depth_q_over_s`, `depth_edge_mask`); float64 channel arrays are stored as float16, float32 / float16 / bool keep their dtype. `depth_strength` is ∞ at sky edges |
| `n_hat` (H, W, 3), `sigma_theta`, `normal_valid` | float16, float16, bool | only when a channel computed the normals; **n_hat components are in the upright camera frame** (all images here have orientation 1) |
| `D`, `sky_mask` | float32, bool | as in detector runs |

Tables (upright network-resolution coordinates): `<stem>_<name>_candidates.parquet`, `<stem>_<name>_<table>.parquet`
(fold and depth: `null`). Depth candidate columns: `docs/depth_channel.md` Appendix A.3. Fold candidate columns, in order: `u v u0 v0 e1_u e1_v r_n width profile_ok k1 k2`, the side fields
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

**Config.** `geometry: {enabled, channels, combine, dense_fit: DenseFitCfg, fold: FoldCfg, depth: DepthCfg}`, built-in
default `enabled: false`; `DenseFitCfg` / `FoldCfg` / `DepthCfg` are field-for-field `DenseFitParams` / `FoldParams` /
`DepthParams` (tested). The JSON `params.geometry.depth` block is written only for runs that include the depth channel
(earlier runs keep their records). Validation:
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

## 6. The depth channel and notes for the integration

**As built** (TASK_DEPTH_CHANNEL; math, parameters, results: `docs/depth_channel.md`). `"depth"` in `CHANNELS`,
`geometry.depth` in the config, runs `depth_v1` (`[depth]`) and `depth_fold_v1` (`[depth, fold]`, fold_v2 settings).
It reads only S, Ω, ρ, ρ_floor, ρ_f, K and the sky probability from the context: a `[depth]` run never builds the dense
fit or the normals (it calls `two_sided_fit` with `usable = Ω`, `fallback = False`, `gap_mode = "at_x0"`). Its
candidate table, null table, arrays and scalars follow the channel protocol; the strength is r = ΔZ/Z_near (∞ at sky
edges). Runtime 0.6–0.8 s per image after a 0.3–0.4 s context.

**What changed in shared code** (fold outputs bit for bit): `two_sided_fit` gained `gap_mode` (`min_span` default,
`at_x0`) and `usable` (default Ω ∧ valid normals) and records `d_rho_x0_se`, `d_rho_min_span` in every mode;
`one_sided_constant` is new. Stage B keeps float32 / float16 / bool channel arrays at their dtype (float64 → float16 as
before); the manifest adds `depth_*` columns for depth runs.

**Integration notes** written after TASK_DEPTH_CHANNEL (the open items are listed in `docs/depth_channel.md` §7; their
outcome after TASK_GEOM_INTEGRATION is in section 6.1, which supersedes the notes below):

- **Fusion on strengths vs weights.** Today the channels are combined by the minimum of their weights, the only mode
  implemented. Each channel also saves a dimensionless strength: fold J ∈ [0, 1] (the worst-case shading jump) and depth
  r = ΔZ/Z_near ∈ (0, ∞], with r = ∞ at sky edges and unbounded far sides. A fusion on strengths must treat ∞ as
  saturated, and must decide how a place where both fire (silhouettes with a normal change; DA3 contact steps such as
  the 079 box–plinth edge) is counted once. The depth weight is nearly binary on real edges (ΔZ ≫ ΔZ₀), so a binary
  weight with w_min is an option to decide together with the fusion.
- **Retiring the Laplacian channel.** On the 10 images w < 0.5 covers 10–33 % in tri_veto4 against 1–8 % for depth and
  3–11 % for depth + fold; the pinned ROIs show depth covering steps and limbs (median w 0.05–0.13) and leaving folds,
  creases and texture alone. Retiring the Laplacian channel needs a recall comparison against tri_veto4 on annotated
  fold / step / limb / texture classes, not only the pinned ROIs, and a decision on what the Laplacian catches that
  neither channel does (fine texture curvature is meant to be dropped; small real features are not).
- **Alignment and band coverage.** DA3 edges sit about (−0.27, −0.52) px from the RGB edges (D2); the depth band is
  2–3 px. Checking band coverage against RGB edges (and the sky-mask boundary) decides whether a subpixel shift of D
  or wider bands are needed.

### 6.1 Integration outcome (TASK_GEOM_INTEGRATION G0–G6): status of items 1–10

The integrated product is **geom_v1** (`configs/candidate_geom_v1.yaml`; consumer description `docs/geometric_edges.md`):
channels [depth, fold, laplacian] combined by the elementwise min; depth with the additive null and
`sky_band: depth_and_mask`; fold v2 with `crease_rescue: laplacian_corroborated` (8°, k_L 3); the Laplacian channel is
tri_veto4's detector with `pool_radius: 2`; band_min 2; no deliverable shift.

| # | item | status | outcome |
|---|---|---|---|
| 1 | DA3 alignment shift | **closed** (G2) | not stable on 20 images (pooled (−0.07, −0.47) px; dy CI excludes 0, but only 39 % of images within 0.3 px): no deliverable shift option |
| 2 | band coverage vs RGB | **closed** (G2) | `depth.sky_band: depth_and_mask` adopted (070 sky band 0.74 → 0.985); band_min stays 2: no band_min ≤ 4 reaches 90 % coverage of the RGB edge (0.82–0.88, the RGB-edge tail; 1.00 at the DA3 edge) |
| 3 | ΔZ₀ | **closed** (G3) | stays 0.10 m (1.00 m loses limbs 1.00 → 0.81 for little texture gain); the weight stays a function of ΔZ |
| 4 | fusion on strengths vs weights | **closed** (G6) | min of the weights (the only mode); strengths documented with units (`docs/geometric_edges.md` §3), ∞ at sky edges saturated |
| 5 | retiring the Laplacian channel | **closed** (G4, G5) | rule → (c) third channel: geom_v0 missed 0.584 [0.512, 0.656] of the tri-only area (labelled audit), mostly steps and folds; pooling radius 2 by the G5 rule |
| 6 | local σ_geo (fold) | **reopened, not fixed** (G4) | textured-surface fold gaps 97, 124, 125 stay open in the fold channel; covered by the Laplacian in geom_v1; geom_v2 candidate (7.1) |
| 7 | narrow-face veto (fold) | **reopened, not fixed** (G4) | narrow faces 15, 37 open in the fold channel (92 closed by the rescue); covered by the Laplacian in geom_v1; geom_v2 candidate (7.1) |
| 8 | fence clutter | **closed** (G3, G4) | relaxation kept (thin-ring clusters on fences, railings, balusters, foliage); no audit unit labelled artifact or clutter |
| 9 | moderate creases | **closed as a documented gap** (G3) | crease rescue as an option, on in geom_v1 (088 ridge 0.125 → 0.76); the 076 façade ledge stays blocked by the capped veto |
| 10 | recall vs tri_veto4 on true classes | **closed** (G4, G5) | labelled audit: missed true geometry 0.584 (geom_v0) → 0.007 [0, 0.021] (geom_v1, per unit; 0.079 per pixel) |

Findings recorded on the way: under min with the Laplacian, the depth channel sets w on ≤ 0.03 % of the pixels (its
deliverable role is the strengths r and ΔZ); fold placement (NMS vs plane site); the depth channel's component minimum on
thin contours. All three are in the geom_v2 candidate list (7.1) or the limitations (`docs/geometric_edges.md` §5).

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
| V2 F6 | targets pinned (`scripts/fold_null_rois.py` TARGETS): narrow-face edges (079, 076, 090 plinths; 080 side face) and two confirmed creases (076 façade ledge, 088 rock ridge); the 070 seam and the 083 collar groove rejected (no plateau–step profile) | as built |
| V2 F6 | the fold runs `fold_only`, `tri_veto4_fold` reproduce bit-for-bit with the v2 options off (repro test); v2 keys are left out of the JSON parameter records when their feature is off (`fold.params_record`) | as built |
| V2 F7 | capped veto (`veto_mode: capped`): k₂± capped by the next obstacle (invalid normals or outside beyond the own rounding, other gate-passed candidates beyond the whole rounding band, so sibling ridges of a rounded fold do not cap); C_min,eff = 1 − 0.4 (1 − C_curv); decidable iff C_curv ≤ **0.8** (task 0.7; calibrated by `scripts/calibrate_veto.py`, margins positive to 0.925 on synthetic data, and one-side-capped skips have C_curv 0.75–0.77); undecidable → strictly planar k₁ sides (`narrow_face`) or `veto_undecidable` | as built; default stays `fixed` |
| V2 F7 | `two_sided_fit(..., k_m=)`: a separate skip for the − side | as built |
| V2 F8 | gate arms with the capped veto: A0 multiplicative, A1 `additive_cand` (candidate-centred σ_geo), A2 `additive_local` 24-px tiles; no arm meets the switching rule, the gate stays multiplicative | as built |
| V2 F7–F9 | **`fold_v2` / `tri_veto4_fold_v2` are the fold reference for the depth channel and for integration**; the built-in default stays `veto_mode: fixed` (and `configs/default.yaml` keeps the geometry channels off) | decided 2026-11-01 |
| V2 F7–F9 | moderate creases: combined w of `tri_veto4_fold_v2` along the confirmed creases equals tri_veto4's (076 façade ledge median 0.747, 088 rock ridge 0.252); not both below 0.5, so the crease gate is **not** parked; open item in `fold_channel.md` §7 | as measured |
| V2 F9 | runs `fold_v2`, `tri_veto4_fold_v2` (`configs/candidate_fold_v2.yaml`, `candidate_lap_fold_v2.yaml`); fold sheet marks `narrow_face` / `veto_undecidable` and plots C / C_min,eff; `eval-fold` classes fold, step, limb, texture (not_fold kept) | as built |
| D0 | depth ROIs pinned (`scripts/depth_rois.py`): 070 sky boundary (band of the tri_veto4 sky mask), 079 silhouettes left (trimmed to v ≤ 238) and top-back L/R, 079 lit/shaded fold, 083 limbs L/R, 083 base rim, 083 bollard interior u 175–205 × v 260–318 (the legacy v 260–330 box overlaps the base rim and is kept only for comparability) | confirmed before D3 |
| D1 | depth field: g1 = −x g0/σ² normalized to Σ x g1 = 1 (correlation convention); Hessian from g2·g0, g0·g2, g1·g1, odd-reflect padding; n from the detrended gradient ∇ρ_σ − ∇ρ_{σ_L}; q = nᵀHn / max(ρ_σ, ρ_floor) | as built |
| D2 | gradient floor: g_n = \|∇Dd\|/max(ρ_σ, ρ_floor) > 3 σ̂_g, σ̂_g = median(g_n over Ω)/1.1774; ŝ_q over Ω away from sky and border (not the floor-valid pixels); q not zeroed on the sky (ρ = 0 there) | as built |
| D2 | centring stage \|s_p + s_m\| ≤ centre_tol s_sep (ghost lines next to steps on ρ_f); centre_tol **0.4** (D3) | as built |
| D2–D3 | σ_w,max = **P99** of σ_w,c at unambiguous real occlusions (P95 recorded): 1.05 px with lobes on q, **1.02 px** after D3 moved the lobe positions to h; s_hi 3.23 px | as built (rule P99, value re-calibrated) |
| D2 | `two_sided_fit(gap_mode, usable)`; `d_rho_x0_se`, `d_rho_min_span` recorded; fold defaults unchanged | as built |
| D2 | DA3-vs-RGB alignment: RGB edge = DA3 edge + (−0.27, −0.52) px at strong depth edges, consistent on 9/10 images; **no correction now**. Integration-task items: (a) confirm the shift on more images; (b) if stable, a versioned Stage A option for a fixed subpixel shift of D; (c) band-coverage check against RGB edges for the depth and fold channels | open (integration task) |
| D3 | lobe positions on the unnormalized h = nᵀHn, heights q at those peaks (q's normalization by ρ_σ drags the far lobe outward for r ≳ 1); the lobe is the nearest local peak of its run reaching ½ of the run maximum (thin structures); Δρ_doublet = 1.03 \|h\|_lobe s_sep² (= the spec's \|q\|_lobe ρ_σ(x₀) s_sep²) | as built |
| D3 | jump retries: plane m = 8 (contamination check) → m = 4 → constant fit → doublet (`thin`); a side whose window lies beyond the next edge (a D-3 candidate with r_doublet > τ_low in its strip) uses the constant fit on the clear span [s_a, d_next − s_a], s_a = ⌈s_hi/2⌉ + 1; doublet significance min(\|q_p\|, \|q_m\|)/ŝ_q | as built |
| D3 | hysteresis components link candidates within 3 px (`link_radius`), size = distinct pixels (8-connectivity of rounded pixels split oblique edges) | as built |
| D3 | contour smoothing: running median over 5 chained candidates (mutual nearest neighbours ahead/behind, n·n ≥ 0.5) of each neighbour's plane evaluated at the candidate's own x₀ (exact on planes) | as built |
| D3 | lower bound with SE inflated by √λ̂ (as the significance gate); r uses the matching lower bound when ρ̂₋ − z√λ̂ SE₋ ≤ 0; ΔZ_lb = inf when even ρ̂₋ + z√λ̂ SE₋ ≤ 0 | as built |
| D3 | sky edges (far span ≥ 50 % sky on S): r = significance = ΔZ = inf, w_c = w_min; strength inf there; sky_near rejects | as built |
| D3 | additive null for Δρ* recorded (`signif_add`, σ_geo,ρ solved so that the null median equals the χ²₁ median 0.455; λ_n = 104.6); gate stays multiplicative; decide in D5 (switch if r_gate P90/P10 > 3) | decided in D5: additive default (row D5 → D6) |
| D4–D5 | `geometry.depth.null_model` (multiplicative default \| additive; renamed from `null` before D5, the fold channel's key name), both significances recorded | as built |
| D4 | JSON `params.geometry.depth` only for runs with the depth channel; channel arrays keep float32/float16/bool dtypes (float64 → float16 as before); manifest `depth_*` columns only for depth runs; repro references extended by `fold_v2`, `tri_veto4_fold_v2` | as built |
| D4 | depth strength is inf at sky edges (and unbounded far sides): any fusion on strengths must treat inf as saturated | as built (integration note) |
| D5 | runs `depth_v1` (multiplicative), `depth_v1_add` (additive), `depth_fold_v1`; depth sheet `viz_depth/`, profile tool `diag-depth`; all D5 spot checks met under both nulls | as built |
| D5 | additive null for the depth channel: meets the decision rule (texture ≤ multiplicative + 0.02 on the 070 floor and both 079 grass ROIs; step/limb targets kept; synthetic box silhouette recall 0.91 → 0.95); **proposed as the default** | decided (row D5 → D6) |
| D5 | 080 side face edge R (fold target) is also an occluding silhouette (depth fires there correctly); 090 chain-link fence reported as a clutter case (thin rings; not tuned) | as measured |
| D5 → D6 | **depth `null_model: additive` is the default** (decision rule met); the fold channel stays multiplicative; the SE inflation of the lower bound and r_gate follows the active null (√λ̂ SE or √(λ_n SE² + σ_geo,ρ²/2) per side) | decided 2026-11-01, as built |
| pre-D6 | 080 side face edge R evaluated as class step in the depth evaluation (`scripts/depth_rois.py` RELABELLED); kept as a fold target (legacy) in the fold evaluation | as built |
| pre-D6 | 079 box–plinth contact and 075 kiosk base: steps in DA3's depth (step beats a continuous kink by ΔBIC 450–2750 at all 8 points; jumps 3–24 %), not folds misread; **no step-vs-kink integration item**; DA3 contact steps noted (both channels fire) | as measured |
| pre-D6 | 070 sky band: 457 of 1766 px with w_depth ≥ 0.5: 375 just beyond the depth band (the accepted edges sit ~1 px inside the sky-mask boundary), 20 gaps, 62 corners / junctions | as measured |
| D6 | final docs: `docs/depth_channel.md` (math, calibration, sensitivity, results, limitations, integration open items), this section 6, `docs/viz_guide.md` (depth sheet, `diag-depth`) | as built |
| G1 | geom_v0 = depth (additive) + fold_v2 by min; audit sampled on 32-px tile units of components > 1000 px (`audit_split/`), the literal-spec sample kept (`audit/`) | as built |
| G2 | DA3-vs-RGB shift not stable on 20 images (CI excludes 0 in dy, 39 % of images within 0.3 px): no deliverable shift option | decided by rule |
| G2 | band coverage: no band_min ∈ {2, 3, 4} reaches 90 % depth-edge coverage at the RGB edge (0.82–0.88; RGB-edge tail); `geometry.depth.sky_band: depth_and_mask` (sky band 0.74 → 0.985) as an option, default off | proposed (G2 report) |
| G3 | `geometry.fold.crease_rescue: laplacian_corroborated` as an option (default off; `ctx.s_hat_L` = tri_veto4's ŝ exactly); targets not met (076 ledge blocked by the capped veto) → item 9 documented gap | as built; closed by rule |
| G3 | ΔZ₀ stays 0.10 m (report: 0.30 / 1.00 m trade limb and step coverage for little texture gain) | as measured |
| G3 | thin-ring clusters (fences, railings, balusters, foliage, distant façades): keep the relaxation, no code | lean (revisit in G4) |
| G2/G3 review | approved: `sky_band: depth_and_mask` with band_min 2; `crease_rescue: laplacian_corroborated` 8° / 3 enters the G4 re-evaluation; ΔZ₀ 0.10 m; item 8 closed (no unit labelled artifact or clutter); the sky edges' `thin` flag kept and documented; the rescue's support is the component's majority sign of L/ρ_f (no separate convexity test) | decided 2026-11-02 |
| G4 | labelled audit (stratified pps estimator, stratified bootstrap): missed true geometry after the re-evaluation 0.584 [0.512, 0.656] of the tri-only area (step 0.341, fold 0.172, limb 0.038, crease 0.033; subscale 0.034 excluded); tri_fp 0.083; geom_fp 0.052 of geom-only | as measured |
| G4 | Laplacian rule → **(c) third channel** (missed ≫ 10 %, crease 5.7 % of it); no Laplacian-corroborated depth rescue proposed (open step gaps dropped mostly at the component stage: fragmented thin contours < 5 px, not at the significance gate) | proposed (G4 report) |
| G4 | items 6 (textured-surface folds 97, 124, 125 open) and 7 (narrow faces 15, 37 open; 92 closed by the rescue) **reopen** by the rule; geom_offset recorded as an open fold-placement item (vertical corners −2.0 px toward the oblique face, nosings +2.9 px up) | by rule |
| G4 review | **Laplacian (c)**: geom_v1 = [depth, fold, laplacian] by min; items 6, 7 reopened, not fixed in this task (the Laplacian covers the affected units in geom_v1; `docs/fold_channel.md` §7); depth component minimum on thin contours recorded (`docs/depth_channel.md` §7, no code); fold placement an open item; offset labels 161 / 173 / 165 / 180 kept (RGB-edge caveat) | decided 2026-11-02 |
| G5 | geom_v1 run on both batches, new bit-for-bit reference; `geometry.laplacian.pool_radius` option (null = detector r = 3, bit for bit); combined sheet `viz_geom/`; under min the depth channel and the sky ring add no w < 0.5 area (the Laplacian covers them), the fold channel 5.8 % / 9.9 % and the rescue 1.2 % / 2.1 % of the w < 0.5 area (batch1 / batch2); r sweep report only; fold-site diagnostic: plane-intersection site halves the convex-fold offset (−1.75 → −0.82 px) | as built / as measured |
| G5 review | pool_radius default stays null (= detector r = 3); rule "r = 2 if step, limb, fold coverage each within 0.02 of r = 3" met (−0.009, 0, −0.017) → **geom_v1 uses `geometry.laplacian.pool_radius: 2`**; the first run kept as `geom_v1_r3`; references geom_v1 and geom_v1_r3 on both batches; texture 0.254 → 0.198, w < 0.5 0.241 → 0.217 (batch1), 0.305 → 0.279 (batch2) | decided by rule 2026-11-03 |
| G6 | consumer doc `docs/geometric_edges.md` final; items 1–10 status (section 6.1); geom_v2 candidate list (7.1); default proposal prepared as a separate, unapplied commit (branch `proposal/geom_v1_default`) | as built |
| G6 review | **geom_v1 is the pipeline default** (proposal merged as it stands, with depth); the [fold, laplacian] variant changes w on ≤ 0.03 % of the pixels and saves 0.65 s per image, but geom_v2 needs depth's accepted edges and tables; Stage B 7.00 / 7.05 s per image (CPU), 4.5 s of it the dense fit; sheets on by default (about 24 s per image for the four sheets), `--no-viz` for dataset-scale runs | decided 2026-11-03 |
| V0 (TASK_GEOM_V2) | geom_v2 design: `docs/geometric_edges.md` §8 (Laplacian as a fallback; fold re-siting, short thin depth components, fold recall; options, runs, references, missing tools, flags) | design |
| V0 review | stage order J_min after the component minimum, last stage "accepted, w ≥ 0.5 (J ≤ J_min + J0)"; `linked` gap {4, 6} px, 20° tangent, same polarity, lateral ≤ 1.5 px; site smoothing in chain order; exclusion with sky edges, without the sky ring; `laplacian_w_fallback` saved only with the fallback; scorer imports the existing scripts | decided 2026-11-03 |
| V1 | scorer `scripts/geom_score.py` (reproduces the geom_v1 numbers); fold-gap map: 29 open fold / crease units, the gate stops most candidates in 21 (75 % of px), and the crease rescue is blocked by an undecidable veto in 21 units; V4 design proposed: `fold.rescue_undecidable: reject \| corroborate` | as measured / proposed |
| V1 review | V4 design approved (`fold.rescue_undecidable`, targets as proposed, plus the narrow-face split, smooth-curvature overlays, a report-only 6-px rescue minimum); item 6 is open (reopened G4, not fixed) | decided 2026-11-03 |
| V2 | `fold.site: nms \| plane_intersection` (default nms); valid sites 0.45–0.50 of accepted candidates; convex median \|offset\| 2.84 → 2.63 px (−0.21; rule ≥ 0.5 not met), vertical corners −2.00 → −1.05, nosings +2.90 → +1.50; fold-target coverage unchanged; geom_gain closures kept, relaxed share 0.664 → 0.573; **not proposed** | as measured |
| V2b | `fold.site: hessian_peak` (peak of −convexity · e1ᵀHe1/ρ within ±4 px; height test, doublet and window-end rejections; 5-candidate median); valid 0.76 / 0.79; convex median \|offset\| 2.84 → 1.70 px (−1.15), vertical corners −2.00 → −0.27; fold-target coverage unchanged; no geom_gain closure lost (relaxed 0.664 → 0.589); **proposed; V5 uses it** | as measured / proposed |
| V2b review | hessian_peak approved for V5 (fixed); same-sign neighbour limit added (the synthetic peak jumped; real images: 1 jump before, 0 after); scorer: RGB-edge coverage of all accepted fold candidates (reliable edges: inside ±8 px, strength ≥ 3 × the image median \|∇Y\|), the fourth V5 constraint; geom_gain / geom_offset drops read as the band moving; nosings stay +2.7 px | decided 2026-11-03 / as built |
| V3 | `depth.short_thin: off \| strong \| linked` (gap 4 / 6 px, 20°, same polarity, lateral ≤ 1.5 px); no variant closes a step-gap unit (strong: +971 accepted pieces, +839 relaxed step-gap px, tri_fp / texture-label +0.0006 / +0.0008); texture unchanged; **not proposed, documented gap** | as measured |
| V3 review | short_thin stays off (V5 included); thin structures stay the Laplacian's domain (depth_channel.md §7 item 11); V5 exclusion edges: accepted fold candidates (incl. rescued) and accepted depth candidates except thin doublets (sky edges kept); V4 runs with `fold.site: hessian_peak` | decided 2026-11-03 |
| V4 | `fold.rescue_undecidable: reject \| corroborate` (+ `ctx.L_veto_over_rho_f`); 1 of 21 veto-blocked units closed (target ≥ 7), their relaxed share 0.063 → 0.129; core missed 0.563 → 0.508; fails 079 grass (+0.017) and the 083 interior (0 → 0.021 core); **not proposed**; item 7 stays open, item 6 open | as measured |
| V4 review | rescue_undecidable stays reject; fold_channel.md §7: the V4 recall gain, the failure mode (the scale-ratio-vetoed Laplacian passes smooth curvature), a hessian-peak-sharpness candidate, unit closure structurally hard for narrow bands | decided 2026-11-03 |
| V5 | `laplacian.role: channel \| fallback` (+ `d_excl`; `illum_baseline/geom/fallback.py`); sweep d_excl {2, 3, 4, 6} × r {2, 3} × band_min {2, 3}: **no setting passes** (fold-candidate RGB coverage fails in all 16, best 0.778 vs 0.780; 99.8 % of the lost samples sit next to accepted folds with w_c ≥ 0.5); held-out agrees; retirement metric 0.563 (gap_unresolved 0.127) | as measured |
| V5 review / rerun | exclusion only around edges that relax (own w_c < 0.5); added: band excess reduced vs geom_v1 with the paired bootstrap CI excluding 0; rerun: fold-candidate RGB failures 16 → 1, others unchanged; **no setting passes** (the 8 that meet the first rule never reduce band excess, the 7 that reduce it fail missed geometry and depth-edge RGB coverage) | as measured |
| TASK_GEOM_V2 end | **the geom_v2 line stops at V5; geom_v1 stays the pipeline default; V6 skipped**; the V2–V5 options stay available at their defaults (geom_v1 bit for bit) | decided by rule 2026-11-03 |
| wrap-up | final check: fallback with fold-only relaxing exclusion (d_excl 2 / 3 / 4, r 2, band_min 2) meets every original constraint but never cuts band excess (best −0.005 [−0.012, +0.003]); **geom_v1_1 = geom_v1 + fold.site hessian_peak**, a new bit-for-bit reference (both batches); default proposal prepared on `proposal/geom_v1_1_default` (not applied) | as built / proposed |
| wrap-up review | **geom_v1_1 is the pipeline default** (`configs/default.yaml` = `configs/candidate_geom_v1_1.yaml`); the proposal (`900766d`) passed the full suite on its branch (218 passed, 2 deselected) and was merged into `yx_dev`; against geom_v1: fold-candidate RGB-edge coverage 0.800 → 0.809 (core 0.671 → 0.696), band excess 0.214 → 0.217 (paired +0.003 [0.000, +0.009]), missed geometry 0.079 → 0.079, ROI coverage and texture within 0.001, w < 0.5 (deliverable) 0.2156 → 0.2151 / 0.2765 → 0.2750, Stage B 7.70 → 7.74 / 7.75 → 7.87 s per image (batch1 / batch2); geom_v1 stays a bit-for-bit reference (`docs/geometric_edges.md` §7.1) | decided 2026-11-03 |
| W1 | `fold.window_cap: off \| next_ridge` (end a side's k1 window 2 px before the next NMS fold ridge, skip 6, only if m′ ≥ 4); run `geom_v1_1_wcap`: 070 middle riser 0.32 / 0.48 → 1.00 / 1.00, every constraint met (all units and batch1 units), 068 laps untouched (2 of 1 049); **proposed by rule, not applied**; flag: 21 % of candidates capped image-wide, core crease coverage −0.16 (088 rock ridge), geom_gain −0.03 (`docs/geometric_edges.md` §8.14) | as measured / proposed |

### 7.0 Default history (`configs/default.yaml`)

| period | default run | config (kept as) | change | decided | record |
|---|---|---|---|---|---|
| before TASK_GEOM_INTEGRATION G6 | `tri_veto4` | `configs/candidate_tri_veto4.yaml` | Laplacian detector only (baseline_v0 → tri ŝ + veto, band factor 4), geometry channels off | inherited from `geometry/baseline/` when the project was seeded | `docs/pipeline_tri_veto4.md` |
| G6 review → TASK_GEOM_V2 wrap-up | `geom_v1` | `configs/candidate_geom_v1.yaml` | [depth, fold, laplacian], min; capped veto, crease rescue, sky band depth_and_mask, pool radius 2 | 2026-11-03 (proposal `ff6f443`) | `docs/geometric_edges.md` §7.0 |
| since the wrap-up review | `geom_v1_1` | `configs/candidate_geom_v1_1.yaml` | geom_v1 + `fold.site: hessian_peak` | 2026-11-03 (proposal `900766d`) | `docs/geometric_edges.md` §7.1 |

All three runs stay bit-for-bit references on both batches (`tests/test_baseline_repro.py`); the configs of the earlier
defaults are unchanged, so each can be rerun with `--config`.

### 7.1 geom_v2 candidate list (recorded in G6; no code)

Outcome (TASK_GEOM_V2, `docs/geometric_edges.md` §8): fold re-siting built (`fold.site: hessian_peak` met its rule);
thin-contour depth components built and not proposed (documented gap); items 6 / 7: `fold.rescue_undecidable` built and not
proposed, both items stay open; the Laplacian fallback built, no setting passed the selection rule, so the line stopped
with geom_v1 as the default; in the wrap-up geom_v1 + the Hessian siting (geom_v1_1) became the default (7.0).

1. **Fold re-siting.** Place the fold at the intersection of the two fitted side planes (ρ̂₊ = ρ̂₋ along e1) instead of
   the NMS ridge of the rotation rate, where the planes cross within ±k₁. G5 diagnostic: on convex folds the RGB-edge
   offset drops from −1.75 to −0.82 px (frontal-face convention; vertical corners −2.02 → −1.05, nosings +2.59 →
   +1.21); no consistent effect on concave contact folds; unusable on narrow plinths (planes cross for 34 % of the
   contact-target candidates). Open: the residual ≈ 1 px, a fallback where the planes do not cross, and an RGB-free
   acceptance test.
2. **Thin-contour depth components.** 25 of the 45 step gaps still open after the G4 re-evaluation are dropped at the
   component stage: contour components < 5 px (`min_component_px`), 82 % `thin` (doublet fits), r above τ_high
   (fences, railings). Candidate: a component rule for thin contours (e.g. linking along the chain direction, or a
   size minimum counted on the chain rather than on distinct pixels), checked against the fence clutter policy.
3. **Items 6 / 7 in the fold channel.** Textured-surface folds (local σ_geo: 097 079 fold, 124 / 125 088 creases) and
   narrow faces narrower than the k₁ windows (15 067 pickets, 37 070 porch nosing). Covered by the Laplacian in
   geom_v1; a fold-channel fix would let a later version drop the Laplacian's texture and band excess there.

## 8. Runtime (CPU, measured in F4)

| step | time per image |
|---|---|
| trilateral ρ_f | 0.3–0.4 s |
| dense fit (side windows + 2 weighted passes) and normals | about 4.5–6 s |
| fold channel after the context (candidates, null sample, two-sided fits, veto, rasterization) | 0.6–0.9 s |
| Laplacian channel | about 0.5 s |
| depth channel after the context (field, candidates, null sample, jumps, gates, rasterization; no dense fit) | 0.6–0.8 s (TASK_DEPTH_CHANNEL D5) |

The task's target (fold channel < 5 s on the CPU) is missed by the context. Inference is expected to run on CUDA; the
torch backend follows after F5, checked against this numpy implementation.
