# Geometric-edge channels: architecture (draft, F0)

Status: **design draft, no code yet**; project root `geometry/dedicated/` (`TASK_FOLD_CHANNEL.md`, milestone F0). The math is fixed by `TASK_FOLD_CHANNEL.md` and is only referenced here (N1, N2, F2.1–F2.8). This document fixes the module map, the data flow, the interfaces, what moves out of `detector.py`, and how a future depth channel plugs in. It is finalized in F5.

Conventions, as in `docs/pipeline_tri_veto4.md`:

- All geometry runs at **network resolution** (504×378 or 378×504), in the **upright frame**, in **float64**, in pure numpy/scipy.
- Pixel `x = (u, v)`: u is the column, v the row (downward). Camera frame is OpenCV (X right, Y down, Z forward).
- `ρ = 1/D` on `Ω = ¬S`, `ρ = 0` on the sky `S`. `ρ_f` is the trilateral-filtered disparity (`detector.py: trilateral`).

## 1. Module map

New subpackage `illum_baseline/geom/`. Nothing in it imports torch, the networks or the RGB image.

| module | contents | depends on |
|---|---|---|
| `camera.py` | `Intrinsics(f_x, f_y, c_x, c_y, W, H, sigma_f)`; `Intrinsics.from_stage_a(meta_a)`; `rays(K) -> (H, W, 3)`; `Intrinsics.scaled(kappa)` (f → κf, for the focal diagnostics) | – |
| `planefit.py` | `dense_fit(rho_f, ground, rho_floor, params) -> DenseFit` (N1); `window_fit(values, xy, mask) -> WindowFit` (generic masked OLS + covariance, F2.3) | `detector.py` helpers |
| `normals.py` | `plane_vector(alpha, beta, u, v, K) -> mu`; `normals(alpha, beta, cov_beta, u, v, K) -> Normals` (n̂, Σ_n, σ_θ, ‖μ‖); `plane_distance(alpha, mu)` (d₀ = 1/(α‖μ‖), for tests) | `camera.py` |
| `context.py` | `GeometryContext`: lazily cached, read-only per-image fields shared by all channels | all of the above |
| `twosided.py` | **shared by fold and the future depth channel.** `two_sided_fit(ctx, x0, e1, k, m, w_t, ...) -> TwoSidedResult`, vectorized over N; `planarity_scale(results) -> eps_hat` | `planefit`, `normals`, `context` |
| `candidates.py` | `normal_structure_tensor(ctx, sigma_d, sigma_T) -> (r_n, e1, valid)`; `fold_candidates(ctx, params) -> Candidates` (F2.1); `rounding_width(ctx, x0, e1, half_len) -> (width, ok)` (F2.2); stub `depth_candidates(ctx, params)` raising `NotImplementedError` | `context` |
| `fold.py` | `FoldChannel`: F2.2–F2.8 on top of `candidates` + `twosided` | `candidates`, `twosided`, `normals` |
| `laplacian.py` | `LaplacianChannel`: adapter around `detector.compute_weights`, which stays unchanged | `detector.py` |
| `channels.py` | `ChannelResult`, `Channel` protocol, `CHANNELS` registry, `combine(results, mode="min")` | – |

Outside the subpackage:

| file | change | old behaviour preserved by |
|---|---|---|
| `detector.py` | two pure code-motion refactors (section 4) | bit-for-bit tests (section 4.3) |
| `config.py` | `GeometryCfg` (+ nested `DenseFitCfg`, `FoldCfg`); validation; `--set` for nested keys (`geometry.fold.p_fa=…`) | `enabled: false` built-in default |
| `stage_b.py` | `process_one` branches on `cfg.geometry.enabled`; the `false` branch is the current code, untouched | repro tests |
| `manifest.py` | tolerate missing Laplacian scalars (fold-only runs); add channel scalar columns when present | only reached in geometry mode |
| `viz.py`, new `viz_fold.py`, `cli.py` (`diag-fold`) | F4 | – |
| `configs/candidate_fold_only.yaml`, `configs/candidate_lap_fold.yaml` | new (F3) | `configs/default.yaml` unchanged |

## 2. Data flow (one image, `geometry.enabled: true`)

```
Stage A cache (upright):  D, sky_prob          Stage A JSON: geocalib fx, fy, focal_uncertainty; network s_x, s_y; focal.mode
        │                                                  │
        │                                     Intrinsics.from_stage_a  (f_x = fx·s_x, f_y = fy·s_y, c = ((W−1)/2, (H−1)/2))
        ▼                                                  ▼
GeometryContext(D, sky_prob, K, det_params, dense_fit_params)        ← built once, fields computed on first access
  S, Ω, ρ, ρ_floor        (detector.prepare_disparity: the exact B1/B2/ρ_floor code of compute_weights)
  ρ_f                     (detector.trilateral with detector.tri_sigma_s/r/g, regardless of detector.prefilter)
  g = ∇ρ_f                (detector.gaussian_gradient, σ_g = dense_fit.sigma_g)
  dense fit (N1)          α, β, Cov(α,β), M0, N_eff, s², cond, fit_valid
  normals (N2)            n̂, Σ_n, σ_θ, ‖μ‖, normal_valid
  L_over_rho_f            (un-vetoed L/ρ on ρ_f; only for the convex/concave record)
        │
        ├── LaplacianChannel.run(ctx, asdict(detector_params(cfg)))   → compute_weights(ctx.D, ctx.sky_prob, Params)
        ├── FoldChannel.run(ctx, cfg.geometry.fold)                    → candidates → two-sided fits → gate → veto → w_fold
        │        (channels run in the order of geometry.channels; each only reads ctx)
        ▼
combine([...], "min")  →  w_net (upright, network res)
        ▼
existing path: deliverable_w (upsample upright, clip, to_stored), net_to_stored, savez, JSON, viz
```

With `geometry.enabled: false`, `process_one` runs the current code path, byte for byte.

## 3. Interfaces

### 3.1 Channels

```python
@dataclass
class ChannelResult:
    name: str
    w_net: np.ndarray                  # (H, W) in [w_min, 1], network res, upright
    strength: np.ndarray | None        # dimensionless per-pixel strength (documented per channel)
    candidates: pd.DataFrame | None    # one row per candidate, all of them, with accept/reject reasons
    arrays: dict[str, np.ndarray]      # extra network-res upright arrays; saved as "<name>_<key>"
    scalars: dict[str, float]          # e.g. lambda_hat, eps_hat, stage counts
    w_min: float                       # the channel's floor (used for the deliverable clip, section 3.4)

class Channel(Protocol):
    name: str
    def run(self, ctx: GeometryContext, params: dict) -> ChannelResult: ...

CHANNELS: dict[str, type[Channel]] = {"laplacian": LaplacianChannel, "fold": FoldChannel}   # future: "depth"

def combine(results: list[ChannelResult], mode: str = "min") -> np.ndarray:
    # mode != "min" -> ValueError("combine mode 'x' is not implemented; only 'min'")
    # elementwise min; for one channel, returns that channel's w_net bits unchanged
```

Strengths:

| channel | `strength` | meaning |
|---|---|---|
| `fold` | `J = sin(min(Δθ(k₁), π/2))`, max over accepted candidates whose band covers the pixel, 0 elsewhere | sharp worst-case shading jump in units of Φ = ∫L dω (F2.7) |
| `laplacian` | *lean:* `q_pool / ε` | so that `w = max(w_min, 1/(1 + strength))`; dimensionless (q is px⁻², ε is px⁻²) |
| `depth` (future) | `ΔZ/Z_near` (metric ΔZ as an extra array) | relative depth jump |

### 3.2 GeometryContext

```python
class GeometryContext:
    def __init__(self, D, sky_prob, K: Intrinsics, det: detector.Params, fit: DenseFitParams): ...
    # inputs (read-only): D, sky_prob, K, det, fit, shape, u, v (pixel-coordinate grids)
    # cached_property, all float64 / bool, all set non-writeable after computation:
    S, omega, rho, rho_floor, n_bad_depth
    rho_f
    g                    # (H, W, 2) Gaussian-derivative gradient of rho_f at sigma_g
    dense: DenseFit      # alpha (H,W), beta (H,W,2), cov (H,W,3,3), m0, n_eff, s2, cond, valid
    n_hat                # (H, W, 3), NaN where invalid
    Sigma_n              # (H, W, 3, 3)
    sigma_theta          # (H, W) rad
    mu_norm              # (H, W)
    normal_valid         # (H, W) bool
    L_over_rho_f         # (H, W), record-only use
    def tangent_plane_weights(self, rho_in, sigma_s, sigma_r, sigma_g): ...  # the shared per-offset weight generator
```

- Channels only read the context. Arrays are returned with `flags.writeable = False`, so an accidental write raises.
- *Flag (lean):* the task lists "the trilateral weights" among the cached fields. Caching them per offset is 253 × H × W × 8 B ≈ 385 MB per image. *Lean:* do not cache them; the context exposes the shared weight generator (section 4.1) and caches only what the fit produces (moments, α, β, Cov).

### 3.3 Dense fit (N1), implementation notes

- One pass over the 253 offsets of the disk `‖d‖ ≤ R_n = 9`, reusing the trilateral weight generator with `(ρ_f, σ_n, σ_r, σ_g)`.
- **Centred regressand** (an implementation choice, algebraically identical to N1): regress `e(x,d) = ρ_f(x+d) − ρ_f(x) − g(x)·d` on `(1, d)` and set `α = ρ_f(x) + α_e`, `β = g(x) + β_e`. The WLS solution is the same as regressing `ρ_f(x+d)` (the transformation is an affine change of the regressand that lies in the column space of A). Why: the residual sum Σ W r² can then be accumulated in one pass as `ΣWe² − 2θ_eᵀb_e + θ_eᵀMθ_e` without catastrophic cancellation, because e is of the size of the residuals, not of ρ. On an exact plane e ≡ 0 wherever g is exact, so α, β are exact.
- Accumulated per pixel: `M = AᵀWA` (6 unique), `b = AᵀWe` (3), `ΣWe²`, `Q = AᵀW²A` (6 unique, for the sandwich), `ΣW²` (in Q). Then a batched 3×3 solve, `cond` from batched SVD (in px units, d up to 9 px), and the sandwich covariance with `s²` and `N_eff` as in N1.
- Validity: sky, `dilate(S, R_n)`, `R_n` from the border, `M0 < min_weight_sum`, `cond > cond_max`.

### 3.4 Two-sided fit (shared with the depth channel)

```python
def two_sided_fit(ctx, x0, e1, k, m, w_t, *, eps_hat=None, planarity_factor=3.0, m_min=4,
                  max_excluded=0.2, fallback=True) -> TwoSidedResult
    # x0 (N, 2) float, e1 (N, 2) unit, k (N,) int; m, w_t int
    # eps_hat None: no planarity check (used for the eps_hat calibration pass itself)

@dataclass
class TwoSidedResult:       # every field (N,) or (N, ...), side suffix _p (+e1) / _m (−e1)
    alpha_p, alpha_m        # fitted disparity at the centroid
    beta_p, beta_m          # (N, 2) disparity per px
    c_p, c_m                # (N, 2) sample centroids (u, v)
    cov_p, cov_m            # (N, 3, 3)  s²(AᵀA)⁻¹
    rms_rel_p, rms_rel_m    # rms(residual) / alpha
    n_p, n_m                # samples used;  excl_p, excl_m: excluded fraction
    m_used_p, m_used_m      # 8, 4, or 0 (invalid)
    valid_p, valid_m        # planar window fit available
    fallback_p, fallback_m  # dense n̂ at the centroid used instead
    nhat_p, nhat_m          # (N, 3);  Sigma_n_p, Sigma_n_m (N, 3, 3);  sigma_theta_p, sigma_theta_m
    d_rho, d_rho_se         # Δρ* and its standard error (NaN if either side fell back)
    rho_p_x0, rho_m_x0      # each side's plane extrapolated to x0
    dZ_rel                  # |Δρ*| / min(ρ₊(x0), ρ₋(x0))
    dtheta                  # atan2(|n₊ × n₋|, n₊·n₋)
    T                       # δᵀ(Σ₊ + Σ₋)⁻¹δ in the tangent basis at n̄
    valid                   # both sides usable (planar or fallback)
```

- `two_sided_fit` knows nothing about folds: no thresholds, no λ̂, no J. Gates, vetoes and weights belong to the channels.
- `depth_jump` (F2.3) is a fold-channel column computed from `dZ_rel`, `d_rho`, `d_rho_se`; the depth channel will compute its own decision from the same fields.

### 3.5 Stage B integration and saving

Saved in `<stem>.npz` when geometry is enabled (stored frame; network res unless noted):

| key | dtype | notes |
|---|---|---|
| `w` | float16 | combined, original resolution, clipped to `[min over channels of w_min, 1]` |
| `w_net` | float16 | combined |
| `<name>_w_net`, `<name>_strength` | float16 | per channel |
| `<name>_<key>` | float16 / bool | per-channel `arrays` (fold: e.g. `fold_r_n`) |
| `n_hat` | float16 (H, W, 3) | *Flag:* the array layout is transformed to the stored frame, but the **vector components stay in the upright camera frame**. Every image in this dataset has orientation 1, so this only matters for other data; the JSON frame note says so. |
| `sigma_theta` | float16 | rad |
| `normal_valid` | bool | |
| `D`, `sky_mask` | float32, bool | as now |
| `q`, `L_over_rho` | float16 | *lean:* also written (unprefixed) when `laplacian` is among the channels, so the existing `viz.render` sheet works unchanged |

- Candidate tables: `<stem>_<name>_candidates.parquet` in the run directory. *Lean:* coordinates in the **upright network** frame (they are analysis data, the viz is upright), noted in the JSON. `image_stems()` and the manifest glob only `*.npz`/`*.json`, so these files do not interfere.
- JSON: the existing blocks, plus `channels: {<name>: scalars}` and `combined: {frac_w_lt_*}`; `params.geometry` is recorded **only when enabled**, so old records stay identical. `STAGE_B_VERSION` is not bumped (it would change baseline JSONs); a `geometry_version` field is added only when enabled.
- `_check_run_config` adds the `geometry` section to its key, so runs with different channel settings cannot mix in one directory. Old snapshots are completed with the built-in default (`enabled: false`), so their key is unchanged.
- *Lean:* `detector` in the JSON is filled from the Laplacian channel's scalars when it runs; in fold-only runs it holds only `sky_frac`, `rho_floor`, `n_bad_depth` and the combined fractions, and the manifest writes NaN for the missing columns.

### 3.6 Config

As specified in `TASK_FOLD_CHANNEL.md`, as dataclasses whose built-in defaults are the YAML values listed there, with `enabled: false` and `channels: [laplacian]`. Validation: every channel name in `CHANNELS`, `combine == "min"`, positive scales, `0 < p_fa < 1`, `lambda_clamp` ordered, `m_min ≤ m`, `0 < C_min ≤ 1`.

- *Note:* the JSON already has a `geometry` block (s_x, s_y, orientation). The new config section is also called `geometry` (settled). They live in different places (`params.geometry` vs top-level `geometry`), so there is no key clash, but the doc will say which is which.

## 4. What moves out of `detector.py`, and why it stays bit-for-bit

### 4.1 Refactors (pure code motion)

1. **`tangent_plane_weights(rho, ground, rho_floor, sigma_s, sigma_r, sigma_g)`**, a generator yielding `(dx, dy, nb, e, w)` per offset of the disk `‖d‖ ≤ ⌈3σ_s⌉`, in the current loop order (dy outer, dx inner). It contains, verbatim, the current body of `trilateral` up to and including `w = ws * np.exp(-0.5 * (e * inv) ** 2) * G[...]` (gradient, padding, `inv`, `ws`).
   - `trilateral` becomes: `for _, _, _, e, w in tangent_plane_weights(...): num += w * e; wsum += w`, then the same output lines.
   - The dense fit (N1) calls the same generator on `ρ_f` with `(σ_n, σ_r, σ_g)` from `geometry.dense_fit`, so the fit's weights are by construction the trilateral weights of N1.
2. **`prepare_disparity(D, sky_prob, p) -> Disparity(S, ground, bad, rho, rho_floor, sky_frac)`**: steps B1, B2 and the `ρ_floor` line of `compute_weights`, moved verbatim, including the two `ValueError`s. `compute_weights` calls it. The context calls it too, so `ctx.rho`, `ctx.S`, `ctx.rho_floor` are the detector's values.

Used as is, without changes: `kernels`, `_correlate_valid`, `smooth_and_laplacian`, `gaussian_gradient`, `disk`, `erode`, `dilate`, `sky_mask`, `compute_weights` (apart from calling `prepare_disparity`).

### 4.2 The bit-for-bit argument

- Each refactor moves statements without changing any expression, operand order, dtype, or loop order. numpy elementwise ufuncs and `math.exp` are deterministic for identical inputs on the same machine, and the only reductions in the moved code are the `num += w*e` / `wsum += w` accumulations, which stay in the same order over offsets. Hence every intermediate, and `w`, is bit-identical.
- Generators add no arithmetic; they only change where the loop body is written.
- `ctx.rho_f` is produced by the same `trilateral` call with the same arguments as inside `compute_weights` (`ρ`, `ground`, `ρ_floor`, `tri_sigma_*`), so it equals the Laplacian channel's `rho_filtered` bit for bit (tested).
- `LaplacianChannel` calls `compute_weights` with the same `Params` as the disabled path (`detector_params(cfg)`), and `combine` over one channel returns its `w_net` unchanged; the downstream path is shared code. So `geometry.enabled: true, channels: [laplacian]` gives a bit-identical `w`.
- *Consequence (lean):* with `channels: [laplacian, fold]`, `trilateral` runs twice (once in `compute_weights`, once in the context), about 0.4 s. Sharing it would need a `rho_f=` argument in `compute_weights`; the task says the adapter calls the detector unchanged, so the duplicate is accepted.

### 4.3 Tests that enforce it (F1/F3)

| test | reference |
|---|---|
| `test_baseline_repro.py` (existing) | `baseline_output/stage_b/baseline_v0` |
| new: same test parametrized over `tri_veto4` | `baseline_output/stage_b/tri_veto4` and `baseline_output_batch2/stage_b/tri_veto4` (section 6, item 3) |
| new: `geometry.enabled: true, channels: [laplacian]` vs `tri_veto4` | `w`, `w_net` bit-identical; `q`, `L_over_rho`, `D`, `sky_mask` too |
| new: `trilateral` vs a frozen copy of the pre-refactor implementation inside the test file | bitwise, synthetic floor-and-box scene with sky, and one real image |
| new: `ctx.rho_f == compute_weights(...)["rho_filtered"]` | bitwise |

## 5. Adding a depth channel later (outline; finalized in F5)

The depth channel (Canny-like on the detrended disparity, strength ΔZ/Z_near) plugs in without touching `fold.py`:

1. `candidates.py`: implement the existing stub `depth_candidates(ctx, params) -> Candidates` (same container as fold candidates: integer pixel, subpixel `x0`, unit `e1`, a rate, and a `width`).
2. `depth.py` (new): `DepthChannel.run(ctx, params)` calls `two_sided_fit(ctx, x0, e1, k, m, w_t, eps_hat=…)` and reads only `d_rho`, `d_rho_se`, `rho_p_x0`, `rho_m_x0`, `dZ_rel`, `valid_*`, `fallback_*`, `rms_rel_*`; metric `ΔZ` comes from `ctx.K` (f_n) and `D`. Its own gate (e.g. `|Δρ*|/SE` with its own null calibration) and weight live in `depth.py`.
3. `planarity_scale` is a `twosided.py` function; each channel calibrates ε̂ on its own null sample, so the channels never share mutable state.
4. Register `"depth": DepthChannel` in `CHANNELS`, add `geometry.depth:` to the config with defaults, and list it in `geometry.channels`. `combine` already takes any number of channels.
5. Saving, candidate tables and JSON scalars follow from `ChannelResult` automatically (`depth_w_net`, `depth_strength`, `<stem>_depth_candidates.parquet`).

**Planned depth-channel defaults (F2 decisions, 2026-10-31; not implemented):**

| parameter | planned value | why |
|---|---|---|
| window skip k | ≥ 11 (= R_s + 2) | windows clear the trilateral bias zone (9 px) and the bilinear/subpixel margin; in F2 arm B (k_min = 11) `depth_jump` recall on single synthetic steps was 0.995, vs 0.04 at k_min = 6 |
| noise inflation λ_n (additive null) | 121.2 | Monte-Carlo E[T]/2 on noisy synthetic planes at k = 11 (`illum_baseline/geom/data/lambda_n.json`); for k = 6 it is 93.2 |
| Δρ* | smallest plane gap over s ∈ [−k, k], with `planes_cross` | `twosided.py`; zero at fold-like plane intersections |

The fold channel keeps k_min = 6 (arm A); `two_sided_fit` takes k per call, so the two channels choose independently.

**Depth-channel test plan (to implement with the channel):** strict silhouette recall: `depth_jump` (or the depth
channel's own decision) set on ≥ 90% of valid candidates at synthetic occlusion silhouettes (box scene and single
steps), with k ≥ 11. The fold channel (k_min = 6) only reports its silhouette recall (0.75 on the F2 box scene):
its k₁ windows next to a step often fail the contamination check, and missed candidates sit a median 4.9 px from the
true edge.

What makes this work: the context owns every shared field (ρ_f, the dense fit, n̂), `twosided.py` owns the shared fit and is decision-free, and channels communicate only through `ChannelResult`.

## 6. Decisions on borderline choices (all leans agreed 2026-10-31)

1. **Project location.** The fold channel, and later the depth channel, live in the self-contained project `geometry/dedicated/`: code, configs, tests, docs, scripts, data, outputs and environment. It started as a copy of `geometry/baseline/` (2026-10-31), which stays untouched as the frozen reference implementation. Code goes in `dedicated/illum_baseline/geom/`, tests in `dedicated/tests/`, this doc in `dedicated/docs/`.
2. **Running.** From `geometry/dedicated/`, as `env -u PYTHONPATH uv run ...`, with its own `.venv` built from the copied `uv.lock`. The only change to `pyproject.toml`/`uv.lock` is the `depth-anything-3` path, `../../DA3/Depth-Anything-3` (DA3 lives at `codebase/DA3/`). Configs resolve `../data` and `../baseline_output` relative to `configs/`, i.e. `dedicated/data` and `dedicated/baseline_output`.
3. **Bit-for-bit references.** Copied from `baseline/` with their Stage A caches: `baseline_output/stage_b/baseline_v0` and `baseline_output/stage_b/tri_veto4` (the 10 `taiwei_cui` images), and `baseline_output_batch2/stage_b/tri_veto4` (the 10 batch2 images). In the copied `config_resolved.yaml` and `qa_report.md` files only the absolute `data_dir`/`out_dir` paths were rewritten to `dedicated/`; arrays and JSONs are byte copies.
4. **Trilateral weights in the context.** Not cached (385 MB); the generator is shared instead (section 3.2).
5. **Name clash `w_fold`.** The task uses `w_fold` both for the rounding width (F2.2) and for the weight (F2.8). *Lean:* call the width `width` in code and tables (`fold_width` in docs), keep `w_fold` for the weight.
6. **Rounding width includes the estimator's own smearing.** F2.2 measures the transition in the *dense* n̂, which is fitted over R_n = 9 px; for weak folds the range weights cut the far side late, so the measured width is DA3's rounding plus the fit's transition. This only moves the k₁ windows outward (k₁ ≤ 8 with the clamp). *Lean:* accept; F2 reports the measured width on ideal synthetic folds of 30–120°.
7. **Rounding-width profile details.** Integer steps s = −12…12, 10%/90% crossings by linear interpolation on the running maximum of the normalized θ(s) (robust to a non-monotone profile). If any profile sample is invalid or θ(12) is below 1e-6 rad, the width defaults to the clamp maximum 12 and the row is flagged `profile_ok = False`.
8. **Bilinear sample exclusion (F2.3).** A sample is excluded if any of its four bilinear corners is sky, has an invalid normal, or lies outside the image. Keeps exactness on planes.
9. **k for the null sample (F2.5).** Null pixels have no fold, so F2.2 is meaningless there. *Lean:* `k₁_null = k_min` (6) for every null pixel. T is normalized by its covariance, so its null distribution depends on k only through noise correlation, which barely changes between k = 6 and 8.
10. **Population for ε̂ (F2.3).** *Lean:* ε̂ = median of `rms_rel` over both sides of the valid null-grid windows at m = 8 (no planarity check in that pass). The planarity check and fallback are then applied identically to null and candidate windows, so `T_null` and candidate T see the same pipeline.
11. **Fallback vs the veto (F2.6).** At k₂ a narrow face's window crosses the next fold; the dense-normal fallback would then return the *next* face's normal, making Δθ(k₂) large and C small, so a genuine fold would be vetoed. *Lean:* fallback is allowed at k₁ (flag `fallback`, as the task says), but a k₂ side that would need fallback counts as an invalid k₂ window, i.e. `no_veto` (accepted, counted).
12. **Fallback normal sampling.** *Lean:* nearest pixel to the window centroid (n̂ and Σ_n from the dense fit), not bilinear (keeps n̂ unit and Σ_n consistent). If that pixel's normal is invalid, the side is invalid and the candidate is rejected as `invalid_window`.
13. **Structure tensor near invalid normals (F2.1).** *Lean:* Gaussian derivatives (σ_d = 1, support 4 px) only where the whole support has valid normals (valid mask eroded by 4), then normalized convolution over valid pixels for the σ_T = 1.5 integration.
14. **Sign of e₁.** Canonical orientation, angle in (−90°, 90°] (u-component > 0, or u = 0 and v > 0). Side "+" is +e₁. Recorded as `e1_u`, `e1_v` in the table.
15. **Folds at silhouettes.** n̂ also jumps at occlusions between surfaces of different orientation, so occlusion edges become fold candidates and, when accepted, are down-weighted by `w_fold`. This is consistent with the settled design (`depth_jump` is a flag only) and with what the Laplacian already does there. Reported per image in F4 (`depth_jump` counts among accepted candidates).
16. **Deliverable clip with several channels.** `deliverable_w` clips at one `w_min`. *Lean:* clip at the minimum of the channels' `w_min` (both 0.05 by default; equals `detector.w_min` for Laplacian-only, so the repro path is unchanged).
17. **`--set` for nested keys.** `apply_sets` handles `section.key` only. *Lean:* extend it to `geometry.fold.p_fa=…`, and the CLI's automatic detector/sky flags stay as they are.
18. **Covariance of ĝ.** Following N2, Σ_ĝ = Cov(β)/α², i.e. the α uncertainty is neglected. With centroid-centred window fits α and β are uncorrelated and the α term is about 1e-2 of the β term for these windows; the dense fit's α–β covariance is likewise dropped. Noted in the doc, not modelled.

## 7. Runtime budget (estimate, checked in F4)

| step | cost |
|---|---|
| trilateral (ρ_f) | ≈ 0.4 s (measured for the whole detector, `pipeline_tri_veto4.md` §2.6) |
| dense fit, 253 offsets × ~20 array ops, batched 3×3 solve/SVD | ≈ 1–1.5 s |
| normals, structure tensor, NMS | < 0.2 s |
| two-sided fits: (candidates × {k₁, k₂, m/2 retries} + ≈ 12k null pixels) × 2 sides × 81 bilinear samples | < 1 s |
| rasterization | negligible |

Target from the task: fold channel < 5 s per image on CPU.
