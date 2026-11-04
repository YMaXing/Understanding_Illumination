# Task: surface-normal edge ("fold") channel, built as a pluggable geometric-edge channel

## How to work

- Context: `BASELINE_SPEC.md`, `docs/pipeline_tri_veto4.md` (current default `tri_veto4`), and `docs/viz_guide.md`. Read the code they reference before designing anything.
- Run everything from `codebase/geometry/` as `env -u PYTHONPATH uv run ...`.
- Work milestone by milestone (F0–F5). **Stop after each milestone**, report tersely (numbers, tables, figure paths), and wait for my go-ahead.
- Decisions in this file are settled; apply them without asking. Flag any other borderline choice **with your lean**. Do not resolve it silently.
- Every new behaviour sits behind config options whose built-in defaults are the old behaviour.
  - `baseline_v0` and `tri_veto4` must reproduce **bit-for-bit**. Extend the repro tests.
- Geometry code is pure numpy/scipy, float64, at network resolution, in the upright frame. No new DA3 passes.
- If a test fails for a principled reason, report it. Do not loosen the assertion silently.

## Goal and scope

1. Build a **fold channel**: surface normals from disparity, then fold detection with a dimensionless fold angle, then a per-pixel weight `w_fold`.
2. Combine it with the existing Laplacian channel: `w = min(w_lap, w_fold)`.
3. Visualization and docs.
4. **Architecture:** a future dedicated depth-edge channel (Canny-like, ΔZ from Δρ*) must plug in **without touching the fold channel**. The depth and fold channels will share one two-sided plane-fit module. That module is built here.

---

## Math (settled)

### Symbols and conventions

- **Pixel:** `x = (u, v)`; u is the column, v is the row (downward). Network resolution (504×378 or 378×504), upright frame.
- **Camera frame (OpenCV):** X right, Y down, Z forward.
- **Intrinsics at network resolution:**
  - `f_x = fx·s_x` and `f_y = fy·s_y`, from the Stage A JSON (active focal).
  - Principal point `c = ((W−1)/2, (H−1)/2)`. GeoCalib assumes the centre.
- **Ray:** `r(u,v) = K⁻¹(u,v,1)ᵀ = (x̄, ȳ, 1)`, with `x̄ = (u−c_x)/f_x` and `ȳ = (v−c_y)/f_y`. The 3D point is `X = Z·r`.
- **Disparities:**
  - `ρ = 1/D`, with D the canonical DA3 depth. Its global scale is irrelevant.
  - `ρ_f` = the trilateral-filtered disparity from `detector.py: trilateral` (σ_s = 3, σ_r = 1%, σ_g = 3). **Normals are always computed from ρ_f, never from raw ρ.**
  - Why: the 7-px DA3 ripple at a = 0.2% tilts normals by about 33° when computed from raw ρ, about 1° with a σ = 3 px gradient, and less than 0.1° from ρ_f plus the fit below.
- **Masks:** `S` = sky mask, `Ω = ¬S`.

### N1. Dense edge-aware plane fit

At each pixel x, over offsets `d` with `‖d‖ ≤ R_n = ⌈3σ_n⌉` (σ_n = 3 px; 253 offsets), solve

```
(α, β) = argmin Σ_d W(x,d) · [ρ_f(x+d) − α − β·d]²
```

**Weights.** `W(x,d)` = spatial Gaussian(σ_n) × range Gaussian on `e/(σ_r·max(ρ_f(x), ρ_floor))` × `1[x+d ∉ S]`.
- `e = ρ_f(x+d) − ρ_f(x) − g(x)·d`, where `g` is the Gaussian-derivative gradient of `ρ_f` at σ_g = 3.
- Reuse `gaussian_gradient` and the trilateral weight code. Refactor it into shared helpers if needed, without changing `tri_veto4` outputs.

**Meaning of the outputs.**
- `α` is the fitted disparity at x.
- `β = (ρ_u, ρ_v)` is the edge-aware gradient, in disparity per px.

**Solving.** Use the 3×3 weighted normal equations, with the moments `M0 = ΣW`, `M1 = ΣW·d`, `M2 = ΣW·ddᵀ`, `b0 = ΣW·ρ_f`, `b1 = ΣW·d·ρ_f`. Vectorize over shifted arrays.

**Validity.** A pixel is invalid if it is in the sky, within `R_n` of the sky or the image border, has `M0 < min_weight_sum`, or has a normal-equation condition number above `cond_max`.

**Covariance.** Use the sandwich form, because the weights are not inverse variances:

```
Cov(α, β) = s² · (AᵀWA)⁻¹ (AᵀW²A) (AᵀWA)⁻¹
s² = (Σ W r²) / (Σ W) · N_eff / (N_eff − 3)
N_eff = (ΣW)² / ΣW²
```

Here `A` is the design matrix with rows `(1, dᵀ)`, `W` is the diagonal of the weights, and `r` are the fit residuals.

### N2. Normals

- **Relative gradient:** `ĝ = β/α` (per px).
- **Plane vector, relative form:**

  ```
  μ = ( f_x·ĝ_u,  f_y·ĝ_v,  1 − (u−c_x)·ĝ_u − (v−c_y)·ĝ_v )
  n̂ = −μ / ‖μ‖
  ```

  - `n̂` is the unit normal pointing toward the camera (`n̂·r < 0`).
  - Derivation: a plane `n·X = d₀` maps to `ρ = m·r` with `m = n/d₀`. Matching `ρ(x+d) ≈ α + β·d` gives `m = α·μ`.
- **Properties to test:**
  - `μ·r(x) = 1`, so `‖μ‖ = 1/(|r|·cos α_inc)`, where α_inc is the angle between n̂ and the reversed viewing ray. It is large on grazing surfaces.
  - A global depth scale cancels.
  - **The plane vector is position-independent on a plane.**
- **Checks:** a camera-facing plane gives `n̂ = (0,0,−1)`. A floor `ρ = (v−c_y)/(f·h)` gives `n̂ = (0,−1,0)` and `d₀ = 1/(α·‖μ‖) = h`.
- **Angular covariance:**

  ```
  J  = [[f_x, 0], [0, f_y], [−(u−c_x), −(v−c_y)]]      (3×2; δμ = J·δĝ)
  P  = I − n̂n̂ᵀ
  Σ_n = P · J Σ_ĝ Jᵀ · P / ‖μ‖²,   with Σ_ĝ = Cov(β) / α²
  σ_θ = sqrt(trace Σ_n)   (rad)
  ```

- **Focal sensitivity:** `f → κf` scales `μ_x` and `μ_y` by κ. Near the principal point this gives `tan θ' = κ·tan θ`.
  - **Intrinsics errors cannot create folds:** both sides of one plane return the same `m` for any K.
  - So `σ_f` enters only the *magnitude* diagnostics (F2.7), never the gate.

### F. Fold measure (per candidate)

**F2.1 Candidates.**
- Form the structure tensor of n̂: `T_s = Σ_c G_{σ_T} ∗ (∇n_c ∇n_cᵀ)`, summed over the 3 components c of n̂.
  - `∇n_c` uses Gaussian derivatives at `cand_sigma_d` = 1 px, and the integration scale is `cand_sigma_T` = 1.5 px.
  - Valid normals only.
- Take its largest eigenvalue λ₁ and the eigenvector e₁ (unit, across the fold). The **rotation rate** is `r_n = sqrt(λ₁)` (rad/px).
- Non-maximum suppression along e₁, with bilinear samples at `x ± e₁`.
- Keep `r_n > cand_rate_factor · median(r_n over valid pixels)` (3).
- Subpixel location `x₀` from a parabola along e₁.
- Side "+" is `+e₁`. Record the sign convention.

**F2.2 Rounding width.**
- Sample n̂ at `x₀ + s·e₁` for `s ∈ [−12, 12]`.
- Compute `θ(s) = angle(n̂(s), n̂(−12))` and normalize it by `θ(12)`.
- `w_fold` = the s-distance from 10% to 90%, clamped to [2, 12].
- Skip `k₁ = max(k_min, ⌈w_fold/2⌉ + k_margin)`, with `k_min = 6`. The value 6 comes from the trilateral bias near features (≤ 4.7e-3 relative within 6 px).

**F2.3 One-sided 2D windows** (shared module, see Architecture):

```
𝒲±(k) = { x₀ + s·e₁ + t·ê_t :  ±s ∈ [k, k+m],  |t| ≤ w_t }   with ê_t = e₁^⊥,  m = 8,  w_t = 4
```

- **Sampling:** sample `ρ_f` on the integer (s, t) grid by bilinear interpolation, which is exact for affine functions.
- **Excluded samples:** those in the sky, with invalid normals, or outside the image. If more than 20% of a window is excluded, the window is invalid.
- **Fit:** ordinary least squares, `ρ_f = α± + β±·(x − c±)`, with `c±` the sample centroid. `Cov(α±, β±) = s²(AᵀA)⁻¹`, with `s² = RSS/(N−3)`.
- **Planarity check:**
  - Compute `rms(residual)/α > planarity_factor · ε̂`. Here `ε̂` is the image's median relative residual over all valid windows.
  - If it fails, retry with `m/2`. If `m < m_min` (4), the window is invalid.
  - For an invalid window, fall back to dense n̂ at the window centroid, flagged as `fallback`.
- **Normals:** compute `n̂±` with N2 at `(u,v) = c±`, and `Σ_n±` with N2's covariance.
- **Jump in value** (recorded for the future depth channel):

  ```
  Δρ* = [α₊ + β₊·(x₀ − c₊)] − [α₋ + β₋·(x₀ − c₋)]
  ```

  Record its standard error, `ΔZ/Z_near = |Δρ*| / min(ρ₊(x₀), ρ₋(x₀))`, and the flag `depth_jump = (ΔZ/Z_near > depth_jump_rel) and (|Δρ*|/SE > 3)`. This is a flag only; it does not affect `w_fold`.

**F2.4 Fold angle.**

```
Δθ = atan2( ‖n̂₊ × n̂₋‖,  n̂₊·n̂₋ )   ∈ [0, π]
```

**F2.5 χ² gate, not a threshold on Δθ.** Under noise Δθ is Rayleigh-distributed and biased: its mean is about 1.77σ.
- `n̄ = normalize(n̂₊ + n̂₋)`, with a tangent basis `B = [b₁, b₂]` at n̄.
- `δ = Bᵀ(n̂₋ − n̂₊)` and `Σ± = Bᵀ Σ_n± B` (both 2×2).
- `T = δᵀ (Σ₊ + Σ₋)⁻¹ δ`, which is χ²₂-distributed under the null if the covariances are right.
- **Calibration (per image):**
  - Evaluate T at all valid pixels on a stride-`null_stride` grid (4), using each pixel's own e₁ and the k₁ windows.
  - `λ̂ = median(T_null) / (2 ln 2)`, clamped to `lambda_clamp`.
  - Most pixels are not folds, so the median is robust, as for ŝ.
- **Accept** if `T > λ̂ · 2·ln(1/p_fa)`, with `p_fa = 1e-3`, i.e. `T > 13.8·λ̂`.

**F2.6 Concentration veto against smooth curvature.**
- Recompute Δθ with `k₂ = veto_k_factor·k₁` (2×), keeping the same m and w_t.
- `C = Δθ(k₁) / Δθ(k₂)`.
  - A fold has C ≈ 1.
  - Constant curvature has C ≈ (2k₁+m)/(2k₂+m), which is 0.63 with the defaults.
- Accept if `C ≥ C_min` (0.85). Apply the veto only to candidates that passed the gate.
- *Flag:* if the k₂ windows are invalid (a narrow face), accept with the flag `no_veto` and report counts. *Lean:* accept.

**F2.7 Magnitude.**

```
J = sin( min(Δθ(k₁), π/2) )
```

- J is the sharp worst-case shading jump in units of the total incident radiance Φ = ∫L dω. It assumes Lambertian surfaces, is exact for convex folds, and covers direct light for concave folds.
- Record `c = 2·sin(Δθ/2)` as well.
- **Focal diagnostics:** recompute n̂± with f·(1 ∓ σ_f/f) to get `[J_lo, J_hi]`. Record them; they are not used in w.

**F2.8 Weight and rasterization.**
- Per candidate: `w_c = max(w_min, J₀ / (max(J − J_min, 0) + J₀))`, with `J_min = sin 5° = 0.0872` and `J₀ = 0.25`.
- Remove accepted candidate components smaller than `min_component_px` (5) in the 8-connected accepted map.
- `w_fold(x)` = the minimum of `w_c` over accepted candidates within radius `b_c = max(band_min, ⌈w_fold/2⌉)` of x. Elsewhere `w_fold = 1`.
- Implement this as min-filters over the few distinct `b_c` values.
- **Optional:** record convex vs concave from the sign of the un-vetoed `L_over_rho` at x₀. Record only.

---

## Architecture (settled)

### New subpackage `illum_baseline/geom/`

| module | contents |
|---|---|
| `camera.py` | `Intrinsics` (f_x, f_y, c_x, c_y at network res, from the Stage A JSON + s_x, s_y); `rays()` |
| `context.py` | `GeometryContext`: lazily cached shared fields computed once per image. Holds ρ, ρ_f, S, Ω, ρ_floor, g, the trilateral weights, the dense fit (α, β, Cov), n̂, Σ_n, σ_θ and the validity mask. Channels only *read* it. |
| `planefit.py` | Dense edge-aware fit (N1); generic window OLS fit with covariance and planarity check |
| `normals.py` | μ, n̂, Σ_n, σ_θ from (α, β, Cov, intrinsics, pixel coords); focal-perturbation helper |
| `twosided.py` | **Shared by the fold and future depth channels.** `two_sided_fit(ctx, x0, e1, k, m, w_t) -> TwoSidedResult`, vectorized over N candidates. Fields: α±, β±, c±, Cov±, residual RMS, flags; n̂±, Σ_n±; Δρ* with its SE and ΔZ/Z_near; Δθ; T. |
| `candidates.py` | Fold candidates (F2.1) and rounding width (F2.2). Leave a stub signature for future depth candidates (Canny on the detrended disparity gradient). |
| `fold.py` | `FoldChannel`: F2.2–F2.8 using `twosided.py` |
| `laplacian.py` | `LaplacianChannel`: an adapter that calls the existing `tri_veto4` detector, unchanged |
| `channels.py` | `Channel` protocol, `ChannelResult`, registry, combiner |

### Channel interface

```python
@dataclass
class ChannelResult:
    name: str
    w_net: np.ndarray                  # (H, W) in [w_min, 1], network res, upright
    strength: np.ndarray | None        # dimensionless per-pixel strength (fold: J at rasterized pixels, else 0)
    candidates: pd.DataFrame | None    # one row per candidate (all, with accept/reject reasons)
    arrays: dict[str, np.ndarray]      # extra arrays to save; keys get the "<name>_" prefix
    scalars: dict[str, float]          # e.g. lambda_hat, counts

class Channel(Protocol):
    name: str
    def run(self, ctx: GeometryContext, params: dict) -> ChannelResult: ...

CHANNELS = {"laplacian": LaplacianChannel, "fold": FoldChannel}   # future: "depth"
def combine(results: list[ChannelResult], mode: str = "min") -> np.ndarray: ...
```

- **Strengths are dimensionless and documented,** so a future fusion rule can act on strengths rather than weights. Fold uses J. The future depth channel will use ΔZ/Z_near, plus metric ΔZ as an extra.
- **Combiner:** `min` is the only mode now. Raise a clear error on any other mode.

### Stage B integration

- If `geometry.enabled: false` (the built-in default), the existing path runs untouched.
- If it is enabled:
  1. Build the context.
  2. Run `geometry.channels` in order.
  3. Combine.
  4. Use the existing upsampling, frames and saving.
- **Saved arrays:**
  - combined: `w`, `w_net`;
  - per channel: `<name>_w_net`, `<name>_strength`, plus its `arrays`;
  - context: `n_hat` (3×float16) and `sigma_theta` (float16), plus `normal_valid`.
- **Candidate tables:** `<stem>_<name>_candidates.parquet`.
- **Repro requirement:** `geometry.enabled: true` with `channels: [laplacian]` must give `w` bit-identical to `tri_veto4`. Test this.

### Config

```yaml
geometry:
  enabled: false
  channels: [laplacian]
  combine: min
  dense_fit: {sigma_n: 3.0, sigma_r: 0.01, sigma_g: 3.0, cond_max: 1.0e6, min_weight_sum: 5.0}
  fold:
    cand_sigma_d: 1.0
    cand_sigma_T: 1.5
    cand_rate_factor: 3.0
    profile_half_len: 12
    k_min: 6
    k_margin: 2
    m: 8
    m_min: 4
    w_t: 4
    planarity_factor: 3.0
    p_fa: 1.0e-3
    null_stride: 4
    lambda_clamp: [0.5, 50.0]
    veto_k_factor: 2.0
    C_min: 0.85
    J_min: 0.0872
    J0: 0.25
    w_min: 0.05
    band_min: 2
    min_component_px: 5
    depth_jump_rel: 0.02
```

Add `configs/candidate_fold_only.yaml` (`channels: [fold]`, run `fold_only`) and `configs/candidate_lap_fold.yaml` (`channels: [laplacian, fold]`, run `tri_veto4_fold`). **Do not change `configs/default.yaml`.**

---

## Milestones

### F0 — Design check (no code)

Read `detector.py` (trilateral, gaussian_gradient, compute_weights), `stage_b.py`, `outputs.py`, `viz.py` and `config.py`. Write `docs/geometry_channels.md` (draft):
- module map, data flow, interfaces;
- what is refactored from `detector.py`, with the bit-for-bit argument;
- how a future depth channel plugs in.

**STOP.**

### F1 — Context, dense fit, normals

Implement `camera.py`, `context.py`, `planefit.py` (dense part) and `normals.py`.

Tests (synthetic, noise-free unless stated):
1. **Planes:** random visible planes, using the K of the BASELINE_SPEC test 5. n̂ equals the ground truth to 1e-12 wherever the window lies on one plane.
2. **Checks:** a camera-facing plane gives (0,0,−1). A floor gives (0,−1,0), with `d₀ = h`.
3. **Depth scale:** `D → c·D` leaves n̂ unchanged.
4. **Focal:** with `K: f → κf` (κ ∈ {0.78, 1.22}), `tan θ' = κ tan θ` near the principal point.
5. **Covariance:** with i.i.d. relative noise on ρ_f, Monte-Carlo σ_θ is within 20% of the analytic σ_θ.
6. **Ripple:** a plane plus a 7-px relative ripple of 0.2%, passed through the trilateral filter, gives a median normal error below 0.1°.

**STOP.**

### F2 — Two-sided fits and the fold measure

Implement `twosided.py`, `candidates.py` and `fold.py`.

Synthetic scene: floor, box, and a cylinder (several radii), with exact surface IDs and normals. Tests:
1. **Exact fold angles:** folds at 30°, 60°, 90° and 120°, in several image orientations. Δθ is exact to 1e-9 (noise-free).
2. **Wrong K:** planes with K perturbed (κ ∈ {0.78, 1.22}) give Δθ = 0, and Δρ* = 0 on single planes.
3. **Veto:** cylinders are vetoed (C < C_min). Box folds pass (C ≥ 0.95).
4. **Noise null:** `median(T_null)/(2 ln 2)` is within 20% of the covariance inflation actually used.
5. **Flag:** `depth_jump` is set at box silhouettes and not at folds.
6. **Narrow faces:** the windows become invalid and the flags are set.

**STOP.**

### F3 — Channel framework and Stage B integration

Implement `channels.py`, `laplacian.py`, the Stage B wiring, saving and configs.

Tests:
- `baseline_v0` and `tri_veto4` remain bit-for-bit.
- `channels: [laplacian]` with geometry enabled is bit-identical to `tri_veto4`.
- `combine` is the elementwise min.
- The saved keys and the candidate parquet schema are as specified.

**STOP.**

### F4 — Visualization, runs, metrics

**Fold sheet** `stage_b/<run>/viz_fold/<stem>.png`, upright, a 3×3 grid plus a text box:
1. sRGB with accepted fold segments coloured by Δθ (deg), veto-rejected segments in gray, and `no_veto` in dashed style.
2. Normal map as RGB = (n̂+1)/2, with the convention noted in the title; invalid pixels gray.
3. σ_θ in degrees, log scale.
4. `r_n` (rad/px, log) with all candidates overlaid.
5. Δθ(k₁) at candidates (deg), on a dark background.
6. Histograms of `T/λ̂` (log–log): null sample vs candidates, with the χ²₂ density and the gate line.
7. Scatter of C vs Δθ for gate-passed candidates, with the C_min line.
8. `w_fold`.
9. Combined `w` over the sRGB (red alpha, as in panel 6 of the existing sheet).
10. Text box: λ̂, ε̂, candidate counts at each stage (`cand → valid → gate → veto → component`), parameters, f and f_unc, run name.

**Profile tool:** `cli diag-fold --stem S --u U --v V`. Plot:
- the dense normal-angle profile along e₁ through the nearest candidate, with `w_fold`, k₁ and k₂ marked;
- the windows drawn on the image;
- n̂± as arrows;
- Δθ(k₁), Δθ(k₂), C, T/λ̂ and J.

**Runs** on the 10 images: `fold_only` and `tri_veto4_fold`.

Report one table per run: share w<0.5, share w<0.9, Δ vs `tri_veto4`, λ̂, ε̂, median σ_θ, candidate counts by stage, median J of accepted candidates, and runtime per image.

**Spot checks (acceptance):**

| check | requirement |
|---|---|
| 079 lit/shaded box folds (rows 230/254), combined w | report; target ≤ 0.45 (baseline_v0 0.35–0.45, tri_veto4 0.50–0.59) |
| 083 bollard (u 175–205, v 260–330), `w_fold` | median ≥ 0.95 |
| 070 floor, combined share w<0.9 | ≤ tri_veto4 value + 0.01 |
| 079 grass, combined share w<0.5 | ≤ tri_veto4 value + 0.03 |
| fold channel runtime | < 5 s/image on CPU |

**Evaluation hook:** load an optional annotation file `annotations/folds/<stem>.json`. It holds polylines in network coordinates labelled `fold` or `not_fold`, with a tolerance in px. Compute precision and recall of accepted candidates. The hook only needs to work; I will annotate later (point-annotator).

**STOP.**

### F5 — Docs

- **`docs/fold_channel.md`.** Same style as `pipeline_tri_veto4.md`: every symbol explained.
  - Math: N1, N2, F2.1–F2.8, with short derivations: plane vector, exactness on planes, K-errors cannot create folds, scale invariance, the Rayleigh bias and the χ² gate, the veto ratio for constant curvature.
  - The J bound, its assumptions and its tightness vs the chord c. Table: Δθ 5/15/30/60/90/120° → c, J.
  - Parameters and sensitivity, results, limitations, open items.
- **`docs/geometry_channels.md`:** finalize, including a step-by-step "adding the depth channel" guide that names the `twosided.py` fields it will use.
- **`docs/viz_guide.md`:** add a section for the fold sheet and the profile tool.
- **`docs/pipeline_tri_veto4.md` §5:** update the open item on the normal-angle channel.

**STOP.**

# TASK_FOLD_V2 (F6–F9): fold channel — veto for narrow faces, local gate for moderate creases

Context: docs/fold_channel.md, docs/geometry_channels.md, docs/viz_guide.md. Reference runs: fold_only and
tri_veto4_fold (multiplicative null). A review of the 10 fold sheets found:

- The weight is effectively binary: w_fold < 0.5 equals w_fold < 0.9 to 3 decimals in 8/10 images (≤ 0.002 apart
  in 088, 090); median J of accepted is 0.96–1.00. Nothing is accepted between ~7° and ~20°. The null tail beyond the
  gate is far heavier than χ²₂ in all 10 images; in 075 and 090 the candidate and null histograms coincide through
  the gate.
- Narrow-face misses: the 079 plinth edge (§4.6), the 076 and 090 plinths, the edges of the 080 sculpture's narrow
  side face.
- no_veto acceptances are 2–12% of accepted (17–133 per image), largely on the 083 bollard limbs and the 079 box's
  left silhouette. Cause unknown.
- C has no upper bound; every image has a tail to C ≈ 1.5.

Goal: fix the veto (narrow faces, no_veto) and the gate (moderate creases) without regressing the existing checks.
Out of scope: silhouette ownership (integration step), geometry narrower than the windows, the Q_0.95 term of the
contamination check, the depth channel, the GPU port.

Constraints (as before):
- Every new behaviour sits behind an option whose built-in default is the current behaviour.
- With the new options off, baseline_v0, tri_veto4, channels: [laplacian], and the current fold_only and
  tri_veto4_fold outputs (arrays, candidate tables, JSON) reproduce bit-for-bit. Add a repro test for the two fold
  runs.
- New configs candidate_fold_v2.yaml / candidate_lap_fold_v2.yaml, runs fold_v2 / tri_veto4_fold_v2; arms get their
  own run names. configs/default.yaml stays unchanged.
- Flag borderline choices with a stated lean and continue; stop only at the F6 checkpoint.

## F6 — targets and diagnostics (stop and report)

1. Pin ROIs in upright network px and record them alongside the existing ones (scripts/fold_null_rois.py):
   - narrow faces: 079 plinth edge, 076 and 090 plinths, 080 side face (both edges);
   - moderate creases: 070 vertical seam where the left wall's near and far sections meet; 083 collar groove under
     the cap; any others with a clear plateau–step profile you find;
   - existing: 070 floor, 079 grass, 079 sunlit grass, 083 bollard, 079 box rows 230/254.
2. Run diag-fold on every target. Keep a crease target only if the dense normal-angle profile shows a step between
   two plateaus. Report its Δθ(k₁) and how it currently fails (gate, veto, invalid_window, component). A crease DA3
   rounded away is not a target.
3. Diagnose no_veto: run diag-fold on 3–5 no_veto candidates each on the 083 limbs and the 079 left silhouette, and
   report which k₂ samples were excluded and why. Hypothesis to check: another step's 1-px invalid-normal band
   crosses the window, and the bilinear-corner rule widens it to 2 samples, > 20% of a 9-wide window.
4. Characterize the C > 1.18 tail by class (§6.3 classes) and location.
5. Report the plan, with any changes to the designs below.

## F7 — veto with capped k₂ (evaluate with the current multiplicative gate)

- Per side, d_next± = distance along ±e₁ from x₀ to the first obstacle beyond the candidate's own rounding
  (|s| > width/2 + 1): an invalid normal, or another gate-passed candidate (lean: gate-passed only, so texture
  ridges don't cap faces). k₂± = clamp(min(2k₁, d_next± − m − 2), k₁, 2k₁).
- Constant-curvature ratio for those skips: C_curv = (2k₁ + m) / (k₂₊ + k₂₋ + m). Threshold
  C_min,eff = 1 − 0.4·(1 − C_curv), which gives 0.85 at k₂ = 2k₁, k₁ = 6. Decidable iff C_curv ≤ 0.7. Calibrate 0.4
  and 0.7 on the synthetic cylinders and box folds and report the margins.
- Undecidable, or a k₂ window invalid: accept only if both k₁ sides are strictly planar (planar fit, not
  fallback, rms_rel ≤ max(3ε̂, 1.25 b_tri(k₁, m)), i.e. without Q_0.95) and flag narrow_face; otherwise reject
  with reason veto_undecidable. This replaces the blanket no_veto acceptance.
- C > 1/C_min,eff: flag C_high, no rejection.
- Tests: synthetic faces 12, 18, 24, 32 px wide between two folds (report which edges are accepted); cylinder deep
  interior still 0/217, rim/limb not worse than 32/217; box folds C ≥ 0.95.
- Real images: report the result per target, and how the former no_veto candidates split into narrow_face and
  veto_undecidable. Losing the 083 limb acceptances is acceptable (they are steps); report how many.

## F8 — gate arms (with the F7 veto)

- A0: multiplicative (reference).
- A1 (lean): candidate-centred local roughness. Additive model; σ_geo per candidate solved so that
  median T_add = 2 ln 2 over the null samples within 32 px of x₀ (≥ 50 samples, else 48 px, else the global value),
  excluding null samples within k_min + m px of this candidate's own ridge component (8-connected NMS pixels).
  λ_n as now.
- A2: additive_local with 24-px tiles, min samples 30.
- Switching rule (the existing one plus one new criterion):
  - every ROI's share of w_fold < 0.5 ≤ A0 + 0.02, including 079 sunlit grass;
  - 079 box folds kept (w at rows 230/254, u 180–215, ≤ 0.25);
  - 083 bollard median w_fold ≥ 0.95;
  - new: at least half of the confirmed crease targets accepted.
  If no arm meets the rule, report the closest arm and leave the gate as is.
- Report per arm: the accepted Δθ histogram (< 5°, 5–20°, 20–50°, > 50°), the share of pixels with w_fold in
  [0.5, 0.9) per image, and the 075 class table (§6.3).

## F9 — runs, viz, docs

- Run fold_v2 and tri_veto4_fold_v2 on the 10 images. Per-image table: cand → valid → gate → veto → component;
  narrow_face, veto_undecidable, C_high counts; depth_jump share of accepted; w_fold < 0.5 / < 0.9; combined w < 0.5
  and its Δ vs tri_veto4; runtime.
- Fold sheet: panel 1 draws narrow_face segments dotted; panel 7 plots C / C_min,eff (line at 1) with narrow_face
  and veto_undecidable marked; the text box shows the new counts and the gate arm.
- eval-fold: extend the schema to polylines with classes fold, step, limb, texture (backward-compatible with
  fold/not_fold) and report precision/recall per class. Annotations will follow from me; no UI work.
- Update fold_channel.md (§4.5, §4.6, §5–§7), the decisions log in geometry_channels.md, and viz_guide.md.
---

## Do not

- Change `tri_veto4` or `baseline_v0` math, outputs or `configs/default.yaml`.
- Compute normals from raw ρ, or differentiate any upsampled map.
- Gate on Δθ directly, or use the rotation rate `r_n` as a magnitude.
- Let focal uncertainty enter the gate.
- Add DA3 passes, or depend on the RGB image in the fold channel (geometry only).
- Implement the depth channel. Only leave the stub and the shared `twosided.py`.
