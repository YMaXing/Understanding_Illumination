# Task: dedicated depth-edge channel (Canny-like, metric ΔZ)

## How to work

- Same session, same project directory as the fold channel (`geometry/dedicated/`). Run everything as `env -u PYTHONPATH uv run ...`.
- Context to read first: `docs/geometry_channels.md` (§5 holds the planned depth-channel defaults and test plan), `docs/fold_channel.md`, `docs/viz_guide.md`, and the decisions log.
- Then read the code: `geom/context.py`, `geom/twosided.py`, `geom/candidates.py` (depth stub), `geom/channels.py`, `geom/synthetic.py`, `geom/fold.py`, the evaluation classes (fold / step / limb / texture) and the saved ROI definitions.
- Work milestone by milestone (D0–D6). **Stop after each milestone**, report tersely (numbers, tables, figure paths), and wait for my go-ahead.
- **Commit at the end of each milestone** once its tests pass: one commit per phase on `yx_dev`, no push. Leave untracked files you didn't create alone.
- Decisions in this file are settled; apply them without asking. Flag any other borderline choice **with your lean**. Do not resolve it silently.
- Every new behaviour sits behind config options. The geometry channels stay off in `configs/default.yaml`.
- These must stay **bit-for-bit**: `baseline_v0`, `tri_veto4`, batch2 `tri_veto4`, geometry `[laplacian]`, `fold_v2` and `tri_veto4_fold_v2`. Extend the repro tests.
- Geometry code is pure numpy/scipy, float64, at network resolution, in the upright frame. No new DA3 passes, no RGB.
- If a test fails for a principled reason, report it. Do not loosen the assertion silently.

## Goal and scope

1. Build a `DepthChannel` that detects depth discontinuities Canny-style in disparity, measures each one's **metric jump ΔZ**, and outputs `w_depth`.
2. Register it, run `[depth]` and `[depth, fold]` (combined with min), and add visualization and docs.
3. **Out of scope:** the final fusion rule and retiring the Laplacian channel. Those belong to the next (integration) task.

---

## Math (settled)

### Symbols and conventions

- **Pixel:** `x = (u, v)`; u is the column, v is the row (downward). Network resolution, upright frame.
- **Disparities:** `ρ = 1/D`, with D the canonical DA3 depth. `ρ_f` is the context's trilateral-filtered disparity, as used by the fold channel. Read the context to confirm which field that is; do not change it.
- **Masks:** `S` = sky mask, `Ω = ¬S`. `ρ_floor` as in the context.
- **Intrinsics:** f_x, f_y, f_n (`f_n = (f_x + f_y)/2` at network resolution, from the Stage A JSON) and σ_f.
- **Metric depth:** `Z = f_n·D/300`. So in canonical units, `Z = f_n / (300·ρ)`.
- **Why disparity:** on any plane ρ is affine in pixel coordinates, so every second derivative vanishes on planes of any slant. Creases give a one-signed second-derivative bump. Steps give an antisymmetric doublet with a zero crossing at the step.

### D-K. Kernels (exact on affine functions)

Reuse the detector's `g0` and `g2` construction (moment-constrained, truncate 4σ, odd-reflect padding) and add a first-derivative kernel:

- `g1[x] = −(x/σ²)·g0[x]`, normalized so that the filter returns **exactly 1** on `f(u) = u`, whatever the convolution convention. It satisfies `Σg1 = 0` by antisymmetry.
- **Hessian of ρ_σ = G_σ ∗ ρ_f:**
  - `ρ_uu = g2(u)·g0(v)`
  - `ρ_vv = g0(u)·g2(v)`
  - `ρ_uv = g1(u)·g1(v)`
- **Exactness:** the Hessian is ≡ 0 on affine inputs, borders included, and exact on quadratics.
- Scale `σ = 1.25 px`, the same as the Laplacian channel.

### D-1. Orientation from a detrended gradient

- `Dd = ρ_σ − ρ_{σ_L}`, with `σ_L = 4·σ_tot` and `σ_tot = sqrt(σ² + σ_w,prior²)`.
  - Blurring preserves affine functions, so Dd cancels the background plane's slope exactly.
- `n = ∇Dd / |∇Dd|`. Compute it with g1 at σ and at σ_L: `∇ρ_σ − ∇ρ_{σ_L}`.
- Reject pixels with `|∇Dd| < k_g·median(|∇Dd| over Ω)`.
- **Why detrend:** a step that opposes the background slope g_p produces no gradient maximum when `|Δρ| < √(2π)·σ_tot·|g_p|`. On a floor that is up to 20% ΔZ/Z_near at 30 m. With n from Dd, the polarity rule holds for any background slope.

### D-2. Normalized second directional derivative

- `q = nᵀ H_σ n / max(ρ_σ, ρ_floor)`, signed.
  - On planes `q ≡ 0`.
  - At a step it is a doublet, positive behind the step and negative ahead, along +n (n points toward the near side, larger ρ).
  - At a crease it is a one-signed bump with no zero crossing.
- Noise scale: `ŝ_q = median(|q| over valid Ω) / 0.6745`.

### D-3. Candidates (zero crossings plus doublet test)

1. **Crossings.**
   - Sign changes of q between pixel pairs (u,v)–(u+1,v) and (u,v)–(u,v+1).
   - Subpixel location `x₀ = x_i + t·(x_j − x_i)`, with `t = q_i/(q_i − q_j)`.
   - Merge crossings that lie within 0.5 px of each other.
   - n at x₀: bilinear interpolation, then renormalize.
2. **Polarity.** Require `q(x₀ − 0.5n) > 0 > q(x₀ + 0.5n)`. This is the gradient-maximum signature; gradient minima are rejected.
3. **Lobes.** Sample q along `x₀ + s·n` for `s ∈ [−8, 8]`, in steps of 0.25 px.
   - The positive-lobe peak `q_p` at `s_p < 0`; the negative-lobe peak `q_m` at `s_m > 0`.
   - The separation `s_sep = s_m − s_p`.
4. **Two-sided height.** `min(|q_p|, |q_m|) > k_h·ŝ_q` (k_h = 3). This rejects planes (both lobes at noise level) and creases (only one lobe).
5. **Sharpness.** `s_lo ≤ s_sep ≤ s_hi`.
   - `s_lo = 0.8·2σ`.
   - `s_hi = 2·sqrt(σ² + σ_w,max²)`.
   - `σ_w,max` is calibrated in D2: P95 of DA3's boundary smear at unambiguous occlusions. Prior 3 px.
   - This rejects S-shaped smooth transitions and curved surfaces.
6. **Symmetry** (`depth.symmetry: off` by default): `|q_p|/|q_m| ∈ [1/4, 4]`.
   - *Flag:* occlusions often coincide with a slope change, which skews the lobes. *Lean:* off.
7. **Per-candidate smear and skip.**
   - `σ_w,c = sqrt(max(0, (s_sep/2)² − σ²))`.
   - `k_c = max(k_min_d, ⌈2σ_w,c⌉ + 1)`, with `k_min_d = 11`. Settled in F2: arm B gave 0.995 single-step recall.

### D-4. Jump at x₀ (shared `twosided.py`)

- Call `two_sided_fit` at x₀, with `e₁ = n`, `k = k_c`, `m = 8`, `w_t = 4`.
- **Add a `gap_mode` option:**
  - `min_span` is the current fold behaviour and stays the fold default, so fold_v2 remains bit-for-bit.
  - `at_x0` is new and is the depth default: `Δρ* = ρ̂₊(x₀) − ρ̂₋(x₀)`, where `ρ̂±(x₀) = α± + β±·(x₀ − c±)`.
  - Always record both, plus `planes_cross`.
- **Why `at_x0`:** at occlusions over grazing planes the two fitted planes can cross inside the band, and the min-span gap would then hide a real jump. The depth candidate's x₀ is the step's own zero crossing.
- Side "+" is +n, the near side. So `ρ̂₊(x₀) > ρ̂₋(x₀)` is expected. If `Δρ* ≤ 0`, the detection contradicts its own polarity: reject it with the reason `sign`.
- **Thin structures:** if a window is contaminated or invalid, retry with `m = 4`, then with a constant fit (β = 0). If both fail, use the doublet estimate `Δρ_doublet = 1.03·|q|_lobe·ρ_σ(x₀)·s_sep²` as the magnitude, flagged `thin`.
  - Always record `Δρ_doublet / Δρ*` as a sanity check.

### D-5. Gates

1. **Relative jump** (f-invariant): `r = ΔZ/Z_near = Δρ* / ρ̂₋(x₀)`, with ρ̂₋ the far-side disparity.
   - Hysteresis along linked contours (8-connected candidates):
     - Keep a component if every member has `r > τ_low` **and** at least one member has `r > τ_high`.
     - Defaults `τ_low = 0.02` and `τ_high = 0.04`.
   - Minimum component length `min_component_px = 5`.
2. **Significance:** `|Δρ*| / (SE(Δρ*)·sqrt(λ̂_Δρ)) > z`, with `z = 3.29` (two-sided p_fa = 1e-3).
   - `λ̂_Δρ` is the per-image inflation: `median((Δρ*/SE)² over null) / 0.455`, where 0.455 is the median of χ²₁.
   - The null is a stride-4 grid over valid Ω, using each pixel's own n, `x₀` = the pixel centre, and `k = 11` windows.
   - Clamp λ̂ to `[0.5, 1e4]`.
   - Also report the Monte-Carlo inflation on synthetic noise for k = 11. The geometry docs record λ_n = 121.2 for normals at k = 11; the Δρ inflation is a separate quantity, so record it separately.

### D-6. Magnitude

- **Contour smoothing:** running median of ρ̂₊(x₀) and ρ̂₋(x₀) **separately**, over 5 candidates along each linked contour. Do not smooth across junctions.
- **Metric jump:**

  ```
  ΔZ = (f_n/300) · (1/ρ̂₋ − 1/ρ̂₊)
  ```

  in metres. This is the settled ruling: the depth-edge magnitude is metric ΔZ, with near-noise edges filtered out by the gates.
- **Sky:**
  - If at least 50% of the far window is sky (prob ≥ 0.5), set `ΔZ = +inf`.
  - If the near window is in the sky, reject with the reason `sky_near`.
- **Very distant far side, not sky:** if `ρ̂₋ − z·SE₋ ≤ 0`, use the lower bound

  ```
  ΔZ_lb = (f_n/300) · (1/(ρ̂₋ + z·SE₋) − 1/(ρ̂₊ − z·SE₊))
  ```

  flagged `lower_bound`.
- **Focal diagnostics:** `[ΔZ_lo, ΔZ_hi]` from `f_n·(1 ∓ σ_f/f)`. Record only.
- **Known bias:** at smooth occluding contours (limbs), the near side falls off steeply, so affine extrapolation overestimates ΔZ. The bias is at most the occluder's own depth change within k px of the limb. Measure it on synthetic cylinders and document it.

### D-7. Weight, rasterization, strength

- **Per candidate:** `w_c = max(w_min, ΔZ₀ / (ΔZ + ΔZ₀))`, with `w_min = 0.05`. At sky (ΔZ = inf), `w_c = w_min`.
  - *Flag:* the knee `ΔZ₀`. *Lean:* 0.10 m. Report how sensitive the results are for ΔZ₀ ∈ {0.05, 0.10, 0.20} m.
- **Rasterize:** `w_depth(x)` = the minimum of `w_c` over accepted candidates within radius `b_c = max(band_min, ⌈σ_w,c⌉ + 1)`, with `band_min = 2`. Implement this as min-filters over the few distinct radii, as in fold. Elsewhere `w_depth = 1`.
- **`ChannelResult` contents:**
  - `strength` = r = ΔZ/Z_near at rasterized pixels (dimensionless), else 0.
  - `arrays`: `dZ` (float32 metres; inf at sky), `q_over_s` (signed `q/ŝ_q`, float16), `edge_mask`.
  - Candidate parquet with every candidate and its accept/reject reason: `polarity`, `height`, `sharpness`, `sign`, `rel`, `signif`, `component`, `sky_near`.
  - Scalars: `ŝ_q`, `λ̂_Δρ`, `σ_w,max` used, counts per stage.

---

## Architecture

- **`geom/depth.py`:** `DepthChannel`, implementing D-1 through D-7. It is registered as `"depth"` in the lazy registry.
- **`geom/candidates.py`:** replace the depth stub with `depth_candidates(ctx, params)`, implementing D-K and D-1 through D-3.
- **`geom/twosided.py`:** add `gap_mode` (`min_span` | `at_x0`). The default `min_span` leaves fold outputs unchanged.
- **The depth channel does not need dense normals.** A `[depth]`-only run must not trigger the dense normal fit; check the context's laziness and keep `normals_computed` accurate.
- **Config:** add a `geometry.depth` section mirroring the D-* parameters. Validate it, support nested `--set`, and test that the config fields match the parameter classes field for field.
- **New configs:**
  - `configs/candidate_depth_only.yaml`: channels `[depth]`, run `depth_v1`.
  - `configs/candidate_depth_fold.yaml`: channels `[depth, fold]` with the fold_v2 settings, run `depth_fold_v1`.

---

## Milestones

### D0 — Design check (no code)

Write `docs/depth_channel.md` (draft) covering:
- the module map and data flow;
- the `two_sided_fit` fields used and the `gap_mode` change, with the bit-for-bit argument for fold;
- context laziness for `[depth]`;
- where the evaluation classes and ROIs come from.

List any ROI this task needs that does not exist yet (e.g. 079 box silhouettes, 070 sky boundary, 083 bollard limb). Propose coordinates and overlay figures for me to check. **STOP.**

### D1 — Kernels and the candidate field

Implement D-K, D-1 and D-2. Tests (synthetic, from `geom/synthetic.py`):
1. `g1` moments, and the Hessian exact on quadratics. `H ≡ 0` to 1e-12 on affine inputs, borders included.
2. Planes of random slant give `q ≡ 0` (≤ 1e-9), noise-free.
3. Detrending theorem: a step against the background slope with `|Δρ| < √(2π)σ_tot|g_p|`.
   - With n from ∇ρ_σ, polarity fails.
   - With n from ∇Dd, polarity holds.

**STOP.**

### D2 — Candidates, doublet test, calibration

Implement D-3 and `scripts/calibrate_depth.py`, which writes to `geom/data/`.
- **Smear:** `σ_w,max` = P95 of σ_w,c on real candidates with r > 0.2 and high significance, from a first pass with the prior.
- **Inflation:** `λ̂_Δρ` per k on synthetic noise at levels 2e-4 to 1e-3, plus the real-image null.

Ablation on the 070 floor ROI: candidate counts at each stage with `ρ_f` vs raw `ρ`. *Lean:* `ρ_f` (it removes the 7-px grid ripple). Report.

Tests:
4. Creases at 30–120° (fold scenes, away from steps): **0 accepted** depth candidates.
5. A plane plus a 7-px grid ripple of 0.2%, through the trilateral filter: **0 accepted**.

**STOP.**

### D3 — Jump, gates, magnitude, weights

Implement D-4 through D-7. Tests:
6. Single steps on slanted backgrounds, both polarities and against-slope, with r ≥ 5%: recall ≥ 95%.
7. **Strict silhouette recall ≥ 90% at k ≥ 11** on the box scene. This moved here from fold test 5.
8. ΔZ accuracy:
   - noise-free steps on planes: relative error ≤ 1e-6;
   - with a DA3-like smear (blur ≤ 3 px): ≤ 1%.
9. Invariances:
   - the accepted set is invariant to `D → cD` and to `f → κf`;
   - ΔZ scales by c·κ.
10. Sky: a step to sky gives ΔZ = inf and w = w_min. A near window in the sky is rejected.
11. A distant far side (ρ̂₋ ≈ 0, not sky) uses the lower bound, with the flag set.
12. Thin structures (poles 4–20 px wide): no crashes, flags set. Report detection rate and ΔZ error.
13. Cylinder limbs: report the ΔZ bias against the analytic value.
14. Noise null: λ̂_Δρ within 20% of the Monte-Carlo inflation.

**STOP.**

### D4 — Channel integration

Implement the registry entry, configs, Stage B saving and the manifest.

Tests:
- All repro references listed under "How to work" stay bit-for-bit, including fold_v2.
- `[depth]` runs without computing dense normals.
- `[depth, fold]` gives the elementwise min.
- Saved keys and the parquet schema are as specified.

**STOP.**

### D5 — Visualization, runs, evaluation

**Depth sheet** `stage_b/<run>/viz_depth/<stem>.png`, upright, a 3×3 grid plus a text box:
1. sRGB with accepted edges coloured by log10 ΔZ (sky edges marked separately) and rejected candidates in gray.
2. log10 Z.
3. Signed `q/ŝ_q` (RdBu_r, ±20).
4. `|∇Dd|` (log), with n arrows subsampled.
5. Histogram of `s_sep` with `s_lo` and `s_hi`.
6. Histogram of r (log x) with τ_low and τ_high.
7. Significance histogram: null vs candidates, with the gate.
8. `w_depth`.
9. Combined w over the sRGB (red alpha).
10. Text box: ŝ_q, λ̂_Δρ, σ_w,max, counts per stage, f and f_unc, ΔZ₀.

**Profile tool:** `cli diag-depth --stem S --u U --v V`. Plot:
- the ρ_f and q profiles along n through the nearest candidate, with the lobes marked;
- the windows on the image;
- the fitted planes extrapolated to x₀;
- Δρ*, r, ΔZ and `Δρ_doublet/Δρ*`.

**Runs** on the 10 images: `depth_v1` and `depth_fold_v1`. Compare them against `tri_veto4` and `tri_veto4_fold_v2`. For each run report:
- share w < 0.5 and share w < 0.9 per image;
- the share of w_depth < 0.5 per evaluation class (fold / step / limb / texture);
- candidate counts per stage and runtime per image.

**Spot-check targets** (my estimates: report them, don't tune to them silently):

| check | target |
|---|---|
| 070 floor, share `w_depth < 0.9` | ≤ tri_veto4 + 0.01 |
| 070 sky boundary, median `w_depth` | ≤ 0.10 |
| 079 grass, both ROIs, share `w_depth < 0.5` | ≤ 0.02 |
| 079 box silhouettes, median `w_depth` | ≤ 0.30 |
| 079 box folds (rows 225–260), median `w_depth` | ≥ 0.90 (the depth channel must not fire on folds) |
| 083 bollard limb, median `w_depth` | ≤ 0.50 |
| 083 bollard base rim (fold), median `w_depth` | ≥ 0.90 |
| 088 rock ridge and 076 façade ledge (creases), median `w_depth` | ≥ 0.90 |
| depth channel runtime, excluding the context | < 3 s/image on CPU |

**STOP.**

### D6 — Docs

- **`docs/depth_channel.md`.** Final version, in the style of `fold_channel.md`: every symbol explained.
  - Math: disparity affinity, the kernels, the detrending theorem, the doublet test and why each criterion exists, the link between lobe separation and smear, `at_x0` vs `min_span`, the ΔZ conversion, both gates, sky and lower-bound handling, the weight, rasterization and strength.
  - Calibration tables and parameter sensitivity (including ΔZ₀), results, limitations (limb bias, thin structures, junctions), open items.
- **`docs/geometry_channels.md`:** a depth-channel section, plus integration notes for the next task: fusion on strengths vs weights, and retiring the Laplacian channel.
- **`docs/viz_guide.md`:** the depth sheet and the profile tool.
- **Decisions log:** every choice made in this task.

**STOP.**

---

## Do not

- Change fold-channel behaviour or outputs, `configs/default.yaml`, or any repro reference.
- Use the min-span gap for the depth magnitude, or the lobe height as a magnitude (except the flagged `thin` fallback).
- Gate on a relative jump without the significance test, or vice versa.
- Differentiate upsampled maps, or compute derivatives from raw ρ in the default path.
- Implement a fusion rule other than min, or retire the Laplacian channel. Both belong to the next task.
