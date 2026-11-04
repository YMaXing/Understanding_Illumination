# The depth channel: depth discontinuities and their metric jump as a geometric-edge weight

Status as of 2026-11-01 (TASK_DEPTH_CHANNEL D0–D6; the development record of each phase is Appendix A):

- **Code:** `illum_baseline/geom/candidates.py` (`depth_kernels`, `depth_field`, `depth_candidates`) and
  `illum_baseline/geom/depth.py` (`DepthParams`, `DepthChannel`), on the shared `geom/twosided.py` and
  `geom/context.py`. Pure numpy/scipy, float64, network resolution, upright frame; depth only (no RGB), no new DA3 pass.
  Architecture and the channel interface: `docs/geometry_channels.md`.
- **Runs:** `configs/candidate_depth_only.yaml` (`[depth]`, run `depth_v1`) and `configs/candidate_depth_fold.yaml`
  (`[depth, fold]` with the fold_v2 settings, combined by the elementwise minimum, run `depth_fold_v1`). The gate null
  is `geometry.depth.null_model: additive` (the default since D5; section 3.7). The saved D5 runs `depth_v1` and
  `depth_fold_v1` were made with the multiplicative null, `depth_v1_add` with the additive one.
- **Default pipeline:** `configs/default.yaml` is geom_v1_1 (this channel with `sky_band: depth_and_mask`; geom_v1 from TASK_GEOM_INTEGRATION G6 until the TASK_GEOM_V2 wrap-up; `docs/geometric_edges.md`); at the time of this task it had the geometry channels off; `baseline_v0`, both
  `tri_veto4`, `[laplacian]`, `fold_only`, `tri_veto4_fold`, `fold_v2` and `tri_veto4_fold_v2` reproduce bit for bit
  (`tests/test_baseline_repro.py`, `tests/test_geom_stage_b.py`).
- **Visualization:** the depth sheet and the profile tool `diag-depth`, `docs/viz_guide.md`.

## 1. What it does

The tri_veto4 weight responds to curvature of the disparity; the fold channel to a change of the surface normal. Neither
measures an **occlusion**: a place where one surface passes in front of another and the depth jumps. The depth
channel finds those jumps Canny-style in the disparity and measures each one's **metric jump ΔZ** in metres:

1. **Field.** Smooth the filtered disparity, take its Hessian H and a *detrended* gradient (the gradient minus the
   gradient at a 4× larger scale), whose direction n points across the edge toward the near side (D-K, D-1).
2. **Candidates.** The second derivative along n, q = nᵀHn/ρ, has a zero crossing with an antisymmetric doublet at a
   step, a one-signed bump at a crease and nothing on a plane. Keep zero crossings that are gradient maxima with two
   significant, sharp, centred lobes (D-2, D-3).
3. **Jump.** Fit one plane on each side, beyond the DA3 smear, and extrapolate both to the crossing: Δρ* = ρ̂₊(x₀) −
   ρ̂₋(x₀) (D-4). Thin structures fall back to shorter windows, a constant fit, then the doublet itself.
4. **Gates.** Keep a candidate when the jump has the right sign, is significant against this image's own null
   sample, and is relatively large (ΔZ/Z_near > 2 %, with hysteresis along the contour) (D-5).
5. **Magnitude.** ΔZ = Z_far − Z_near in metres, smoothed along the contour; infinite at the sky; a lower bound when
   the far side is indistinguishable from infinity (D-6).
6. **Weight.** w_c = max(0.05, ΔZ₀/(ΔZ + ΔZ₀)) with ΔZ₀ = 0.10 m, painted in a 2–3 px band around the edge; combined with
   the other channels by the minimum (D-7).

## 2. Notation

| symbol | meaning |
|---|---|
| x = (u, v) | pixel; u column, v row (downward); network resolution, upright frame |
| D | canonical DA3 depth (Stage A cache); Z = f_n D / 300 metric depth, f_n = (f_x + f_y)/2 at network resolution |
| ρ = 1/D, ρ_f | disparity; ρ_f its trilateral-filtered version (the context's `rho_f`, the fold channel's field; ρ = 0 on the sky) |
| S, Ω = ¬S | sky mask (B1, from DA3's sky probability ≥ 0.5 after morphology) and its complement |
| ρ_floor | the detector's disparity floor (0.5 × the 1st percentile of ρ over Ω) |
| σ = 1.25 px | Hessian scale (as the Laplacian channel); σ_w,prior = 1 px DA3 smear prior; σ_tot = √(σ² + σ_w,prior²) = 1.60 px |
| g0, g1, g2 | moment-constrained Gaussian, first- and second-derivative kernels at σ (truncate 4σ) |
| ρ_σ = G_σ ∗ ρ_f; H | smoothed disparity and its Hessian (H_uu, H_vv, H_uv) |
| Dd = ρ_σ − ρ_{σ_L}, σ_L = 4σ_tot = 6.4 px | detrended disparity; ∇Dd its gradient; n = ∇Dd/\|∇Dd\| (toward larger ρ, the near side) |
| g_n = \|∇Dd\|/max(ρ_σ, ρ_floor); σ̂_g | normalized gradient; its Rayleigh scale median(g_n over Ω)/1.1774 |
| h = nᵀHn; q = h/max(ρ_σ, ρ_floor) | second derivative along n, unnormalized and normalized (scale-free) |
| ŝ_q | noise scale of q: median(\|q\|)/0.6745 over Ω ≥ 5 px from the sky and the border |
| x₀ | subpixel zero crossing of q; s the coordinate along n through x₀; t across (e_t = n rotated by +90°) |
| s_p < 0 < s_m; q_p, q_m; h_p, h_m | positions and heights of the positive and negative lobes; s_sep = s_m − s_p |
| σ_w,c = √(max(0, (s_sep/2)² − σ²)) | the candidate's own smear; k_c = max(11, ⌈2σ_w,c⌉ + 1) its window skip |
| W±(k) | one-sided windows {x₀ ± s n + t e_t : s ∈ [k, k + m], \|t\| ≤ w_t}, m = 8, w_t = 4 |
| ρ̂±(x₀), SE± | each side's plane extrapolated to x₀ and its standard error; side + is +n, the near side |
| Δρ*, SE(Δρ*) | the jump at x₀, ρ̂₊(x₀) − ρ̂₋(x₀), and its SE |
| r = Δρ*/ρ̂₋ = ΔZ/Z_near | relative jump (f-invariant) |
| λ̂_Δρ, σ_geo,ρ, λ_n | null inflation (multiplicative), additive model error and noise-only inflation (additive) |
| ΔZ, ΔZ₀, w_c, b_c | metric jump, weight knee (0.10 m), candidate weight, band radius |

## 3. Method

### 3.1 Why disparity, and the kernels (D-K)

A plane n·X = d seen through a pinhole camera has inverse depth 1/Z = (n·K⁻¹x̃)/d: **ρ is affine in (u, v) on every
plane**, of any slant. All second derivatives of ρ therefore vanish on planes; a crease is a one-signed second
derivative, and a step an antisymmetric doublet.

g0 and g2 are the detector's kernels (Σ g0 = 1, Σ g2 = 0, Σ x² g2 = 2). The new first-derivative kernel is
g1 = −(x/σ²) g0, normalized so that the implemented correlation returns exactly 1 on f(u) = u (Σ x g1 = 1; Σ g1 = 0
by antisymmetry). The Hessian of ρ_σ is ρ_uu = g2(u)·g0(v), ρ_vv = g0(u)·g2(v), ρ_uv = g1(u)·g1(v), with odd-reflect
padding, so H ≡ 0 on affine inputs including the borders (tested to 1e-12) and is exact on quadratics.

### 3.2 Orientation from a detrended gradient (D-1) and the gradient floor

n should point across the edge toward the near side. The gradient of ρ_σ fails for a step that opposes the
background slope g_p: along s, ∂ρ_σ/∂s = g_p + Δρ φ(s/σ_tot)/σ_tot, which keeps the sign of g_p everywhere when
**|Δρ| < √(2π) σ_tot |g_p|** (the detrending theorem). On a floor that is a jump of up to ~20 % ΔZ/Z_near at 30 m. The
step then has no gradient maximum and the polarity rule below fails. Blurring preserves affine functions, so
Dd = ρ_σ − ρ_{σ_L} removes the background plane exactly and its gradient follows the step for any background slope
(test 3: plain gradient 0/26 rows pass, detrended 26/26).

Pixels whose normalized gradient is at noise level have no meaningful n. g_n = |∇Dd|/max(ρ_σ, ρ_floor) is
scale-free; for i.i.d. Gaussian gradient components |∇| is Rayleigh with median 1.1774 σ, so σ̂_g = median(g_n)/1.1774
and a pixel is valid when g_n > 3 σ̂_g (about 1.1 % of pure noise passes). A crossing is kept when g_n, interpolated
between its two pixels, is above that floor. The floor costs no recall on synthetic steps (edge coverage 1.000 with and
without it); on real images 24–38 % of Ω is valid, because real gradients are slopes and edges, not noise.

### 3.3 The normalized second derivative (D-2)

h = nᵀHn and q = h/max(ρ_σ, ρ_floor) (q makes the noise test scale-free). A step of height Δρ smeared to σ_tot gives
along s

  h(s) = Δρ φ′(s/σ_tot)/σ_tot² — positive behind the step (s < 0), negative ahead,

with peaks at s = ∓σ_tot of height e^{−1/2} Δρ/(√(2π) σ_tot²) = 0.242 Δρ/σ_tot², separation **s_sep = 2σ_tot**,
and a zero crossing at the step. A crease gives a one-signed bump with no crossing; a plane gives h ≡ 0. q is not
zeroed on the sky: ρ = 0 there is the far limit, so a sky boundary is a doublet like any step.

### 3.4 Candidates: zero crossings and the doublet test (D-3)

Each stage rejects something specific:

1. **Crossings.** Sign changes of q between 4-neighbour pixel pairs (at least one pixel in Ω), subpixel
   x₀ = x_i + t (x_j − x_i), t = q_i/(q_i − q_j); crossings within 0.5 px merged; n at x₀ by bilinear interpolation.
2. **Polarity** q(x₀ − n/2) > 0 > q(x₀ + n/2): the crossing is a gradient *maximum* (an edge), not a minimum.
3. **Lobes.** Along x₀ + s n, s ∈ [−8, 8] (step 0.25), each lobe is the run of one sign contiguous with x₀; its peak is
   the nearest local maximum reaching ½ of the run's maximum (inside a thin structure the near lobes of both edges
   form one run), parabola-refined. Positions come from **h**, heights are **q** at those positions: q's division by
   the varying ρ_σ drags the far lobe outward for large jumps (to −1.76 σ_tot·px at r = 1, −3.3 at r = 100), h does
   not.
4. **Two-sided height** min(|q_p|, |q_m|) > 3 ŝ_q: rejects planes (both lobes at noise level) and creases (one lobe).
5. **Sharpness** s_lo = 0.8·2σ = 2.0 px ≤ s_sep ≤ s_hi = 2√(σ² + σ_w,max²) = 3.23 px: rejects S-shaped smooth
   transitions and curved surfaces. The lobe separation **is** the smear: σ_w,c = √((s_sep/2)² − σ²). Pixel sampling
   adds about 0.55 px in quadrature (an ideal step measures s_sep 2.72, σ_w,c 0.55).
6. **Centring** |s_p + s_m| ≤ 0.4 s_sep: rejects ghost lines. Beyond a step's core the detrended n flips sign
   (−∇ρ_{σ_L} dominates), and the trilateral filter leaves small wiggles 5–7 px out; a crossing there borrows the step's
   own lobe ~5 px away (centring ≈ 0.54–0.9). Real occlusions: P99 0.19, max 0.28.
7. **Symmetry** of the lobe heights: off (occlusions often coincide with a slope change that skews them).

### 3.5 The jump at x₀ (D-4)

`two_sided_fit` fits an OLS plane to ρ_f on each window W±(k_c) (samples on Ω only, all four bilinear corners usable,
≤ 20 % excluded) with the fold channel's contamination check (rms/α ≤ max(3 ε̂, Q₀.₉₅ of the null rms,
1.25 b_tri(k, m)), retry m = 4). k_c ≥ 11 keeps the windows clear of the trilateral bias zone (9 px) and of the smear.

**Gap at x₀ vs min-span.** The fold channel's Δρ* is the smallest plane gap over s ∈ [−k, k] and 0 when the planes
cross there. At occlusions over grazing planes the two fitted planes can cross inside that span, and the min-span gap
hides a real jump (079 box bottom edge, u 226: planes_cross, min-span 0, gap at x₀ 0.026, r 4.6 %). The crossing x₀ is
the step's own location, so the depth channel uses **Δρ* = ρ̂₊(x₀) − ρ̂₋(x₀)** (`gap_mode: at_x0`); the min-span gap
and `planes_cross` are recorded.

**Retries for thin structures.** Per side: plane with m = 8 → m = 4 → a constant fit (β = 0) → the doublet estimate
Δρ_doublet = 1.03 |q|_lobe ρ_σ(x₀) s_sep² = 1.03 |h|_lobe s_sep² (from the peak height above; flag `thin`). A side
whose window lies beyond the next edge (another D-3 candidate whose own doublet estimate passes τ_low, inside the
strip) uses the constant fit on the clear span [s_a, min(d_next − s_a, k_c + m)], s_a = ⌈s_hi/2⌉ + 1.
Δρ_doublet/Δρ* is always recorded (median 1.05 on real candidates).

**Sign.** Side + is the near side, so Δρ* > 0 is expected; Δρ* ≤ 0 contradicts the candidate's own polarity (reason
`sign`).

### 3.6 Gates (D-5)

A candidate passes D-3, has the right sign, and then:

- **Significance.** Additive null (default): T = Δρ*²/(λ_n SE² + σ_geo,ρ²) > z², z = 3.29 (two-sided p_fa = 1e-3), with
  λ_n = 104.6 the noise-only inflation of (Δρ*/SE)² at k = 11 (synthetic calibration) and σ_geo,ρ solved per image so
  that the median of T over the null sample is the χ²₁ median 0.455. Multiplicative null: |Δρ*|/(SE √λ̂_Δρ) > z,
  λ̂_Δρ = median((Δρ*/SE)² over the null)/0.455, clamped to [0.5, 1e4]. The null sample: a stride-4 grid over Ω, each
  pixel's own n, x₀ = the pixel centre, k = 11. Both statistics are always recorded. Thin (doublet) candidates use
  min(|q_p|, |q_m|)/ŝ_q > z.
- **Relative jump with hysteresis.** r = Δρ*/ρ̂₋ (a lower bound when ρ̂₋ is consistent with 0, section 3.7). Candidates
  within 3 px of each other form a contour component (merged crossings lie up to ~2.2 px apart on oblique edges); a
  component is kept when every member has r > τ_low = 0.02, one member has r > τ_high = 0.04, and it covers ≥ 5
  distinct pixels.
- **Sky.** If ≥ 50 % of the far span is sky (S), the edge is a sky edge: r = significance = ΔZ = ∞. If ≥ 50 % of the
  near span is sky, the candidate is rejected (`sky_near`). A sky edge's far window is sky, so its jump falls back to
  the doublet and it carries the `thin` flag (kept, G3 decision): exclude sky edges when counting thin-ring clusters.

Neither gate is used alone: significance protects against noise at small ρ, the relative jump against large but
physically small jumps. On real images significance is the binding gate for 73–100 % of candidates (multiplicative;
r_gate = z√λ̂ SE/ρ̂₋ P50 0.035–0.34): it decides most edges, τ_low few.

### 3.7 Magnitude (D-6)

- **Contour smoothing.** Accepted candidates are chained (each links to its nearest neighbour ahead and behind along the
  tangent within 1.5 px, n·n ≥ 0.5, links kept only when mutual; junctions break chains). ρ̂₊ and ρ̂₋ are each replaced
  by the running median over 5 chained candidates of the neighbours' planes evaluated at the candidate's own x₀ (exact
  on planes; window shrunk symmetrically at chain ends).
- **Metric jump** ΔZ = (f_n/300)(1/ρ̂₋ − 1/ρ̂₊) in metres. It scales with the focal (f → κf: ΔZ → κΔZ) and the depth
  scale (D → cD: ΔZ → cΔZ); the accepted set is invariant to both (test 9).
- **Sky:** ΔZ = ∞, w_c = w_min.
- **Lower bound.** If ρ̂₋ − z SE_eff,₋ ≤ 0 (the far plane is consistent with infinity; SE inflated as the active null
  does: √λ̂ SE, or √(λ_n SE² + σ_geo,ρ²/2) per side), ΔZ_lb = (f_n/300)(1/(ρ̂₋ + z SE_eff,₋) − 1/(ρ̂₊ − z SE_eff,₊)),
  infinite when even ρ̂₋ + z SE_eff,₋ ≤ 0; flag `lower_bound`, and the relative-jump gate uses the matching lower bound.
- **Focal bracket** [ΔZ_lo, ΔZ_hi] from f_n (1 ∓ σ_f/f): recorded only.

### 3.8 Weight, rasterization, strength (D-7)

w_c = max(w_min, ΔZ₀/(ΔZ + ΔZ₀)), w_min = 0.05, ΔZ₀ = 0.10 m (0.5 at a 10-cm jump, 0.09 at 1 m). Each accepted
candidate paints its rounded pixel's disk of radius b_c = max(2, ⌈σ_w,c⌉ + 1) (2–3 px) with w_c; w_depth is the
minimum over all disks (min-filters per radius), 1 elsewhere. The strength is r = ΔZ/Z_near (maximum over the disks),
0 elsewhere, and **∞ at sky edges** (and unbounded far sides): any future fusion on strengths must treat ∞ as
saturated. Saved arrays: `depth_w_net`, `depth_strength`, `depth_dZ` (float32 metres), `depth_q_over_s`,
`depth_edge_mask`; the candidate and null tables (Appendix A.3).

## 4. Parameters, calibration and sensitivity

| parameter | value | source |
|---|---|---|
| σ, truncate | 1.25 px, 4 | the Laplacian channel's scale |
| σ_w,prior, σ_L | 1 px, 4σ_tot = 6.4 px | settled (D-1) |
| k_g | 3 (Rayleigh floor) | D2 |
| k_h | 3 | settled |
| s_lo | 0.8·2σ = 2.0 px | settled |
| σ_w,max | **1.02 px** (s_hi 3.23 px) | P99 of σ_w,c at unambiguous real occlusions (below) |
| centre_tol | 0.4 | D3 |
| k_min, m, w_t | 11, 8, 4 | F2 arm B; settled |
| z | 3.29 | settled (p_fa = 1e-3) |
| null_model, λ_n | additive, 104.6 | D5 decision rule; synthetic calibration |
| τ_low, τ_high, min component | 0.02, 0.04, 5 px; link radius 3 px | settled; D3 |
| smoothing | 5 candidates | settled |
| ΔZ₀, w_min, band_min | 0.10 m, 0.05, 2 px | settled lean; sensitivity below |

**σ_w,max** (`scripts/calibrate_depth.py` → `geom/data/depth_calibration.json`). First pass on the 10 images with the
prior in the detrending but no smear bound; unambiguous occlusions: passed D-3, both sides planar at k_c,
r > 0.2, significance > 2z, and the largest g_n on their normal line within ±8 px.

| selection | n | σ_w,c P50 | P90 | P95 | P99 |
|---|---|---|---|---|---|
| main (r > 0.2, sig > 2z, dominant) | 576 | 0.00 | 0.68 | **0.77** | **1.02** |
| sig > z, dominant | 704 | 0.00 | 0.73 | 0.79 | 1.04 |
| sig > 4z, dominant | 448 | 0.00 | 0.60 | 0.73 | 0.84 |
| sig > 2z, not dominance-filtered | 3607 | 0.71 | 0.78 | 0.83 | 1.93 |

The rule is P99 (P95 rejects 5 % of real occlusions by construction; between the two the Δρ* gate at k ≥ 11 decides).
DA3 edges at network resolution are sharper than an ideal pixel-sampled step (real s_sep P50 2.45 vs 2.72).

**λ̂_Δρ on synthetic noise** (noisy planes, the null grid, k ∈ {11, 13, 16}): median-based 97.9–107.1, Monte-Carlo
mean(z²) 106.2–119.2, flat in k (λ_n = 104.6 is k = 11 at noise 5e-4). Real images (multiplicative): λ̂_Δρ 660–1391 —
real surfaces are not planes at the window scale, which the additive model absorbs as σ_geo,ρ instead.

**ΔZ₀** (share of w_depth < 0.5, multiplicative run): ΔZ₀ = 0.05 / 0.10 / 0.20 m gives identical shares of w < 0.9 on every
image and changes w < 0.5 by at most 0.013 (084: 0.045 / 0.040 / 0.028; 090: 0.080 / 0.079 / 0.069), because accepted
edges have ΔZ P50 0.2–6.7 m, far above the knee. The weight is nearly binary in practice.

**ρ_f vs raw ρ** (070 floor ROI): raw ρ's 7-px grid ripple leaves 193 D-3 candidates in the floor ROI, ρ_f none.

## 5. Results

### 5.1 Synthetic checks (tests 1–14)

`tests/test_depth_field.py`, `tests/test_depth_candidates.py`, `tests/test_depth_channel.py`:

| check | result |
|---|---|
| kernels, planes, detrending theorem | H ≡ 0 on affine (1e-12); q ≤ 1e-9 on planes, no normals built; polarity 0 → 26/26 rows with detrending |
| creases 30–120°, 7-px ripple | no D-3 candidate (creases: none within 6 px of the crease with noise) |
| single steps r 5–200 %, both polarities, against the slope, smear ≤ 0.8 px | edge recall 1.000 |
| box silhouettes, k ≥ 11 | recall 0.950 (additive; 0.910 multiplicative) |
| ΔZ, noise-free / smear 0.5–3 px | relative error ≤ 1.3e-14 / ≤ 0.9 % |
| invariance D → 2D, f → 1.3f | accepted set identical, ΔZ scales by cκ (1e-9) |
| sky; near side in the sky | ΔZ = ∞, w = w_min; `sky_near` |
| far side on the horizon | `lower_bound` on all edge candidates |
| poles 4 / 6 / 8–12 / 16–20 px | not detected / doublet (ΔZ error 15 %) / constant fit / planes; no component merges both sides |
| cylinder limbs R 0.25 / 0.5 / 1 m | ΔZ bias +9 / +32 / +7 %, within the cylinder's own depth change over k px (14 / 44 / 10 %) |
| null | λ̂_Δρ −11 % against the Monte-Carlo inflation |

### 5.2 Runs (10 images)

Share of w < 0.5 per image (`scripts/depth_d5_report.py` → `baseline_output/diag/depth_v1/d5/d5_report.json`):

| img | depth (mult) | depth (add) | depth + fold | tri_veto4 | tri_veto4_fold_v2 |
|---|---|---|---|---|---|
| 070 | .017 | .017 | .042 | .215 | .215 |
| 075 | .037 | .040 | .091 | .286 | .307 |
| 076 | .024 | .026 | .061 | .224 | .234 |
| 079 | .019 | .021 | .065 | .216 | .237 |
| 080 | .033 | .037 | .080 | .215 | .225 |
| 082 | .025 | .032 | .081 | .253 | .257 |
| 083 | .016 | .017 | .041 | .159 | .160 |
| 084 | .040 | .048 | .097 | .271 | .295 |
| 088 | .010 | .010 | .030 | .105 | .109 |
| 090 | .079 | .081 | .114 | .332 | .346 |

Runtime: 0.57–0.83 s per image after a 0.3–0.4 s context (trilateral ρ_f; shared with the fold channel); the dense
normal fit is never built by `[depth]`.

**Per class** (share of w_depth < 0.5 on the pinned ROIs, multiplicative / additive): step 0.774 / 0.775, limb
0.997 / 0.997, fold 0.002 / 0.002, texture 0.001 / 0.003. "080 side face edge R" is evaluated as a step (its upper half is
the side face's outer silhouette; the fold evaluation keeps it as a legacy fold target).

**Spot checks** (all met under both nulls): 070 floor share w_depth < 0.9 = 0.000 (target ≤ tri_veto4 + 0.01 = 0.041);
070 sky boundary median 0.05 (≤ 0.10); 079 grass 0.000 / 0.003 and sunlit grass 0.003 / 0.006 (≤ 0.02); 079 box
silhouettes median 0.05–0.08 (≤ 0.30); 079 box folds median 1.00 (≥ 0.90); 083 limbs 0.11 / 0.13 (≤ 0.50); 083 base
rim, 088 rock ridge, 076 façade ledge 1.00 (≥ 0.90); runtime ≤ 0.83 s (< 3 s).

**Null model.** Additive accepts 3–33 % more candidates per image, recovers occlusions the multiplicative null rejects
only by significance (075: building corners, the kiosk base; 362 → 258 significance-only rejections) and raises the box
silhouette recall 0.910 → 0.950. Decision rule (texture ≤ multiplicative + 0.02 on the 070 floor and both 079 grass ROIs,
step and limb targets kept): met; additive is the default since D5.

### 5.3 Checks before the final docs

- **Box–plinth contact (079) and kiosk base (075)**, 4 points each: every point is a step in DA3's depth, not a contact
  fold misread. On the ρ_f samples within ±k (transition ±2 px excluded) a two-plane step beats a continuous kink by
  ΔBIC 450–2750; step jumps 3–8 % (24 % at the box's right corner); r 4.4–26 %, ΔZ 0.09–0.52 m. The fold channel
  accepts folds at the same places (Δθ 52–95°). One point has planes_cross with a min-span gap of 0 (section 3.5).
  Whether the physical contact overhangs cannot be told from depth; the channel reads DA3 faithfully.
- **070 sky boundary band** (1766 px, the 2-px band of the sky-mask boundary): 457 px with w_depth ≥ 0.5, of which
  375 lie just beyond the depth band (median 2.9 px from the nearest accepted edge, band radius 2: the accepted sky
  edges sit ~1 px on the building side of the sky-mask boundary), 20 in gaps between accepted candidates and 62 at
  corners or junctions (46 junctions). Median w_depth over the band 0.05.
- **090 (chain-link fence)**: DA3 renders the mesh as small depth blobs; 1995 of 2904 accepted candidates lie in the
  fence region, 89 % thin (doublet) rings; w_depth < 0.5 on 24.5 % of the region vs 3.1 % elsewhere. Not tuned: a clutter
  policy belongs to the integration.

## 6. Limitations

- **Limb bias.** At smooth occluding contours the near side falls away steeply, and the affine extrapolation
  overestimates ΔZ by up to the occluder's own depth change within k px of the limb (+7 to +32 % on synthetic
  cylinders).
- **Thin structures.** Below ~5 px both lobes merge into one peak (4-px poles not detected); 6-px poles fall back to the
  doublet (ΔZ error ~15 %); 8–12 px use constant fits. Fences and foliage produce many small thin rings (090).
- **Junctions.** Contour chains break at junctions (no smoothing across them); corners of the sky band and T-junctions
  are where coverage gaps concentrate.
- **Smear.** Edges smeared beyond σ_w,max ≈ 1 px fail the sharpness test by design (DA3's own edges are sharper).
- **Contacts.** DA3 renders some physical contacts (box on plinth, kiosk base) as small steps; both the depth and the
  fold channel then fire there.
- **Alignment.** DA3 edges are offset from RGB edges by about (−0.27, −0.52) px (RGB = DA3 + offset), consistently
  across images (D2 diagnostic); no correction is applied.

## 7. Open items for the integration task

Status after TASK_GEOM_INTEGRATION (G0–G6; details `docs/geometry_channels.md` §6.1): items 1, 2, 3, 4, 5, 8, 9, 10
**resolved** (closed as stated there; 9 as a documented gap), 6 and 7 **reopened, not fixed** (geom_v2 candidates), 11
**recorded** (geom_v2 candidate). Under min with the Laplacian channel (geom_v1), the depth channel sets w on ≤ 0.03 % of
the pixels; its deliverable role there is the strengths (r, ΔZ).

1. *[resolved G2: not stable, no shift]* **Alignment shift** (decisions log, D2): (a) confirm the shift on more images; (b) if stable, a versioned Stage A
   option for a fixed subpixel shift of D; (c) a band-coverage check against RGB edges for the depth and fold channels.
2. *[resolved G2: sky_band depth_and_mask; band_min 2]* **Band coverage vs RGB.** Band radii (2–3 px) against the RGB edge positions and the sky-mask boundary (070: the
   depth band sits ~1 px inside; 375 of 457 uncovered band pixels).
3. *[resolved G3: ΔZ₀ 0.10 m, weight stays ΔZ-based]* **ΔZ₀ / binary weights.** The weight is nearly binary on real edges (section 4); decide between a knee and a binary
   weight with w_min.
4. *[resolved G6: min of weights; strengths documented]* **Fusion on strengths** (fold J, depth r) vs weights; depth strength is ∞ at sky edges and must saturate.
5. *[resolved G4/G5: kept as a third channel, pool_radius 2]* **Retiring the Laplacian channel** once fold + depth cover its true positives.
6. *[reopened G4, not fixed: geom_v2]* **Local σ_geo** for the fold channel's additive null (48-px tiles), as specified there.
7. *[reopened G4, not fixed: geom_v2]* **Narrow-face veto** (fold channel, capped veto: `narrow_face`, `veto_undecidable`).
8. *[resolved G3/G4: keep the relaxation]* **Fence clutter policy** (090): thin doublet rings in dense meshes.
9. *[resolved G3 as a documented gap; crease rescue on in geom_v1]* **Moderate-crease gate** (fold channel, `docs/fold_channel.md` §7).
10. *[resolved G4/G5: labelled audit]* **Recall against tri_veto4 on true geometry classes** (annotated fold / step / limb / texture), not only the pinned ROIs.

11. *[TASK_GEOM_V2 V3: option `geometry.depth.short_thin: off | strong | linked` built, not proposed; short_thin stays off, V5 included]* DA3 resolves railing bars and chain-link mesh only as 1–4 px fragments; accepting them (`strong`) relaxes 3.3 % of the open step-gap px and closes no unit, so the component minimum is not the real limit, and thin structures stay the Laplacian's domain. **Component minimum on thin contours** (TASK_GEOM_INTEGRATION G4, record only, no code). Of the 45 labelled step
    gaps still open after the G4 re-evaluation, 25 (15 733 of 25 089 px) are dropped at the component stage: their best
    candidates passed every gate, but 84 % sit in contour components < 5 px (median 4 px; `min_component_px` 5), 82 %
    are `thin` (doublet fits) and their r is above τ_high (r P50 0.10, component max r P50 0.11). These are fragmented
    thin contours on fences and railings (090, 073, batch2 075, 067). The other open step gaps: τ_low 6, significance
    6, D-3 tests 5, sign 2, no candidate 1. In geom_v1 the Laplacian channel covers them.

The step-vs-kink model selection in `two_sided_fit` is not needed as an integration item: the 079 and 075 checks
(section 5.3) found steps in DA3's depth, not folds misread as steps.

## 8. Where things live

| what | where |
|---|---|
| field, candidates | `illum_baseline/geom/candidates.py`: `depth_kernels`, `depth_field`, `depth_candidates` |
| channel | `illum_baseline/geom/depth.py`: `DepthParams`, `DepthChannel`, `jump`, `chains`, `running_median`, `depth_null` |
| shared fit | `illum_baseline/geom/twosided.py`: `two_sided_fit(..., gap_mode, usable)`, `one_sided_constant` |
| calibration | `scripts/calibrate_depth.py` → `illum_baseline/geom/data/depth_calibration.json` |
| config | `geometry.depth` (`illum_baseline/config.py`: `DepthCfg`); `configs/candidate_depth_only.yaml`, `configs/candidate_depth_fold.yaml` |
| viz | `illum_baseline/viz_depth.py` (sheet), `illum_baseline/diag_depth.py` (`diag-depth`) |
| ROIs | `scripts/depth_rois.py` (pinned depth ROIs), `scripts/fold_null_rois.py` (fold targets, texture areas) |
| reports | `scripts/depth_d2_report.py`, `depth_d3_report.py`, `depth_d5_report.py`, `depth_pre_d6_checks.py` → `baseline_output/diag/depth_v1/` |
| tests | `tests/test_depth_field.py`, `test_depth_candidates.py`, `test_depth_channel.py`, `test_depth_stage_b.py` |
| decisions | `docs/geometry_channels.md` §7 (rows D0–D6) |

## Appendix A. Development record (D0–D5)

The phase records below were written as each milestone closed; where they differ from sections 1–8, the sections
above are current (e.g. σ_w,max 1.02 px, centre_tol 0.4, the additive default).

### A.0 D0 design decisions (summary)

The D0 draft fixed the module map and data flow above, the context's laziness (`[depth]` reads only S, Ω, ρ, ρ_floor,
ρ_f, K and the sky probability; `two_sided_fit` is called with `usable = Ω` and `fallback = False` so the dense fit is
never built), the evaluation classes (pinned ROIs as a stand-in until annotations exist) and the flags: sample mask Ω,
thin-structure retries driven by the channel, the fold channel's contamination formula on the depth null, sky fraction
on S, contours as linked candidates, n not canonicalized (toward the near side), the null grid, ΔZ₀ = 0.10 m and
symmetry off, `[depth, fold]` with the fold_v2 settings.

### A.1 D2 record (candidates, calibration)

- **Gradient floor** (replaces k_g × median |∇Dd|): g_n = |∇Dd| / max(ρ_σ, ρ_floor), σ̂_g = median(g_n over Ω) / 1.1774
  (Rayleigh median), valid where g_n > 3 σ̂_g. A crossing is kept when g_n, linearly interpolated between its two
  pixels, is above the floor. No recall loss on synthetic steps with r ≥ 5 % (edge coverage 1.000 with and without);
  on the real images 24–38 % of Ω is valid (g_n is far from Rayleigh there: real slopes and edges, not noise).
- **ŝ_q sample:** Ω at least R = ⌈4σ⌉ px from the sky and the border, *not* the gradient-valid pixels (with the new
  floor those are mostly edges).
- **q on the sky:** not zeroed. ρ = 0 there (the far limit), so a sky boundary is a doublet like any step; D-6 decides
  what it means (ΔZ = inf, sky_near).
- **D-3 implementation:** sign changes of q between 4-neighbour pixel pairs with at least one pixel in Ω; merged within
  0.5 px (single linkage, mean position); n at x₀ bilinear and renormalized; lobes are the runs of one sign *contiguous
  with x₀* along ±n (step 0.25 px, to ±8 px), peaks parabola-refined.
- **Centring stage (added):** |s_p + s_m| ≤ s_sep/3. Beyond a step's core the detrended n flips sign
  (−∇ρ_{σ_L} dominates) and the trilateral filter leaves small wiggles 5–7 px out; a crossing there borrows the step's
  own lobe ~5 px away (|s_p + s_m|/s_sep ≈ 0.54) and, with the prior smear bound, passed D-3 as a parallel ghost line
  (hundreds per step). Real unambiguous occlusions: max 0.29. Residual noise doublets 6–10 px from large noisy steps
  carry the flipped n, so their two-sided jump is negative (D-4 `sign` rejects them; checked in D2).
- **σ_w,max = 0.83 px** (P95 of σ_w,c over 576 unambiguous occlusions on the 10 images; P50 0, P99 1.05), so
  s_hi = 3.0 px. Real DA3 edges at network resolution are sharper than an ideal pixel-sampled step (s_sep 2.45 vs
  2.72 at P50): DA3 slightly over-sharpens.
- **λ̂_Δρ:** synthetic noise ≈ 98–106 (median-based; MC mean 106–119), flat in k ∈ {11, 13, 16}; real images 660–1391.
- **ρ_f vs ρ (070 floor ROI):** ρ_f leaves 0 candidates in the floor ROI after the height test, raw ρ 193 after all of
  D-3. ρ_f stays the default.
- **Shared fit:** `two_sided_fit(..., gap_mode, usable)` and the new `d_rho_x0_se`, `d_rho_min_span` fields (section 3).
- Scripts: `scripts/calibrate_depth.py` → `geom/data/depth_calibration.json`; `scripts/depth_d2_report.py` →
  `baseline_output/diag/depth_v1/d2/` (counts, ablation, DA3-vs-RGB alignment).

### A.2 D3 record (jump, gates, magnitude, weights)

Changes to the D2 parameters (decided after D2): `centre_tol` 0.4; σ_w,max = P99 (now 1.02 px, see below).

- **Lobes on h.** q = h/ρ_σ with h = nᵀHn; ρ_σ varies across a doublet, which drags the far lobe outward for large
  jumps (far peak at −1.76 σ_tot·px for r = 1, −2.1 for r = 5, −3.3 for r = 100; s_sep then exceeds s_hi and
  steps in front of distant backgrounds failed sharpness). Lobe positions are now found on h (peaks at ∓σ_tot for any
  r); heights stay q at those peaks; the doublet magnitude is 1.03 |h|_lobe s_sep². Recalibrated σ_w,max: P99 1.02 px
  (P95 0.77).
- **Nearest peak.** Inside a thin structure the near lobes of both edges form one run; the lobe is the nearest local
  peak reaching ½ of the run maximum (8-px poles: s_sep 8.0 → 2.9).
- **Jump (D-4).** Per side: plane at k_c, m = 8 with the contamination check (internal retry m = 4) → m = 4 → constant
  fit → doublet. Obstacles: D-3 candidates whose own doublet estimate passes τ_low (noise doublets ≤ 1e-3);
  a side whose window lies beyond one uses the constant fit on the clear span. Fit modes and `d_next_±` recorded.
- **Gates (D-5).** sky_near → sign (Δρ* > 0) → significance (|Δρ*|/(SE√λ̂) > 3.29; doublet: min lobe/ŝ_q) → r > τ_low →
  components (link radius 3 px, ≥ 5 distinct pixels, one member r > τ_high). Sky edges pass with r = inf.
- **Magnitude (D-6).** Plane-aware running median along mutual-nearest-neighbour chains; ΔZ = (f_n/300)(1/ρ̂₋ − 1/ρ̂₊);
  lower bound with √λ̂-inflated SEs (inf when unbounded); focal bracket recorded.
- **Weight (D-7).** w_c = max(0.05, ΔZ₀/(ΔZ + ΔZ₀)); bands max(2, ⌈σ_w,c⌉ + 1); strength r (inf at sky edges);
  arrays `dZ` (float32, NaN off the bands, inf at sky), `q_over_s` (float16, clipped), `edge_mask`.

**Tests 6–14** (`tests/test_depth_channel.py`): single-step recall 1.000 (r 5–200 %, smear ≤ 0.8, against slope);
box silhouette recall 0.910; noise-free ΔZ error ≤ 1.3e-14, smear ≤ 3 px ≤ 0.9 %; scale/focal invariance exact; sky
inf / w_min and sky_near; horizon lower bound flagged (unbounded); poles 6–20 px detected (6 px doublet, ΔZ error
15 %; 8–12 px constant fit; 16–20 px planes; 4 px not detected: both lobes merge into one peak); limb bias +7 to +32 %
(within the cylinder's own depth change over k px, 10–44 %); λ̂_Δρ −11 % vs the Monte-Carlo inflation.

**Real images** (`scripts/depth_d3_report.py` → `baseline_output/diag/depth_v1/d3/d3_report.json`): 0.56–0.79 s per
image after the context, normals never built. Significance, not τ_low, binds for 73–100 % of candidates (r_gate P50
0.035–0.34; P90/P10 7–51). 090: a chain-link fence (DA3 renders the mesh as depth blobs) gives 1714 thin accepted
candidates in small rings (w < 0.5 on 7.9 % of the image).

### A.3 D4 record: integration and saved outputs

**Registry and config.** `CHANNELS["depth"] = DepthChannel`. `geometry.depth` (`DepthCfg`) mirrors `DepthParams` field
for field (tested), with validation and nested `--set` (e.g. `--set geometry.depth.null_model=additive`). The gate null
is `geometry.depth.null_model` (the fold channel's key name): `multiplicative` (default; λ̂_Δρ) or `additive`
(λ_n SE² + σ_geo,ρ²); both significance columns are always recorded. Configs: `configs/candidate_depth_only.yaml`
(`[depth]`, run `depth_v1`), `configs/candidate_depth_fold.yaml` (`[depth, fold]` with the fold_v2 settings, run
`depth_fold_v1`); `configs/default.yaml` was unchanged by this task (geometry off; geom_v1 since TASK_GEOM_INTEGRATION G6).

**Records.** The JSON `params.geometry.depth` block is written only for runs that include the depth channel, so every
earlier geometry run reproduces its record bit for bit (`fold_only`, `tri_veto4_fold`, `fold_v2`,
`tri_veto4_fold_v2` in `tests/test_baseline_repro.py`). Channel arrays keep float32 / float16 / bool dtypes; float64
arrays are stored as float16 as before (fold's `r_n` unchanged).

**Saved outputs** (`stage_b/<run>/`, arrays in the stored frame at network resolution):

| key / file | content |
|---|---|
| `depth_w_net` (float16) | w_depth ∈ [w_min, 1] |
| `depth_strength` (float16) | r = ΔZ/Z_near at rasterized pixels (max over overlapping bands), 0 elsewhere. **inf at sky edges** (and where the far side is unbounded): any future fusion on strengths must treat inf as saturated |
| `depth_dZ` (float32) | ΔZ in metres at rasterized pixels (max over bands), NaN elsewhere, inf at sky edges / unbounded far sides |
| `depth_q_over_s` (float16) | signed q/ŝ_q, clipped to ±6.5e4 |
| `depth_edge_mask` (bool) | the accepted candidates' rounded pixels |
| `<stem>_depth_candidates.parquet` | every D-3 candidate (merged crossings above the gradient floor): position, n, lobes (q at the h peaks), s_sep, centring, σ_w,c, k_c; for D-3 survivors the jump (fit modes per side, d_next±, ρ̂±, SE±, Δρ*, its SE, the min-span gap, planarity / m_used / rms / exclusion of the m = 8 fit), the doublet and its ratio, `signif` (multiplicative) and `signif_add`, r, r_gate, sky shares, flags (`sky_edge`, `sky_near`, `thin`, `planes_cross`, `r_lower_bound`, `lower_bound`), component id / size / max r, smoothed ρ̂±, ΔZ with its focal bracket, w_c, band, `accepted` and `reason` ∈ {polarity, height, sharpness, centring, symmetry, sky_near, sign, signif, rel, component, accepted} |
| `<stem>_depth_null.parquet` | the null grid: u, v, Δρ*, SE, planar± |
| JSON `channels.depth` | ŝ_q, σ̂_g, λ̂_Δρ (raw and clamped), σ_geo,ρ, `null_model`, σ_w,max, s_lo, s_hi, counts per stage and per fit mode, r_gate P10/P50/P90, the share where significance binds, focal f_n and σ_f/f, shares w < 0.5 / 0.9, params, context and runtime |
| manifest | `depth_*` columns (counts, ŝ_q, λ̂, σ_geo,ρ, null_model, σ_w,max, r_gate P50, share binding, shares of w, runtime) for depth runs only |

A `[depth]` run never builds the dense fit or the normals: no `n_hat` / `sigma_theta` / `normal_valid` arrays,
`normals_computed: false` (tested). `[depth, fold]` saves the elementwise min of the two channels' w (tested).

### A.4 D5 record: visualization, runs, evaluation

**Tools.** Depth sheet `stage_b/<run>/viz_depth/<stem>.png` (`illum_baseline/viz_depth.py`; rendered by Stage B and
`viz` for runs with the depth channel; panel 4 recomputes the field from the Stage A cache). Profile tool
`python -m illum_baseline diag-depth --config configs/candidate_depth_only.yaml --stem S --u U --v V`
(`illum_baseline/diag_depth.py` → `stage_b/<run>/diag_depth/<stem>_u<U>_v<V>.png`): ρ_f and q along n with the lobes,
the side planes extrapolated to x₀, the windows used, Δρ*, r, ΔZ, the doublet ratio, both significances.

**Runs** (10 images): `depth_v1` (multiplicative), `depth_v1_add` (`--set geometry.depth.null_model=additive`),
`depth_fold_v1`. Report: `scripts/depth_d5_report.py` → `baseline_output/diag/depth_v1/d5/d5_report.json`.

- Share w_depth < 0.5 per image: 1.0–7.9 % (multiplicative), 1.0–8.1 % (additive); `depth_fold_v1` combined 3.0–11.4 %;
  tri_veto4 10.5–33 %, tri_veto4_fold_v2 10.9–34.6 %. Runtime 0.57–0.83 s per image after the 0.3–0.4 s context.
- Per class (share of w_depth < 0.5; multiplicative / additive): step 0.809 / 0.809, limb 0.997 / 0.997, fold
  0.089 / 0.089, texture 0.0010 / 0.0029. The fold share comes from the 080 side face edge R target (0.57): its upper
  half is the side face's outer silhouette against the background (r P50 2.2, ΔZ P50 4.9 m), an occlusion that the
  fold targets also list; every other fold / crease target is ≤ 0.03.
- Spot checks: all 16 met under both nulls (070 floor 0.000 vs ≤ 0.041; sky boundary median 0.05; 079 grass
  0.000 / 0.003, sunlit 0.003 / 0.006; 079 silhouettes median 0.05–0.08; 079 box fold 1.00; 083 limbs 0.11 / 0.13;
  083 rim 1.00; creases 1.00; runtime ≤ 0.83 s).
- Additive null: +3 to +33 % accepted candidates; synthetic box silhouette recall (test 7) 0.910 → 0.950. Decision
  rule (texture ≤ multiplicative + 0.02 on the 070 floor and both 079 grass ROIs, step / limb targets kept): **met**.
  Proposed, not switched (the default stays multiplicative until decided).
- 090 (chain-link fence, clutter case): 1995 of 2904 accepted candidates in the fence region (u 0–150, v 0–280), 89 %
  thin (doublet) rings; w_depth < 0.5 on 24.5 % of the region vs 3.1 % elsewhere. Not tuned.
