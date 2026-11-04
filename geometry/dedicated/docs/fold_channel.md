# The fold channel: surface-normal discontinuities as a geometric-edge weight

Status as of 2026-10-31 (TASK_FOLD_CHANNEL F0–F5; TASK_FOLD_V2 F6–F9, sections 4.5, 4.6, 5–7):

- **Code:** `illum_baseline/geom/` (pure numpy/scipy, float64, network resolution, upright frame). Architecture and the
  channel interface: `docs/geometry_channels.md`.
- **Runs:** `configs/candidate_fold_only.yaml` (Stage B run `fold_only`) and `configs/candidate_lap_fold.yaml`
  (run `tri_veto4_fold`: the tri_veto4 Laplacian weight and the fold weight, combined by the elementwise minimum);
  with the capped veto of section 4.6 (v2): `configs/candidate_fold_v2.yaml` (run `fold_v2`) and
  `configs/candidate_lap_fold_v2.yaml` (run `tri_veto4_fold_v2`).
- **Default pipeline:** `configs/default.yaml` is geom_v1_1 (fold v2 with the crease rescue and `site: hessian_peak`, depth and Laplacian channels; geom_v1 from TASK_GEOM_INTEGRATION G6 until the TASK_GEOM_V2 wrap-up; `docs/geometric_edges.md`); before that it was tri_veto4 with the geometry channels off. `baseline_v0`
  and `tri_veto4` reproduce bit-for-bit, and `channels: [laplacian]` reproduces `tri_veto4` bit-for-bit (tests).
- **Visualization:** the fold sheet and the profile tool, `docs/viz_guide.md`.

## 1. What it does

The tri_veto4 weight responds to curvature of the disparity (a Laplacian). Folds, where two planar faces meet at an
angle, are only partly captured: DA3 rounds them over a few pixels and their Laplacian overlaps with texture
(079 lit/shaded box fold: w = 0.50–0.59 in tri_veto4, against 0.35–0.45 in baseline_v0). The fold channel measures
the fold **directly**, as the angle between the surface normals on the two sides, and turns it into a weight:

1. **Normals.** Fit a plane to the filtered disparity around every pixel (edge-aware) and convert it to a unit
   normal with its covariance (N1, N2).
2. **Candidates.** Where the normal field turns fast (structure tensor), keep thin ridge lines (F2.1).
3. **Two-sided fits.** On each candidate, fit one plane on each side, beyond the fold's rounding (F2.2, F2.3).
4. **Fold angle and gate.** The angle Δθ between the two normals, tested against the noise and roughness of this
   image with a χ² statistic, not by a threshold on Δθ (F2.4, F2.5).
5. **Veto.** Reject smooth curvature (cylinders, spheres): a fold's angle does not grow when the windows move
   farther out, a curved surface's does (F2.6).
6. **Weight.** Convert the angle into the worst-case shading jump J and a weight w_c, and paint it around the fold
   line (F2.7, F2.8). Combined with the Laplacian channel by the minimum.

The channel uses depth only (no RGB) and adds no DA3 pass.

## 2. Notation

| symbol | meaning |
|---|---|
| x = (u, v) | pixel; u column, v row (downward); network resolution, upright frame |
| K: f_x, f_y, c_x, c_y | intrinsics at network resolution: f_x = fx·s_x, f_y = fy·s_y (GeoCalib, active focal), principal point ((W−1)/2, (H−1)/2) |
| r(x) = (x̄, ȳ, 1) | camera ray, x̄ = (u − c_x)/f_x, ȳ = (v − c_y)/f_y; camera frame X right, Y down, Z forward |
| D, ρ = 1/D | DA3 canonical depth and its disparity (global scale irrelevant) |
| ρ_f | the trilateral-filtered disparity of tri_veto4 (`detector.trilateral`, σ_s = 3, σ_r = 1%, σ_g = 3); all geometry uses ρ_f |
| R_s = 9 | the trilateral support ⌈3σ_s⌉; ρ_f is biased within R_s of a feature and exact beyond |
| S, Ω | sky mask, its complement |
| α, β | fitted disparity at a point and its gradient (disparity per px) |
| ĝ = β/α | relative gradient (per px) |
| μ, n̂ | plane vector, unit normal toward the camera |
| Σ_n, σ_θ | normal covariance (3×3), angular standard deviation sqrt(trace Σ_n) (rad) |
| e₁ | unit vector across a candidate fold line (in the image); side "+" is +e₁ |
| x₀ | subpixel position of a candidate on its fold line |
| k, m, w_t | window skip from x₀ along e₁, window length along e₁, window half-width along the line (px) |
| k₁, k₂ | the skip of the measurement windows and of the veto windows (k₂ = 2 k₁) |
| Δθ | angle between the two sides' normals |
| T | χ² statistic of the normal difference; λ̂, λ_n, σ_geo its calibrations |
| C | concentration ratio Δθ(k₁)/Δθ(k₂) |
| J | worst-case shading jump in units of the total incident radiance Φ |
| w_c, w_fold | a candidate's weight; the rasterized fold weight |

## 3. Normals from the filtered disparity

### 3.1 N1: dense edge-aware plane fit

At every pixel x, over the offsets d with ‖d‖ ≤ R_n = ⌈3σ_n⌉ = 9 (253 offsets):

$$
(\alpha, \beta) = \arg\min \sum_d W(x, d)\,\big[\rho_f(x+d) - \alpha - \beta\cdot d\big]^2 .
$$

**Weights.** The trilateral weights of tri_veto4 evaluated on ρ_f (the same code, `detector.tangent_plane_weights`):

$$
W(x, d) = e^{-\lVert d\rVert^2/2\sigma_n^2}\;
e^{-\frac12\left(\frac{e(x,d)}{\sigma_r \max(\rho_f(x),\,\rho_{\mathrm{floor}})}\right)^2}\;
\mathbf 1[x + d \notin S],
\qquad e(x, d) = \rho_f(x+d) - \rho_f(x) - g(x)\cdot d .
$$

e is the deviation of the neighbour from the tangent plane with gradient g. Neighbours across a step or a sharp fold
deviate by far more than σ_r = 1% and get no weight, so the fit stays on the pixel's own surface.

**The tangent-plane gradient g.** A Gaussian-derivative gradient (σ_g = 3, support 12 px) sees a depth step up to 12 px
away; e is then large for the pixel's own side as well, and the fit failed in a band along every step (Σ W = 3–5
within ±3 px of a 1.5× step, below the minimum 5). The gradient therefore starts from **side windows**:

1. Eight half-disks of radius R_n, one every 45° (offsets with d·ω_j ≥ 0), spatial Gaussian weights, sky excluded;
   an unweighted-range LS plane of ρ_f(x+d) − ρ_f(x) on (1, d) in each.
2. g₀ = the slope of the half-disk with the smallest relative residual Σ w r² / (Σ w ρ_f(x)²) among those with
   Σ w ≥ 5. At least one half-disk lies on the pixel's own side of any straight edge, so g₀ is that side's slope.
3. The weighted fit runs with g₀, then once more with g = β of that fit (`refit = 1`).

Result: Σ W ≥ 35 everywhere except the pixel straddling a step (synthetic steps of 1.1–3×), and planes stay exact.

**Solving.** The 3×3 weighted normal equations M θ = b with the moments of A = [1, dᵀ]. The regressand is centred on
the tangent plane: since ρ_f(x) + g·d lies in the column space of A, fitting e instead of ρ_f(x+d) gives the same
estimator, (α, β) = (ρ_f(x), g) + θ_e, but lets Σ W r² be accumulated in one pass without cancellation
(Σ W r² = Σ W e² − θ_eᵀ b_e, with e of the size of the residuals rather than of ρ).

**Validity.** Invalid: sky; within R_n of the sky or the border; Σ W < 5; condition number of M above 10⁶ (d in px;
the median is 8.5).

**Covariance (sandwich).** The weights are not inverse variances, so

$$
\mathrm{Cov}(\alpha, \beta) = s^2\, M^{-1} (A^\top W^2 A)\, M^{-1},\qquad
s^2 = \frac{\sum W r^2}{\sum W}\,\frac{N_{\mathrm{eff}}}{N_{\mathrm{eff}} - 3},\qquad
N_{\mathrm{eff}} = \frac{(\sum W)^2}{\sum W^2}.
$$

Checked by Monte Carlo (200 draws of i.i.d. relative noise 1e-3 on ρ_f): empirical/analytic σ_θ = 0.997 (slanted
plane) and 0.998 (grazing floor).

### 3.2 N2: plane vector and normals

**Derivation.** A plane n·X = d₀ seen through the pinhole X = Z r has 1/Z = (n/d₀)·r = m·r: disparity is linear in
the ray, hence affine in (u, v). With ρ = m·r and r = ((u − c_x)/f_x, (v − c_y)/f_y, 1):

$$
\partial_u \rho = m_x/f_x,\quad \partial_v \rho = m_y/f_y,\quad
\rho = m_x \bar x + m_y \bar y + m_z .
$$

Matching ρ(x + d) ≈ α + β·d at pixel (u, v) gives m = α μ with the dimensionless **plane vector**

$$
\mu = \big(f_x \hat g_u,\; f_y \hat g_v,\; 1 - (u - c_x)\hat g_u - (v - c_y)\hat g_v\big),\qquad \hat g = \beta/\alpha,
\qquad \hat n = -\mu/\lVert\mu\rVert .
$$

n̂ points toward the camera (μ·r = ρ/α = 1 > 0, so n̂·r < 0). The plane distance is d₀ = 1/(α‖μ‖) in the units of
1/ρ, and ‖μ‖ = 1/(|r| cos α_inc), large on grazing surfaces.

**Properties.**

| property | why | tested |
|---|---|---|
| exact on planes | an affine ρ_f is fitted exactly by any weighted LS (all samples on the plane); ρ_f is exact farther than R_s from a feature | random planes and box faces: ≤ 1e-12 |
| checks | constant ρ: ĝ = 0, μ = (0, 0, 1), n̂ = (0, 0, −1). Floor ρ = (v − c_y)/(f h): ĝ_v = 1/(v − c_y), μ = (0, f/(v − c_y), 0), n̂ = (0, −1, 0), d₀ = h | ≤ 1e-12 |
| global depth scale cancels | ĝ = β/α is unchanged by ρ → cρ | c ∈ {0.37, 2.9}: ≤ 1e-9 |
| position-independent on a plane | for ρ = a + b·x, at any x: α = a + b·x, β = b, so μ = (f_x b_u, f_y b_v, a + b·c)/α, whose direction does not depend on x | (used below) |
| **intrinsics errors cannot create folds** | the previous line holds for any K: both sides of one plane give the same direction n̂, so Δθ = 0 and Δρ* = 0 whatever f_x, f_y, c | κ ∈ {0.78, 1.22}: Δθ ≤ 1e-9 |
| focal sensitivity | f → κf scales μ_x, μ_y by κ and leaves μ_z alone, so the tangent of the angle to the optical axis scales by exactly κ (near the principal point that is the angle to the ray) | exact to 1e-9 at every pixel |

So a wrong focal length changes fold **magnitudes** (section 4.7) but cannot make a plane look folded; the focal
uncertainty therefore never enters the gate.

**Covariance.** δμ = J δĝ with J = [[f_x, 0], [0, f_y], [−(u − c_x), −(v − c_y)]], Σ_ĝ = Cov(β)/α² (the α
uncertainty is neglected: about 1% of the β term for these windows), and with P = I − n̂n̂ᵀ

$$
\Sigma_n = P\, J \Sigma_{\hat g} J^\top P / \lVert\mu\rVert^2,\qquad \sigma_\theta = \sqrt{\operatorname{trace}\Sigma_n}.
$$

**Why ρ_f and not ρ.** DA3's 7-px grid ripple (relative amplitude a ≈ 0.2%) tilts normals: median error 0.62–0.70°
when the fit runs on raw ρ, 0.03° on ρ_f (synthetic plane with the ripple; 0.002° on a floor).

On the 10 images 87–92% of the pixels have a valid normal; median σ_θ = 0.03–0.21°.

## 4. The fold measure

### 4.1 Candidates (F2.1)

Structure tensor of the normal field, summed over the three components:

$$
T_s = \sum_{c} G_{\sigma_T} \ast \big(\nabla n_c \nabla n_c^\top\big),
$$

with Gaussian derivatives at σ_d = 1 px on valid normals only (pixels whose 9×9 derivative support touches an
invalid normal are excluded) and normalized convolution at σ_T = 1.5 px. Its largest eigenvalue λ₁ and eigenvector
e₁ give the **rotation rate** r_n = sqrt(λ₁) (rad/px) and the direction across the fold. Candidates: non-maximum
suppression of r_n along e₁ (bilinear samples at x ± e₁), r_n > 3·median(r_n), subpixel x₀ from a parabola along e₁.
e₁ is oriented with angle in (−90°, 90°] (e₁_u > 0, or e₁ = (0, 1)); side "+" is +e₁.

r_n is a rate, not a magnitude: it says where the normal turns, not by how much in total (a 90° fold rounded over
12 px and a 30° fold rounded over 4 px have similar r_n). The magnitude comes from the two-sided fits.

### 4.2 Rounding width and skip (F2.2)

Sample n̂ along x₀ + s e₁ for s = −12…12 px, θ(s) = angle(n̂(s), n̂(−12)) normalized by θ(12); the rounding width is
the 10%–90% distance of the running maximum of that profile, clamped to [2, 12] px (12 if a sample is invalid or the
profile is flat). The measurement windows start beyond it:

$$
k_1 = \max\!\big(k_{\min},\ \lceil \text{width}/2\rceil + k_{\mathrm{margin}}\big),\qquad k_{\min} = 6,\ k_{\mathrm{margin}} = 2 .
$$

The measured width is DA3's rounding plus the dense fit's own transition (synthetic ideal folds: 4.5–10.3 px; the
079 box edges reach the 12-px clamp). k₁ is 6–8.

### 4.3 Two-sided windows and the contamination check (F2.3)

$$
\mathcal W_\pm(k) = \{x_0 + s e_1 + t\, e_t : \pm s \in [k, k + m],\ |t| \le w_t\},\qquad e_t = e_1^\perp,\ m = 8,\ w_t = 4 .
$$

ρ_f is sampled on the integer (s, t) grid by bilinear interpolation (exact for affine functions). A sample is
excluded if any of its four bilinear corners is sky, has an invalid normal or lies outside the image; a window with
more than 20% excluded is invalid. Each window gets an ordinary LS plane ρ_f = α± + β±·(x − c±) about its sample
centroid c±, with Cov = s²(AᵀA)⁻¹, s² = RSS/(N − 3), and N2 normals n̂±, Σ_n± at c±.

**Contamination check.** k_min = 6 lies inside the trilateral support R_s = 9, where ρ_f is slightly biased next to
the fold. A window fit is accepted when

$$
\frac{\mathrm{rms}(\text{residual})}{\alpha} \le \max\!\big(3\,\hat\varepsilon,\ 1.25\, b_{\mathrm{tri}}(k, m),\ Q_{0.95}\big),
$$

- ε̂: the median rms/α of the null windows (section 4.5) of this image;
- Q_0.95: their 95th percentile;
- b_tri(k, m): the rms/α that the trilateral bias alone puts into a window at skip k, measured on 58 noise-free
  single-feature scenes (folds 15–120° at 4 orientations, convex and concave; occlusion steps of depth ratio 1.1–3)
  at the channel's own candidates, using only windows that lie on one surface ≥ R_s + 1 px inside the border, made
  non-increasing in k (`scripts/calibrate_fold.py`, `geom/data/b_tri.json`):

  | k | 6 | 7 | 8 | 9 | 10 | 11 |
  |---|---|---|---|---|---|---|
  | b_tri(k, m = 8) | 8.4e-5 | 8.4e-5 | 8.4e-5 | 1.2e-5 | 6.3e-6 | 5.6e-6 |
  | b_tri(k, m = 4) | 1.0e-4 | 1.0e-4 | 1.0e-4 | 1.5e-5 | 7.9e-6 | 6.6e-6 |

- **Margin 1.25.** With noise of rms ε the window rms is about sqrt(b² + ε²), at most 1.054 b while b ≥ 3ε, so a
  bias-limited window right next to a feature would fail an unmargined check by a few percent; 1.25 also covers
  geometries outside the finite calibration set (held-out 45° fold: every window at k ≥ 6 passes).

A failing window is retried with m/2 = 4; below m_min = 4 it is invalid and the side **falls back** to the dense
normal at the pixel nearest the nominal window centre (flag `fallback`).

**Delta rho\* (record, for the depth channel).** The smallest gap between the two fitted planes along the normal line,

$$
\Delta\rho^* = \min_{s \in [-k_1, k_1]} \big|\hat\rho_+(x_0 + s e_1) - \hat\rho_-(x_0 + s e_1)\big| ,
$$

closed form because the gap is affine in s; zero with `planes_cross` when the planes intersect in that span (a fold).
Its standard error, ΔZ/Z_near = |Δρ*|/min(ρ̂₊, ρ̂₋) at the gap point, and `depth_jump` = ΔZ/Z_near > 2% and
|Δρ*|/SE > 3 are recorded; they do not affect w_fold.

### 4.4 Fold angle (F2.4)

$$
\Delta\theta = \operatorname{atan2}\big(\lVert \hat n_+ \times \hat n_- \rVert,\ \hat n_+\cdot\hat n_-\big) \in [0, \pi]
$$

(atan2 rather than arccos: accurate for small and large angles). Exact on planes: Δθ is exact to 6e-15 rad on
synthetic folds of 30–120° with windows beyond the trilateral support (k = R_s + 2 = 11); at the channel's k₁ = 6–8
the residual trilateral bias leaves ≤ 0.154°.

### 4.5 The gate (F2.5)

**Why not a threshold on Δθ.** Under pure noise each normal scatters with per-axis σ, the difference with per-axis
σ√2, and Δθ is Rayleigh-distributed with mean σ√2·sqrt(π/2) = sqrt(π) σ ≈ 1.77σ, never zero. σ also varies by an
order of magnitude across an image (grazing surfaces, near the sky). A fixed Δθ threshold would flag noise on grazing
surfaces and miss weak folds on well-measured ones.

**χ² statistic.** With n̄ = normalize(n̂₊ + n̂₋), a tangent basis B = [b₁, b₂] at n̄, δ = Bᵀ(n̂₋ − n̂₊) and
Σ± = Bᵀ Σ_n± B:

$$
T = \delta^\top (\Sigma_+ + \Sigma_-)^{-1} \delta \;\sim\; \chi^2_2 \text{ under the null if the covariances are right,}
\qquad P(T > t) = e^{-t/2}.
$$

For p_fa = 10⁻³ the gate is t = 2 ln(1/p_fa) = 13.8.

**Calibration on the image itself.** The window covariances describe noise only. The **null sample** is every
stride-4 pixel with a valid structure tensor, using its own e₁ and k = k_min, through the same pipeline (contamination
check, fallback) as the candidates; most pixels are not folds, so medians over it are robust. Three null models:

| null model | statistic | per-image calibration | accept |
|---|---|---|---|
| **multiplicative (default)** | T | λ̂ = median(T_null)/(2 ln 2) (the χ²₂ median is 2 ln 2), clamped to [0.5, 10⁴] | T > 13.8 λ̂ |
| additive | T_add = δᵀ(λ_n(Σ₊ + Σ₋) + σ_geo² I)⁻¹δ | σ_geo solved so that median(T_add, null) = 2 ln 2 | T_add > 13.8 |
| additive_local | as additive | σ_geo per 48-px tile (≥ 50 null samples, else the global value), bilinear between tile centres | T_add,local > 13.8 |
| additive_cand (v2) | as additive | σ_geo per candidate from the null samples within 32 px of x₀ (≥ 50, else 48 px, else global), excluding those within k_min + m px of the candidate's own ridge component | T_add,cand > 13.8 |

- **What the data say.** On synthetic noisy planes the inflation is λ ≈ 90 (the trilateral filter and bilinear
  sampling correlate the noise; λ̂ matches the Monte-Carlo E[T]/2 to 8%). On the 10 real images λ̂ = 625–1480: real
  surfaces are not planar at the 20-px window scale (DA3 undulation, true curvature), σ_geo ≈ 0.9–6.3°. The upper
  clamp was raised from 50 (task spec) to 10⁴ in F2; at 50 the gate passed 60–80% of the valid candidates.
- **λ_n** (additive nulls): the noise-only inflation, Monte-Carlo E[T]/2 on noisy synthetic planes
  (`geom/data/lambda_n.json`); it is tied to the window skip:

  | k_min | noise 2e-4 | 5e-4 | 1e-3 | used |
  |---|---|---|---|---|
  | 6 (fold) | 94.3 | 93.2 | 86.6 | **93.2** |
  | 11 (planned depth channel) | 123.0 | 121.2 | 111.6 | 121.2 |

- **Effective angle threshold.** Δθ_gate ≈ sqrt(13.8 λ̂)·σ for the multiplicative model: it scales with each
  candidate's own σ and spreads by P90/P10 = 8–24 within an image (median 13–133°). The additive models separate
  noise from roughness and spread less (P90/P10 2.9–9.4), but on rough grass a σ_geo set by the smoother surfaces is
  too permissive (section 6.3). The default stays multiplicative.
- **Local roughness (v2, F8).** With the capped veto, three gate arms were compared under the switching rule (every
  ROI's share of w_fold < 0.5 ≤ the multiplicative arm + 0.02, 079 box folds w ≤ 0.25, 083 bollard median w_fold
  ≥ 0.95, at least half of the confirmed crease targets accepted):

  | arm | ROIs | box rows | bollard | creases | accepted Δθ < 5 / 5–20 / 20–50 / > 50° (10 images) |
  |---|---|---|---|---|---|
  | A0 multiplicative | reference | 0.216 / 0.215 | 1.00 | 0 / 2 | 74 / 46 / 363 / 7270 |
  | A1 additive_cand | sunlit grass 0.026 (> 0.000 + 0.02) | 0.216 / 0.215 | 1.00 | 0 / 2 | 64 / 142 / 647 / 7539 |
  | A2 additive_local, 24-px tiles, ≥ 30 | ok | **1.00 / 1.00** (box folds lost) | 1.00 | 0 / 2 | 18 / 21 / 102 / 2015 |

  No arm meets the rule; the gate stays multiplicative. A1 is the closest: it doubles the 5–50° acceptances, but a
  local null is dominated by the object around the candidate, not by the candidate's own faces: around the 088 rock
  ridge the null samples lie on the rest of the faceted rock (σ_geo,cand 24.8° against 1.2° globally, T_add falls from
  49.7 to 1.3), around the 076 façade ledge on a busy façade (10.6° against 2.5°). The share of pixels with w_fold in
  [0.5, 0.9) stays ≤ 0.003 per image in every arm.

### 4.6 The veto against smooth curvature (F2.6)

**Ratio for constant curvature.** If the normal turns at a constant rate κ (rad/px) across the line, a window over
s ∈ [k, k + m] measures the normal near its centre s = k + m/2, so Δθ(k) ≈ κ(2k + m) grows with k; an ideal fold has
the same Δθ for every k beyond its rounding. With the veto windows at k₂ = 2k₁ and the same m, w_t:

$$
C = \frac{\Delta\theta(k_1)}{\Delta\theta(k_2)} \approx
\begin{cases} 1 & \text{fold} \\ \dfrac{2k_1 + m}{2k_2 + m} = \dfrac{2k_1 + m}{4k_1 + m} & \text{constant curvature} \end{cases}
\qquad (k_1 = 6: 0.625,\ k_1 = 8: 0.60).
$$

Accept if C ≥ C_min = 0.85; applied only to gate-passed candidates. Measured: synthetic cylinders C = 0.52–0.63 (all
217 deep-interior candidates vetoed), box folds C ≥ 0.95.

**The k₂ windows get no contamination check**, only the 20% exclusion rule: curved surfaces fail the check, and an
invalid k₂ window means `no_veto` (accept), which let 703 of 913 cylinder candidates through in F2. The cost: a fold
next to a face narrower than about 2k₂ + m is vetoed, because the k₂ window then reaches the next face (079 box region:
41 gate-passed candidates under the multiplicative null, 64 under the additive one, 58 of these along the plinth edge;
none has both adjacent faces ≥ 2k₂ + m wide).

**Capped veto (v2, `veto_mode: capped`; TASK_FOLD_V2 F7).** The fixed k₂ = 2k₁ reaches past a narrow face into the
next one (079 plinth edge: Δθ(k₁) 30°, Δθ(k₂) 94°, C 0.32), and invalid k₂ windows near the image border, the sky or a
step's 1-px invalid-normal band (a 1-px band becomes 2 excluded samples under the corner rule: 2/9 = 22% > 20%) were
accepted blindly as `no_veto` (2–12% of accepted). The capped veto measures how far each side can go:

- d_next± = distance along ±e₁ from x₀ to the first obstacle over the window's width (|t| ≤ w_t): an invalid normal or
  the outside of the image beyond the candidate's own rounding (|s| > width/2 + 1), or another gate-passed candidate
  (NMS pixels dilated by 1 px) beyond the whole rounding band (|s| > width + 1: DA3-rounded folds have parallel sibling
  ridges inside the band, which would otherwise cap almost every candidate).
- k₂± = clamp(min(2k₁, d_next± − m − 2), k₁, 2k₁); the constant-curvature ratio for these skips is
  C_curv = (2k₁ + m)/(k₂₊ + k₂₋ + m), and the threshold C_min,eff = 1 − 0.4 (1 − C_curv) (0.85 at k₂ = 2k₁, k₁ = 6).
- Decidable iff C_curv ≤ 0.8 and both k₂ windows are valid: accept iff C ≥ C_min,eff; C > 1/C_min,eff is flagged
  `C_high` (no rejection).
- Undecidable: accept only if both k₁ sides are strictly planar (planar fit, not fallback, rms/α ≤ max(3ε̂,
  1.25 b_tri(k₁, m)), i.e. without the Q_0.95 term): `narrow_face`; otherwise reject as `veto_undecidable`.
- `no_veto` (table column, `n_no_veto`) keeps its name in capped mode but means only "gate-passed with an invalid
  capped k₂ window" (076: 102, against 35 invalid uncapped k₂ windows under the fixed veto, because capped windows sit
  closer to obstacles). Unlike the fixed veto, such a candidate is never accepted on that ground: it goes to the
  `narrow_face` / `veto_undecidable` rule.

Calibration (`scripts/calibrate_veto.py`, synthetic cylinders' deep interior and single folds, k₂ forced over all
skips): cylinder C tracks C_curv (P95 ≈ C_curv + 0.01, max + 0.03), fold C ≥ 0.97. With the coefficient 0.4 the
margins to the cylinder maximum / fold minimum are 0.19 / 0.12 at C_curv 0.625 and 0.10 / 0.07 at 0.775, positive up to
C_curv 0.925. The decidability limit is **0.8** rather than the task's 0.7: one-side-capped skips (the common case next
to a narrow face) have C_curv 0.75–0.77, and at 0.7 they were all undecidable (the strict-planarity rule then
rejected most real folds: accepted −17% to −77% per image).

### 4.7 Magnitude: the shading jump J (F2.7)

**Bound.** For a Lambertian surface of albedo 1 lit by radiance L(ω), the irradiance is
E(n) = ∫ L(ω) max(n·ω, 0) dω. Across the fold,

$$
|E(\hat n_+) - E(\hat n_-)| \le \Phi \cdot \sup_{\omega}\big|\max(\hat n_+\!\cdot\omega, 0) - \max(\hat n_-\!\cdot\omega, 0)\big|,
\qquad \Phi = \int L\, d\omega .
$$

The supremum: if Δθ ≥ 90°, ω = n̂₊ lights the + face fully and the − face not at all, giving 1; if Δθ < 90°, the
largest n̂₊·ω on the − face's terminator n̂₋·ω = 0 is sin Δθ (interior points of the region where both faces are lit
give (n̂₊ − n̂₋)·ω, which grows toward that boundary). Hence

$$
J = \sin\!\big(\min(\Delta\theta, \pi/2)\big),
$$

attained by a point source grazing the − face: the bound is **sharp**. Assumptions: Lambertian faces; for a **convex**
fold both faces see the whole hemisphere and J is the exact worst case; for a **concave** fold the faces shade and
light each other, and J covers the direct light only.

**Against the chord.** The naive Lipschitz bound is the chord c = ‖n̂₊ − n̂₋‖ = 2 sin(Δθ/2). J ≤ c always, and much
smaller for large angles:

| Δθ | 5° | 15° | 30° | 60° | 90° | 120° |
|---|---|---|---|---|---|---|
| c = 2 sin(Δθ/2) | 0.087 | 0.261 | 0.518 | 1.000 | 1.414 | 1.732 |
| J = sin(min(Δθ, 90°)) | 0.087 | 0.259 | 0.500 | 0.866 | 1.000 | 1.000 |
| J / c | 1.00 | 0.99 | 0.97 | 0.87 | 0.71 | 0.58 |
| w_c | 1.000 | 0.593 | 0.377 | 0.243 | 0.215 | 0.215 |

**Focal diagnostics.** n̂± recomputed with f·(1 ∓ σ_f/f) give [J_lo, J_hi] per candidate (recorded, never used in w).

### 4.8 Weight and rasterization (F2.8)

$$
w_c = \max\!\Big(w_{\min},\ \frac{J_0}{\max(J - J_{\min}, 0) + J_0}\Big),\qquad J_{\min} = \sin 5^\circ = 0.0872,\ J_0 = 0.25,\ w_{\min} = 0.05 .
$$

Folds below 5° get w = 1; a 90° fold gets 0.215 (table above). Accepted candidates (gate and veto passed) that form
8-connected components smaller than 5 px are dropped. w_fold(x) is the minimum w_c over accepted candidates within
the band radius b_c = max(2, ⌈width/2⌉) of x (min-filters over the few distinct b_c), 1 elsewhere. The channel's
strength is J, max-filtered with the same bands.

The convexity record: the sign of the un-vetoed L/ρ at x₀ (disparity ridge: convex, +1).

## 5. Parameters and sensitivity

| parameter | value | role | sensitivity / evidence |
|---|---|---|---|
| dense_fit σ_n, σ_r, σ_g | 3 px, 1%, 3 px | N1 window, range, initial gradient | σ_r as in tri_veto4; σ_g only matters for the fallback when no side window qualifies |
| dense_fit g_init, refit | side_window, 1 | tangent-plane gradient | `gaussian, 0` reproduces F1: Σ W fell to 3–5 at steps and silhouette candidates sat 3–5 px off |
| cand σ_d, σ_T, rate factor | 1, 1.5 px, 3 | candidate ridges | not varied |
| k_min, k_margin | 6, 2 | window skip | k_min = 11 (arm B): folds both planar 0.996 vs 0.997, box folds 0.74 vs 0.82, step depth-jump recall 0.995 vs 0.04; 6 kept for folds, 11 planned for the depth channel |
| m, m_min, w_t | 8, 4, 4 | window size | not varied |
| planarity_factor, b_tri_margin, null_rms_quantile | 3, 1.25, 0.95 | contamination check | 1.25: section 4.3 |
| p_fa | 10⁻³ | gate | T threshold 13.8 λ̂ |
| lambda_clamp | [0.5, 10⁴] | λ̂ | at the task's 50 the gate passed 60–80% of valid candidates (real λ̂ 625–1480) |
| null_model | multiplicative | gate | section 6.3 |
| λ_n, local tile, min samples | 93.2, 48 px, 50 | additive nulls | λ_n per k_min (section 4.5) |
| veto_k_factor, C_min | 2, 0.85 | veto | cylinders 0.52–0.63, folds ≥ 0.95: wide margin both ways |
| veto_mode | fixed (v2 runs: capped) | veto | section 4.6 |
| veto_curv_coef, veto_curv_max | 0.4, 0.8 | capped veto threshold and decidability | calibrated (section 4.6); at 0.7 most one-side-capped real folds were undecidable |
| cand_radius, cand_radius_wide, cand_min_samples | 32, 48 px, 50 | additive_cand null | not adopted (section 4.5) |
| J_min, J₀, w_min | 0.0872, 0.25, 0.05 | weight | table in section 4.7 |
| band_min, min_component_px | 2, 5 | rasterization | not varied |
| depth_jump_rel | 2% | record only | — |

## 6. Results (10 images)

### 6.1 Synthetic checks (tests)

| test | result |
|---|---|
| planes, checks, depth scale, focal, Monte-Carlo σ_θ, ripple | section 3 |
| exact fold angles (6 folds 30–120°, 4 orientations, convex and concave) | ≤ 6e-15 rad at k = 11; ≤ 0.154° at k₁ |
| wrong K on planes | Δθ ≤ 1e-9, Δρ*/ρ ≤ 1e-12 |
| veto | cylinder deep interior: 0 of 217 pass; nearer the rim / limb: 32 / 217 of 847 pass; box folds C ≥ 0.952 |
| noise null | λ̂ / Monte-Carlo inflation 0.92 |
| depth_jump | none on 1436 single-fold candidates; silhouette recall 0.75 (reported) |
| narrow faces | 12-px face: k₁ fallback; 18-px face: m/2 retry; flags set |

### 6.2 Runs (null model multiplicative)

| | fold_only | tri_veto4_fold |
|---|---|---|
| share w < 0.5 | 0.026–0.069 | 0.109–0.347 (Δ vs tri_veto4 +0.000 to +0.025) |
| share w < 0.9 | 0.027–0.069 | 0.169–0.406 (Δ +0.000 to +0.015) |
| λ̂, ε̂ | 625–1479, 1–3e-4 | same |
| candidates → valid → gate → veto → component | 2068–5671 → 1780–5236 → 743–1246 → 629–1243 → 585–1208 | same |
| median J of accepted | 0.96–1.00 | same |
| runtime per image (CPU) | context 4.6–6.8 s + fold 0.6–0.9 s | same |

Spot checks:

| check | requirement | result |
|---|---|---|
| 079 lit/shaded box fold, combined w (rows 230 / 254, min over u 180–215) | ≤ 0.45 | **0.216 / 0.215** (tri_veto4 0.586 / 0.498, baseline_v0 0.449 / 0.353) |
| 083 bollard, median w_fold (u 175–205, v 260–330) | ≥ 0.95 | 1.00; the 5.9% below 0.5 are all on the base contour |
| 070 floor, combined share w < 0.9 | ≤ tri_veto4 + 0.01 | 0.031 (tri_veto4 0.031) |
| 079 grass, combined share w < 0.5 | ≤ tri_veto4 + 0.03 | 0.267 (0.267); sunlit grass 0.512 (0.504) |
| fold runtime on CPU | < 5 s | 5.3–7.7 s with the context (the dense fit); GPU port planned |

### 6.3 Null models

Share of w_fold < 0.5 (rule for switching the default: ≤ multiplicative + 0.02 in every ROI and the 079 box folds kept):

| ROI | multiplicative | additive | additive_local |
|---|---|---|---|
| 070 floor | 0.000 | 0.000 | 0.000 |
| 079 grass (u 5–110 × v 5–110, u 5–60 × v 360–495) | 0.000 | 0.009 | 0.000 |
| 079 grass, sunlit (u 110–290 × v 5–85) | 0.009 | 0.054 | 0.033 |
| 083 bollard | 0.059 | 0.059 | 0.058 |
| 079 box rows 225–260: accepted (w at rows 230 / 254) | 60 (0.216 / 0.215) | 60 (same) | 60 (same) |

Neither additive model meets the rule (sunlit grass), so the default stays multiplicative. taiwei_cui_075 (a
cylindrical kiosk), pixels with w_fold < 0.5 by the class of the candidate that sets them
(`baseline_output/diag/fold_f4/075_wfold_classes.png`):

| null | share | fold | rim | limb | texture |
|---|---|---|---|---|---|
| multiplicative | 0.069 | 9330 | 3487 | 0 | 294 |
| additive | 0.135 | 16263 | 7500 | 307 | 1640 |
| additive_local | 0.036 | 2988 | 3072 | 141 | 621 |

(step = a depth jump between the window centres; fold = no step, both windows planar; rim = step, one side planar;
limb = step, neither planar; texture = no step, not both planar.) Under the multiplicative null, most "fold" pixels
are vertical lines on the kiosk's curved surface (possibly panel seams) and the façade behind it; additive_local
removes both and keeps the base contour, the silhouette and the curb.

### 6.4 Version 2: capped veto (TASK_FOLD_V2)

Synthetic (tests): cylinder deep interior 0/217 accepted (as before); passes near the rim / limb 5 / 61 (fixed veto
32 / 217); box folds with a decided veto C ≥ 0.990, 18 undecidable; faces between two folds:

| face width | fixed veto: edges accepted | capped veto |
|---|---|---|
| 12 px | both (by accident: a fallback normal and k₂ windows beyond both folds) | both (340 / 352, 352 / 352) |
| 18 px | neither (C 0.54) | both |
| 24 px | 281 / 353, 311 / 353 | both, all |
| 32 px | both | both |

Real targets (coverage of the polyline by accepted candidates, `scripts/fold_null_rois.py` TARGETS):

| target | fixed | capped |
|---|---|---|
| 079 plinth edge | 0.00 | **0.83** |
| 080 side face edge L | 0.88 | **1.00** |
| 076 plinth top edge L; 090 plinth edge upper | 0.18; 0.20 | 0.18; 0.20 |
| other plinth edges (076 bottom L, top R, bottom R; 090 lower) | 0.00 | 0.00 (gate: the k₁ windows already span the narrow face, out of scope) |
| 080 side face edge R | 0.00 | 0.00 (no candidates: a 2.3× depth step's invalid normals void the structure tensor) |
| 076 façade ledge, 088 rock ridge (creases) | 0.00, 0.03 | 0.00, 0.03 (gate) |

The former `no_veto` acceptances split into decided passes (3–79 per image), decided vetoes (0–22), `narrow_face`
(0–1) and `veto_undecidable` (10–68). The 083 bollard limbs lose 125 of 325 acceptances (steps; acceptable). Losses
beyond them are mostly rim-class silhouettes (44–179 per image), which the Laplacian channel and the planned depth
channel cover.

Runs `fold_v2` / `tri_veto4_fold_v2` (multiplicative gate, capped veto):

| | fold_v2 | fold_only (v1) |
|---|---|---|
| cand → valid → gate → veto → component | e.g. 079: 3915 → 3712 → 1043 → 888 → 822 | 079: … → 972 → 908 |
| narrow_face / veto_undecidable / C_high per image | 0–35 / 28–288 / 11–530 | — |
| share w_fold < 0.5 | 0.024–0.062 | 0.026–0.069 |
| depth_jump share of accepted | 0.02–0.28 | 0.02–0.34 |
| combined w < 0.5 (tri_veto4_fold_v2), Δ vs tri_veto4 | +0.000 to +0.025 | +0.001 to +0.025 |
| runtime (context + fold, CPU) | 5.5–6.2 s | 5.3–7.5 s |

ROIs and spot checks are unchanged by the capped veto: 070 floor, 079 grass 0.000, 079 sunlit grass 0.009 → 0.000,
083 bollard 0.059 (median w_fold 1.00), 079 box rows 0.216 / 0.215.

## 7. Limitations and open items

Status after TASK_GEOM_INTEGRATION (`docs/geometry_channels.md` §6.1): local σ_geo (item 6) and narrow faces (item 7)
**reopened, not fixed** (geom_v2 candidates; covered by the Laplacian channel in geom_v1); moderate creases (item 9)
**resolved as a documented gap** (`crease_rescue: laplacian_corroborated` on in geom_v1; 076 ledge blocked by the capped
veto); fold placement **open** (geom_v2 candidate: re-siting at the plane intersection).

- *[item 6: open (reopened G4, not fixed)]* Why TASK_GEOM_V2 V4 leaves the χ² gate alone: the open fold gaps fail it
  through noisy normals (gate failures Δθ P50 25°, T/λ̂ P50 0.68 against 13.8), and a local σ_geo would make the gate
  stricter on exactly those noisy surfaces, not looser. **Local σ_geo.** One σ_geo (or λ̂) per image treats grass and boxes alike. Neither local variant meets the switching
  rule: additive_local (48-px tiles: sunlit grass 0.033; 24-px tiles: box folds lost) and the candidate-centred
  additive_cand (sunlit grass 0.026, no crease accepted); section 4.5.
- *[item 7: reopened G4; TASK_GEOM_V2 V4 outcome: not fixed]* `geometry.fold.rescue_undecidable: corroborate` lets
  eligible gate failures whose capped veto is undecidable (the narrow-face mechanism) into the crease rescue,
  corroborated by the scale-ratio-vetoed Laplacian. 7 758 candidates are rescued that way, 6 535 of them within
  10 px of another accepted edge; it closes 1 of the 21 veto-blocked units (target ≥ 7), leaves the narrow-face unit
  15 at 0, and fails two constraints (079 grass +0.017, the 083 interior 0 → 0.021 core: the vetoed response passes the
  bollard dome and shaft). Not adopted; item 7 stays open (geom_v2: covered by the Laplacian fallback).
- *[item 7, TASK_GEOM_V2 W1 / W1b: closed, not adopted]* `geometry.fold.window_cap: next_ridge` ends a side's k1 window 2 px before the next
  NMS fold ridge (skip 6; only if the shortened m ≥ 4). It fixes faces narrower than k1 + m + 2:
  - the 070 middle riser (14 px) goes from 32 % / 48 % to 100 % accepted; the rounding width saturating at 12 px had
    pushed k1 to 8, and the riser-side window crossed the opposite fold, with σ_θ 13× the clean risers';
  - the audit's narrow-face units 15 / 37 / 92 are unchanged, and the 068 laps (10 px apart) stay out of reach.
  It meets the W1 constraints, but it was held: it caps 21 % of all candidates, and core crease coverage falls by 0.16
  (`docs/geometric_edges.md` §8.14).

  W1b (`next_fold_ridge`) counts only ridges with a valid Hessian peak, and the shortened windows get their own
  contamination thresholds. It reaches only 84 % / 82 % on the middle riser and keeps the 088 crease loss.
  **The narrow-face limit, as it stands:** a face narrower than about k1 + m + 2 ≈ 16 px gives its k1 windows no clean
  plane on the face side, and neither repair in the fold channel works:
  - A cap at the next ridge fixes the face, but only where the opposite fold is a detectable ridge. DA3 rounds narrow
    faces, so their folds often fail the Hessian height test (40 % on the 070 riser's lower fold).
  - The same cap shortens the skip of rounded single folds near parallel ridges (the 088 rock ridge), which lowers T
    below the gate there.

  The Laplacian doesn't cover this case either. On the 070 middle riser's fold pixels, geom_v1_1's Laplacian w is
  < 0.5 on 0 %; the combined w is < 0.5 on 46 %, all of it from the fold channel's accepted part. Item 7 stays open,
  with W1 / W1b recorded as not adopted.
  - *Recall gain under core w* with `corroborate` (and `site: hessian_peak`): missed true geometry 0.563 → 0.508 of the
    tri-only area, fold ROI coverage 0.817 → 0.907, fold-target RGB coverage (core) 0.698 → 0.767. *Not adopted
    because* only 1 of the 21 veto-blocked units closes, 079 grass rises +0.017 (combined) and the 083 interior changes
    (rescues on the bollard's dome top and shaft).
  - *Failure mode:* the scale-ratio-vetoed Laplacian (band term at 4σ ≈ 5 px, reach about ±10–15 px) passes smooth
    curvature at the scale of the bollard's dome, so it cannot stand in for the concentration veto there.
  - *Candidate for later (no code):* hessian_peak sharpness as the stand-in where the concentration veto is
    undecidable: a valid V2b site whose s·q peak clearly exceeds its values at ±3 px. It needs only ±4 px, so it stays
    decidable on narrow faces.
  - *Unit closure* (≥ 50 % of tri_veto4's wide relaxation strips) is structurally hard for any narrow-band channel
    (V3, V4: the fragments add 2–6 px bands inside strips of ~600 px); V5's selection rule uses per-pixel shares.
  **Narrow faces (v2).** The capped veto recovers folds next to faces ≥ 12–18 px (079 plinth edge 0.83, 080 side face
  1.00). Plinth edges whose narrow face is narrower than the k₁ windows (076, 090) still fail at the gate (out of
  scope: geometry narrower than the windows).
- *[item 9: resolved G3 as a documented gap]* **Moderate creases (v2).** The weight stays effectively binary (w_fold in [0.5, 0.9) on ≤ 0.003 of the pixels);
  neither confirmed crease (076 façade ledge 14°, 088 rock ridge 22°) is accepted by any gate arm. A local null built
  from the surroundings is dominated by the object's other geometry; a null that separates the candidate's own faces
  from their surroundings is open. The combined w of `tri_veto4_fold_v2` along the creases equals tri_veto4's (the fold
  channel adds nothing there): 076 façade ledge median 0.747 (19% of the polyline below 0.5), 088 rock ridge median
  0.252 (79% below 0.5); so the Laplacian covers the ridge but not most of the ledge, and the crease gate is not
  parked.
- **Items 6 / 7 after the labelled audit (TASK_GEOM_INTEGRATION G4): reopened, not fixed.** Reopen rule applied after
  the re-evaluation (geom_v0 + sky_band depth_and_mask + crease_rescue 8° / 3; a unit closed if w < 0.5 on ≥ 50 % of
  it). Narrow faces (item 7): unit 92 (076 plinth contact fold) closed by the crease rescue (0.81), 15 (067 picket
  front / side folds, 0.00) and 37 (070 porch nosing, labelled step, 0.32) open. Textured-surface folds (item 6): 97
  (079 fold), 124, 125 (088 creases) open (0.00, 0.00, 0.14). In geom_v1 the Laplacian channel covers all five: w < 0.5
  on 100 % of each unit, median w 0.055 (15), 0.265 (37), 0.157 (97), 0.164 (124), 0.168 (125); the fold channel alone
  0.00 / 0.32 / 0.00 / 0.00 / 0.14 (`baseline_output/diag/geom/g5/units_v1.csv`).
- *[TASK_GEOM_V2 V2 / V2b]* **Fold siting.** `geometry.fold.site: nms | plane_intersection | hessian_peak`
  (default nms). plane_intersection fails the V2 rule (convex median |offset| −0.21 px); hessian_peak (the sub-pixel peak
  of −convexity · e1ᵀHe1/ρ within ±4 px, limited to half the distance to the next accepted fold centreline, with the
  height, doublet and window-end rejections) meets it: convex median |offset| 2.84 → 1.70 px, vertical corners −2.00 →
  −0.27 px, fold-target coverage unchanged; proposed for geom_v2. The geom_gain / geom_offset relaxed shares fall
  (0.664 → 0.589, 0.250 → 0.183) because those units were drawn on the NMS band position: the band moves, geometry is
  not lost. Nosings stay at about +2.7 px (narrow treads, DA3's smoothing of stairs, RGB search limit).
- *[open: geom_v2 candidate]* **Fold placement (open).** The fold is placed at the NMS ridge of the rotation rate. On the 15 audit units labelled
  geom_offset the RGB edge lies off the NMS site along e1 (G4): vertical corners −2.0 px (toward the oblique face,
  all 5 units negative), nosings +2.9 px (up, all 5 positive), contact folds mixed (+1.2 median, −2.4 to +5.6). The
  G5 diagnostic (`scripts/geom_g5_fold_site.py`) sites the fold where the two fitted side planes intersect
  (ρ̂₊ = ρ̂₋ along e1, the k₁ fit's `s_gap` when `planes_cross`): vertical corners −2.0 → −1.05 px, nosings +2.6 → +1.2
  px, contact folds +0.8 → +0.3 (mixed); all convex candidates in the frontal-face convention −1.75 → −0.82 px, the
  plane site 0.57 px toward the oblique face from the NMS site. The plane site halves the bias on convex folds but does
  not remove it; on the narrow plinth targets the planes often do not cross within ±k₁ (contact targets 34 %) and the
  plane site is no better. RGB-edge caveat: the luminance maximum can be a shadow edge, and 12 % of the contact-fold
  samples sit at the ±8 px search limit; the labels of 161 / 173 (near 0) and 165 / 180 (> 3 px) are kept.
- **Silhouettes in the capped veto.** A step's own 1-px invalid-normal band caps k₂ at once, so silhouettes usually
  become undecidable and are rejected unless strictly planar (rim-class losses, 083 limbs). Silhouette ownership is
  the integration step's; the Laplacian keeps them down-weighted in the combined w.
- **080 side face, right edge.** No candidates at depth steps of about 2× and more (the structure tensor needs valid
  normals over its 4-px support).
- **C_high** (C > 1/C_min,eff) is a flag only: mostly thin structures narrower than the windows (084 handrails) and
  silhouettes whose k₂ window lands on another background.
- **Silhouette depth_jump recall** is 0.75 at k_min = 6 (misses sit a median 4.9 px off the edge, and k = 6 windows
  next to a step often fail the contamination check). The strict requirement (≥ 90% at k ≥ 11) belongs to the depth
  channel's tests (`docs/geometry_channels.md` section 5).
- **Curved surfaces with panel seams** (075 kiosk) produce fold candidates that pass the veto under the multiplicative
  null; possibly real creases, to be settled by annotations.
- **Evaluation.** The hook (`eval-fold`, `annotations/folds/<stem>.json`) takes polylines labelled fold, step, limb or
  texture (not_fold kept for older files) and reports fold precision / recall plus per-class recall and precision
  (`docs/viz_guide.md`, "Fold evaluation"). No image is annotated yet; point-annotator writes CSV points (labels
  material/texture/specular) and would need these polyline classes.
- **Runtime.** The context's dense fit costs about 5 s per image on the CPU; the fold measure itself 0.6–0.9 s. The
  torch/CUDA backend follows as a separate step, checked against this numpy reference.
- **n_hat frames.** Saved arrays are in the stored frame, the normal components in the upright camera frame (every
  image here has orientation 1).

## 8. Where things live

| what | where |
|---|---|
| context, N1, N2 | `geom/context.py`, `geom/planefit.py`, `geom/normals.py`, `geom/camera.py` |
| candidates, rounding width | `geom/candidates.py` |
| two-sided fits, contamination bound, χ² | `geom/twosided.py`, `geom/data/b_tri.json` |
| fold channel | `geom/fold.py` (`FoldParams`), `geom/data/lambda_n.json` |
| calibration | `scripts/calibrate_fold.py` (b_tri, λ_n; rerun when N1, the candidates or the trilateral filter change); `scripts/calibrate_veto.py` (capped veto: C_min,eff coefficient and decidability limit) |
| reports | `scripts/fold_arms_report.py` (F2 arms), `scripts/fold_null_rois.py` (ROIs, line TARGETS), `scripts/fold_f4_report.py`; v2: `scripts/fold_v2_f6.py`, `scripts/fold_v2_crease_search.py`, `scripts/fold_v2_f7_report.py`, `scripts/fold_v2_f8_report.py`, `scripts/fold_v2_f9_report.py` |
| configs | v1: `configs/candidate_fold_only.yaml` (`fold_only`), `configs/candidate_lap_fold.yaml` (`tri_veto4_fold`); v2 (capped veto): `configs/candidate_fold_v2.yaml` (`fold_v2`), `configs/candidate_lap_fold_v2.yaml` (`tri_veto4_fold_v2`) |
| viz, profile, evaluation | `viz_fold.py`, `diag_fold.py`, `geom/evaluate.py`; CLI `viz`, `diag-fold`, `eval-fold` |
| synthetic scenes | `geom/synthetic.py` |
| tests | `tests/test_geom_normals.py` (N1, N2), `tests/test_geom_fold.py` (fold measure), `tests/test_geom_stage_b.py` (Stage B), `tests/test_geom_viz_eval.py` |

Run (from `geometry/dedicated/`):

```bash
env -u PYTHONPATH uv run python -m illum_baseline stage-b --config configs/candidate_lap_fold.yaml     # run tri_veto4_fold
env -u PYTHONPATH uv run python -m illum_baseline diag-fold --config configs/candidate_lap_fold.yaml \
    --stem taiwei_cui_079 --u 197 --v 230                                                               # profile tool
env -u PYTHONPATH uv run python -m illum_baseline stage-b --config configs/candidate_lap_fold.yaml \
    --run-name lap_fold_local --set geometry.fold.null_model=additive_local                             # another null
env -u PYTHONPATH uv run python -m illum_baseline stage-b --config configs/candidate_fold_v2.yaml       # run fold_v2
env -u PYTHONPATH uv run python -m illum_baseline stage-b --config configs/candidate_lap_fold_v2.yaml   # run tri_veto4_fold_v2
env -u PYTHONPATH uv run python -m illum_baseline diag-fold --config configs/candidate_fold_v2.yaml \
    --stem taiwei_cui_079 --u 224 --v 311                                                               # v2 profile
env -u PYTHONPATH uv run python -m illum_baseline eval-fold --config configs/candidate_lap_fold_v2.yaml # with annotations
env -u PYTHONPATH uv run python scripts/calibrate_veto.py                                               # veto calibration
```
