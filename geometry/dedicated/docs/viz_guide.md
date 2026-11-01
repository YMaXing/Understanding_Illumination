# Reading the Stage B visualization panels

Each file `baseline_output/stage_b/<run>/viz/<stem>.png` is one image's diagnostic sheet. It is rendered by `illum_baseline/viz.py: render()` from saved outputs only, and can be re-rendered with `python -m illum_baseline.cli viz --run-name <run> [stems...]`.

**Default pipeline note (since 2026-10-27, `tri_veto4`, see `docs/pipeline_tri_veto4.md`):**

- `q` and `L_over_rho` are computed from the trilateral-filtered disparity.
- `q` passes through the scale-ratio veto, min(q, g·|band response|).
- The sign in panel 3 is the sign of the un-vetoed Laplacian.
- ŝ is the noise scale of the filtered response.
- The formulas below describe the `baseline_v0` detector. They still hold with q replaced by this vetoed response.

Layout: 2 rows × 4 panels, always in the **upright** frame. Panels 1–5 are in network pixel coordinates (504×378 or 378×504), the resolution the detector runs at. Panel 6 is at the original resolution, and panel 7 is a histogram.

The sheet follows the pipeline in order: input image, depth, signed detector response, response magnitude against the threshold, weight, weight on the photo, response statistics, then the numbers.

## Two ideas behind all the panels

**Why disparity.** DA3 outputs canonical depth `D` per pixel. The detector works on disparity `ρ = 1/D`. On any planar surface `ρ` is affine (linear) in pixel coordinates, whatever the plane's slant. The custom Laplacian kernels therefore give exactly zero on planes (test 5), and respond only where planarity breaks: depth discontinuities and creases.

**Why "units of ŝ".** `ŝ` is a per-image estimate of the background noise level of the response:

```
ŝ = median(q over clean pixels) / 0.6745
```

The 0.6745 factor converts a median absolute value into a standard deviation for Gaussian noise. "Clean pixels" means away from the sky and at least `R` px from the image border. The decision rule is defined in multiples of `ŝ` (`τ = k_tau·ŝ`, `ε = k_eps·ŝ`). Plotting everything in units of `ŝ`, with fixed colour limits, therefore makes sheets comparable across images and aligns the colours with the actual decision.

## Panel 1: sRGB + sky contour

**What it shows**

- The full-resolution sRGB image from Stage A (`stage_a/srgb/<stem>.png`). This is the copy the networks saw: black level 0, no white balance, an exposure gain bringing the 99th luminance percentile to 0.9, then the sRGB transfer curve.
- The cyan line is the final sky mask `S`:
  - DA3 sky score ≥ 0.5;
  - morphological open then close with a radius-1 disk;
  - 8-connected components under 50 px removed (holes are not filled).
- The title gives the sky percentage.

**Why it is included**

- It gives context for every other panel.
- It verifies the sky mask, which matters because sky is treated specially:
  - disparity is set to 0 on the sky, so the sky boundary is meant to respond strongly;
  - the eroded sky interior is forced to `q = 0`, hence `w = 1`.
- On the current dataset only 070 has meaningful sky.

## Panel 2: log₁₀ Z (metres)

**What it shows**

- Metric depth `Z = f_n · D / 300`, where `f_n = (fx·s_x + fy·s_y)/2` is GeoCalib's focal length rescaled to network resolution.
- Turbo colormap, sky in gray.
- A log scale, because depth spans about 1 to 50 m.

**Why it is included**

- It is a sanity check on DA3: boxes should sit in front of walls and floors should recede.
- It confirms whether a strong response in panels 3–4 corresponds to a real depth change or to a DA3 artifact.

**Note:** the weights do **not** depend on `f`. The detector is invariant to global depth scale (test 8). A wrong GeoCalib focal length only rescales this panel's colour scale.

## Panel 3: signed L/ρ ÷ ŝ (RdBu_r, fixed ±20)

**What it shows**

- `L` is the Laplacian of the smoothed disparity, computed with the custom separable kernels: σ = 1.25 px, odd-reflect padding, exact zero sum.
- It is normalized as `L / max(ρ_σ, ρ_floor) / ŝ`:
  - dividing by local disparity makes the response invariant to depth scale;
  - `ρ_floor` keeps the division finite next to the sky.

**Why it is included.** This is the only panel that keeps the **sign**, and the sign identifies the kind of feature:

- **Occlusion edge (depth step):** a **red/blue doublet**, blue on the near (foreground) side and red on the far side. The box outlines in 079 show this.
- **Crease:** a **one-signed band**.
  - Convex edges facing the camera (a box's corner edge) are blue, because disparity there is a ridge (the minimum of the two planes' affine disparities).
  - Concave corners (box meets floor, wall meets floor) are red, because disparity there is a valley.

The baseline deliberately does not use this distinction (no doublet or polarity tests). The panel shows what a later tier could exploit.

The ±20 limit is deliberately tight. Strong edges saturate, so weak structure near the noise level stays visible: texture, the horizontal stripes on flat ground, image-border effects.

## Panel 4: log₁₀(q/ŝ), clipped to [−1, 3]

**What it shows**

- `q = |L / max(ρ_σ, ρ_floor)|`, the magnitude of panel 3, on a log scale because it spans about four orders of magnitude.
- Two contours:
  - **cyan at q/ŝ = `k_tau`** (3): the threshold `τ`. Below it, a pixel contributes nothing.
  - **green at q/ŝ = `k_tau + k_eps`** (6): where `w = 0.5` before pooling. Above `τ`, `w = ε / (q − τ + ε)`, which equals 0.5 when `q − τ = ε`.

**Why it is included.** It shows the decision itself: which pixels cross the threshold and by how much. This is the panel to check when judging whether `k_tau` is sensible. For example, grass in 079 is covered in cyan rings.

## Panel 5: w at network resolution (grayscale, [w_min, 1])

**What it shows.** The detector's weight. White means full smoothness; black means strongly down-weighted. It is computed as:

1. **Threshold:** `q_eff = max(q − τ, 0)`.
2. **Pool:** `q_pool` is the maximum of `q_eff` over a 29-px disk, radius `r = ceil(sqrt(σ² + σ_w²)) + 1 = 3`.
3. **Weight:** `w = max(w_min, ε / (q_pool + ε))`, and `w = 1` wherever `q_pool = 0`.

**Why it is included**

- It is the detector's raw output before any resampling (`w_net` in the `.npz`, in the stored frame).
- Its shapes reveal the pooling. The round dots in textured areas are single isolated pixels above threshold, each widened to the disk size.
- Pooling exists because DA3 smears boundaries by about 1 px, so the weight is widened to cover that smear. It is done after thresholding, because pooling first would raise the noise floor.

## Panel 6: red overlay at original resolution

**What it shows**

- The actual deliverable: the `w` stored in the `.npz`. It is bilinearly upsampled with half-pixel centres to 1023×768, clipped to `[w_min, 1]`, and shown upright.
- It is overlaid on the sRGB with red opacity `alpha = (1 − w) / (1 − w_min)`: transparent at `w = 1`, fully red at `w = w_min`.

**Why it is included**

- **Alignment:** it shows that the delivered full-resolution map lines up with the image. Test 10 checks this numerically.
- **What gets relaxed:** it is the most direct way to judge whether the red regions are real geometry or texture.
- **Shadows:** the detector sees only depth, so cast-shadow edges should stay clear unless DA3's depth reacts to them. For example, the rock's shadow in 088 is untouched.

## Panel 7: histogram of q/ŝ (log–log)

**What it shows.** The distribution of `q/ŝ` over non-sky pixels. Bins are log-spaced on [1e-2, 1e4]; values outside are clipped into the end bins. Dashed lines sit at 1, at `k_tau` (3) and at `k_tau + k_eps` (6).

**Why it is included.** It tests the noise model behind the threshold.

- If the response were pure noise, the histogram would peak near 0.67, the median by construction of `ŝ`, and vanish soon after 3.
- The visible shoulder, and a long tail to 10³, are real non-planar structure and DA3 artifacts. This is why about 20% of pixels pass the threshold on real images, rather than the 0.27% a Gaussian would give.
- The mass to the right of the cyan line is exactly what gets thresholded.

**Two artifacts to ignore**

- The tall bar at the far left is a pile-up: every value below 0.01 is clipped into the first bin.
- The "N px with q = 0 omitted" count in the title (about 30–60 px) is mostly not true zeros. These are tiny values (below about 3e-8) that round to 0 in the float16 file; this was verified on 070 and 088. On images with sky, the count also includes the sky interior, where `q = 0` exactly.

## Panel 8: text box

The per-image numbers behind the other panels. All are also in `<stem>.json` and in `manifest.parquet`.

| entry | meaning |
|---|---|
| `s_hat` | noise scale `ŝ`; "(clamped)" if `s_hat_min`/`s_hat_max` applied |
| `tau`, `eps` | `k_tau·ŝ`, `k_eps·ŝ` |
| `rho_floor` | `rho_floor_frac` × the 1st-percentile disparity; the lower bound of the normalizer |
| `r`, `R` | pooling / sky-band radius (3), kernel half-width `ceil(truncate·σ)` (5) |
| `sky_frac`, `n_bad` | sky fraction; count of non-finite or non-positive depths outside the sky |
| `w<0.5`, `w<0.9`, `w<1` | fractions of network-resolution pixels below each level |
| `f_active`, `f unc`, `f_n` | GeoCalib focal (upright full-res px, per-image mode), its uncertainty (px and %), focal at network res |
| `vfov`, `roll/pitch` | GeoCalib field of view and gravity angles |
| `median Z` | median metric depth outside the sky |
| `k_tau`, `k_eps`, `w_min`, `sigma`, `sigma_w` | detector parameters of this run |
| `s_x`, `s_y` | resize factors `W_n/W_up`, `H_n/H_up` (they differ by about 0.1%) |
| `orientation` | TIFF Orientation tag (1 for every image in this dataset) |

Use the text box to tie a sheet to its manifest row, and to spot unreliable images. For example, 070's 22% focal uncertainty means panel 2's absolute depths are shaky, while its weights are unaffected.

## Quick reading recipe

1. In panel 6, find a red region.
2. In panel 3, check what it looks like there:
   - a red/blue doublet or a one-signed band means geometry;
   - speckle means texture or DA3 noise.
3. In panel 4, see how far above the cyan contour it sits. That tells you whether raising `k_tau` would remove it.

---

# The fold sheet and the fold profile tool

Runs with the fold channel (`configs/candidate_fold_only.yaml`, `configs/candidate_lap_fold.yaml`; see
`docs/fold_channel.md`) also write `stage_b/<run>/viz_fold/<stem>.png`. It is rendered by
`illum_baseline/viz_fold.py: render_fold()` from saved outputs only (`<stem>.npz`, `<stem>.json`,
`<stem>_fold_candidates.parquet`, `<stem>_fold_null.parquet`) and re-rendered by
`python -m illum_baseline viz --config <that config> --run-name <run>`, which draws the detector sheet too when the
Laplacian channel ran.

Layout: 3 × 3 panels plus a text box, always **upright**. Panels 1–8 are in network pixel coordinates; panel 9 is at
the original resolution. The sheet follows the channel in order: result on the photo, normals, their uncertainty,
where the normals turn, how much they turn, the gate, the veto, the weight, the combined weight.

**Candidate segments.** Each candidate is drawn as a short segment along its fold line (perpendicular to e₁),
centred at its subpixel position x₀.

## Panel 1: accepted folds on the sRGB

- Accepted candidates coloured by the fold angle Δθ(k₁) in degrees (plasma, 0–120°).
- **Dashed** segments: accepted with `no_veto` (the veto windows were invalid, e.g. off the image).
- **Gray** segments: passed the gate but were rejected by the curvature veto.
- Use it to see *what* the channel calls a fold: box edges and plinth steps should be orange/yellow (60–100°),
  shallow creases purple; gray lines on curved objects show the veto working.

## Panel 2: normal map

- RGB = (n̂ + 1)/2, with n̂ the unit normal **toward the camera** in the camera frame (x right, y down, z forward).
  Blue is always low (n̂_z < 0). Examples: a wall facing the camera, n̂ = (0, 0, −1), is olive (0.5, 0.5, 0); ground
  seen at 45°, n̂ ≈ (0, −0.7, −0.7), is dark red (0.5, 0.15, 0.15); a face turned to the right of the image
  (n̂_x > 0) is redder, one turned to the left less red; a face turned up (n̂_y < 0) less green.
- Gray: no valid normal (sky, within 9 px of the sky or border, rare degenerate fits).
- Folds are where the colour changes abruptly; smooth colour ramps are curvature.

## Panel 3: σ_θ, log scale

- The angular standard deviation of each normal (degrees, log10 from −2 to 1).
- It is small on well-measured near surfaces and grows on grazing ones, near the sky and next to edges, where the
  fit has fewer effective samples. The gate is relative to this uncertainty, which is why a fixed Δθ threshold
  would not work (`fold_channel.md` §4.5).

## Panel 4: rotation rate r_n with all candidates

- r_n = sqrt(λ₁) of the normals' structure tensor (rad/px, log10 from −4 to 0); cyan dots are **all** candidates
  (ridges of r_n above 3× its median).
- Saturated white bands are silhouettes, where the normal jumps by more than 1 rad within a pixel.
- r_n says *where* the normal turns, not how much: shallow but sharp creases and strong but rounded folds can look
  alike here. Panel 5 gives the amount.

## Panel 5: Δθ(k₁) at the valid candidates

- Every candidate whose two side windows gave normals, coloured by the angle between them (degrees, dark
  background). Texture and noise appear as short purple scraps; geometry as long coherent lines.

## Panel 6: gate statistic

- Log–log densities of the gate statistic for the **null sample** (gray: a stride-4 grid of ordinary pixels) and for
  the candidates (red), with the χ²₂ density (dotted) and the gate (green, 2 ln(1/p_fa) = 13.8).
- The statistic follows the run's null model: T/λ̂ (multiplicative, the default), T_add or T_add_local.
- After calibration the null sample's median sits at the χ²₂ median (1.39). A heavier tail than χ²₂ in the null
  sample is real geometry (roughness, curvature); the candidates' excess to the right of the gate is what gets
  accepted.

## Panel 7: concentration veto

- C = Δθ(k₁)/Δθ(k₂) against Δθ(k₁), for the gate-passed candidates whose veto windows were valid; blue passed,
  gray vetoed; red line C_min = 0.85.
- A fold keeps its angle when the windows move out (C ≈ 1); constant curvature gives C ≈ 0.6. A cloud near 1 is
  folds; points far below are curved surfaces or folds next to a narrow face.

## Panel 8: w_fold

- The fold channel's weight at network resolution (gray, w_min to 1): bands of radius max(2, ⌈width/2⌉) around the
  accepted candidates, each at w_c (0.215 for a 90° fold, 0.38 for 30°, 1 below 5°).

## Panel 9: combined w on the sRGB

- The delivered `w` (the min over the run's channels; Laplacian + fold in `tri_veto4_fold`) at the original
  resolution as red alpha (1 − w)/(1 − w_min), as in panel 6 of the detector sheet.

## Text box

| entry | meaning |
|---|---|
| null model | which gate the run used (multiplicative, additive, additive_local) |
| `lambda_hat` (raw) | multiplicative inflation median(T_null)/(2 ln 2) (clamped to [0.5, 10⁴]) |
| `sigma_geo` global / local | additive models: geometric roughness solved from the null sample; local: per 48-px tile (min–max, number of tiles) |
| `lambda_n` | noise-only inflation used by the additive models |
| `eps_hat`, `null rms Q` | median and 95th percentile of the null windows' relative residual (contamination check) |
| `null` | valid / total null samples |
| candidates by stage | cand → valid (two-sided fits usable) → gate (with the counts under all three nulls) → veto (no_veto) → component (accepted); fallback sides; accepted with `depth_jump` |
| `w_fold < 0.5`, `< 0.9`, combined | shares of pixels |
| parameters | k_min, m, w_t, p_fa, C_min, J_min, J0, w_min, b_tri_margin, dense-fit gradient init |
| `f_active`, `f unc`, `f_n`, orientation | GeoCalib focal (full-res px), its uncertainty, focal at network res, TIFF orientation |
| runtime | context (dense fit) and fold-channel seconds for this image |

## Fold profile tool

`python -m illum_baseline diag-fold --config configs/candidate_lap_fold.yaml --stem <stem> --u <U> --v <V>`
recomputes the fold channel for one image (same configuration as the run), takes the candidate nearest (U, V)
(upright network px, as in the sheets) and writes `stage_b/<run>/diag_fold/<stem>_u<U>_v<V>.png`:

- **Left: dense normal-angle profile** along e₁ through the candidate: the angle between the dense normal at s and at
  the far − end, with the rounding width (orange), the k₁ windows (blue) and k₂ windows (green), and lines at
  Δθ(k₁) and Δθ(k₂). A fold is a step from one plateau to another; a curved surface is a ramp that keeps rising
  through the windows (then Δθ(k₂) > Δθ(k₁) and C < 0.85).
- **Middle: the windows on the image** (k₁ blue, k₂ green, w_t = 4 px wide) with the two-sided normals n̂± drawn as
  their image-plane components (× 12 px), labelled with all three components.
- **Right: the numbers**: reason (accepted / gate / veto / component / invalid_window), Δθ(k₁), Δθ(k₂), C (and
  `no_veto`), T/λ̂, T_add, T_add_local against the gate, J with its focal bracket, w_c and band, k₁, k₂, width, how
  each side was fitted (planar m = 8 or 4, fallback, invalid), σ_θ±, Δρ* and ΔZ/Z_near with `planes_cross`,
  `depth_jump`, convexity, the k₂ normals, and the image's λ̂, σ_geo and local σ_geo.

Typical use: pick a pixel on a suspicious line in panel 1 or 8 of the fold sheet and run the tool there.
