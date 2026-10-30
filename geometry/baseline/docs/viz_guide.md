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
