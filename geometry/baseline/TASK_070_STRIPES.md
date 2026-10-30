# Task: diagnose and remove the floor stripes in image 070

## Context

- The baseline described in `BASELINE_SPEC.md` is implemented, and the run `baseline_v0` exists. **Do not change its defaults or outputs.**
- In `taiwei_cui_070` the flat alley floor shows horizontal stripes in `w`:
  - Panel 3 shows ± alternating lines along the rows. That is a ripple, not curvature.
  - The side walls show the same pattern along columns.
  - The wall facing the camera at the end of the alley is clean.
- Amplitude: about 3–20 ŝ, with ŝ = 7.8e-5. That is a relative ripple in ρ of about 0.1–0.5%.
- Planes give q ≡ 0 (test 5), so the ripple comes from DA3's output. Planes seen at a grazing angle make it visible and set its orientation, through perspective compression and the curvature of D.

Candidate sources:

- **P — posterization of the network input.** Exposure maps the 99th luminance percentile to 0.9. In 070 that percentile is set by the sky and the sunlit wall, so the shadowed ground gets very few uint8 codes. DA3 copies the band edges into depth.
- **G — DA3 artifacts locked to its pixel grid** (14-px patch grid, or upsampling in the output head).
- **T — real ground texture or micro-relief**, copied into depth.

## Working rules

- Stop after each task. Report tersely: numbers, a table, figure paths.
- Items settled in this file: apply them without asking. Flag any other borderline choice with your lean.
- Put all new behaviour behind config options that default to the current behaviour.
  - `baseline_v0` must reproduce bit-for-bit. Add a test for this.
- Stage A variants write to `stage_a/<variant>/`. Stage B run names include the variant.

## T1: Diagnostics on 070 (no fixes)

Command: `python -m illum_baseline.cli diag --stem taiwei_cui_070 --run-name baseline_v0`

Outputs go to `out_dir/diag/070/`.

1. **Floor region (ROI).**
   - Fit ρ ≈ αu + βv + γ with RANSAC on the bottom third of the image. A pixel is an inlier if the residual is below 1% of ρ.
   - ROI = the largest connected inlier component, eroded by R + r, minus pixels with q/ŝ > 100.
   - Save the mask and an overlay.
   - Rationale: the floor is exactly affine in ρ, so no depth thresholds are needed.
2. **Residual.**
   - Define e = (ρ − ρ_fit)/ρ_fit on the ROI.
   - Plot e(v) for 5 columns.
   - Compute a Welch spectrum of e along v, averaged over columns.
   - Report the peak wavelengths, and whether they change between 3 bands of v.
3. **Code-transition test.**
   - Use the uint8 sRGB network input, i.e. what DA3 saw. Mark every pixel where the code changes between vertical neighbours, in any channel.
   - Define stripe lines as the zero crossings of L along v with |L/ρ|/ŝ > 3, inside the ROI.
   - Report the fraction of stripe lines within 1 px of a code transition.
   - Report the same fraction against a null: the code map shifted by 3–10 rows.
4. **Shift test.**
   - Run DA3 on the 070 network input shifted by (7, 7) px: reflect-pad top/left by 7, crop bottom/right by 7.
   - Shift the output back and compute e on the overlap.
   - Cross-correlate e_shifted with e_original along v.
   - A peak at lag 0 means the stripes are tied to the scene (P or T). A peak at lag ±7 means they are tied to DA3's grid (G).
5. **Summary.** Write `summary.json` with: code-transition score, null score, the cross-correlation at lag 0 and at lag 7, and the peak wavelengths. Add a one-line verdict with your lean.

**STOP.**

## T2: Input-side fix (070 only)

New `preprocess` options. Defaults equal the current behaviour.

- **`exposure_mode: p99 | median_rolloff`**
  - `median_rolloff`: gain g = 0.18 / median(Y), capped at `exposure_gain_max`.
  - Then apply a global extended-Reinhard curve to the luminance gY, with the white point at the 99.5th percentile of gY.
  - Apply the curve to RGB as a luminance ratio, so hue is preserved.
  - Global and monotone only; no local operators.
- **`dither: none | tpdf`**
  - `tpdf`: before rounding to uint8, add triangular-PDF noise (the sum of two U(−0.5, 0.5) samples).
  - Use a fixed seed per image.
- **`da3_float_input: false`**
  - Implement only if the M0 notes show DA3 accepts float input without re-quantizing it. Otherwise leave it unimplemented and say so.

Variants:

| Variant | Exposure | Dither | Float input |
|---|---|---|---|
| A0 | current (p99) | none | no |
| A1 | median_rolloff | none | no |
| A2 | median_rolloff | tpdf | no |
| A3 (only if supported) | median_rolloff | — | yes |

For each variant: run Stage A on 070, then Stage B with the baseline parameters, the visualization, and the Metrics section below.

## T3: DA3 test-time augmentation (070 only)

Run this even if T1 says the stripes are tied to the scene, and report it as such.

New option `da3.tta: none | shift_flip`.

- **Passes:** 8 in total, from the shifts (0,0), (7,0), (0,7), (7,7) in (x, y) px, each with and without a horizontal flip. Shift by reflect-pad/crop as in T1.
- **Invert:** undo the flip and the shift on each output.
- **Rescale:** for each pass, scale ρ to match the unshifted pass, using the median ratio over their overlap.
- **Average:** at each pixel, average ρ over the passes that cover it. Then D = 1 / mean ρ.
- **Sky:** average the sky probability over the passes, then threshold.
- Save the per-pass outputs.

Run TTA on the A0 input, and on the best of A1–A3.

## T4: Threshold floor (Stage B only)

New option `detector.a_min`, default `null` (off). When it is set:

- τ = max(k_tau·ŝ, τ_min)
- τ_min = a_min · k₀² · exp(−σ²k₀²/2), with k₀ = 2π/λ₀
- λ₀ = `detector.lambda_ref`, default 15 px

Report τ_min / (k_tau·ŝ) per image. Try a_min ∈ {0.25%, 0.5%, 1%}.

*Flag:* this departs from the pure noise gate. *Lean:* structured DA3 artifacts count as near-noise.

## Metrics (every variant, on 070)

- **Floor ROI** (fixed from T1): fraction of pixels with w < 0.9, fraction with w < 0.5, median q/ŝ, RMS of e, and the peak spectral amplitude of e.
- **Edge retention:**
  - Reference set F = pixels with q/ŝ > 100 in `baseline_v0`, outside the ROI dilated by r.
  - Report the fraction of F with w < 0.5.
  - It must stay ≥ 0.95 × the baseline value.
- **Global:** ŝ, and the fraction of pixels with w < 0.5.

Present one table. Input changes alter ŝ, so compare absolute q as well as q/ŝ. Add side-by-side visualization sheets for the baseline and the best variant.

**STOP.**

## T5: Generalization (the 20-image subset from M4)

- Run the best 1–2 combinations.
- Report per image: the change in the fraction of pixels with w < 0.5, the change in ŝ, and edge retention. Flag regressions.
- Recommend one configuration, with your lean. **Do not change the defaults.**

**STOP.**

## Do not

- Retune k_tau, k_eps, or σ to hide the stripes. A larger σ makes coherent ripples *stronger* in units of ŝ.
- Use local tone mapping. Its halos create false edges.
- Change `baseline_v0` outputs or defaults, or any detector math other than the τ floor.
