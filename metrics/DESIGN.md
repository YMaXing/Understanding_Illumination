# Metrics v1: design (Part A of SPEC_v1.md, verbatim)

Source: `metrics/SPEC_v1.md` lines 11–200 (Part A §1–§8), copied byte for byte below. Rulings made after the spec
was written are in "Deltas from Checkpoint 0" at the end; where a delta and Part A disagree, the delta wins.

## Part A · §1 Principles

Every metric is a functional of log R given log I. Metrics are organized by pair type, and each type has a factor that must stay flat.

1. **One degree of freedom per pixel.** log L = log I − log R holds pointwise, and log I is fixed by the input. Endpoint tests and profile tests alike are functionals of log R. "R metrics" and "L metrics" name failure modes (which factor wrongly absorbed a change); they are not independent measurements.
2. **Organize by pair type, not by map.** R must stay flat across i, s and same-material d pairs. L must stay flat across m pairs.
3. **Two kinds of evidence per near pair.** Endpoints (patch medians) show whether the change went to the right factor. The profile along the segment shows shape: halos, blur and displaced edges, which endpoints cannot see.
4. **Sparse is headline, dense is diagnostic.** Dense metrics reuse the edge and region maps that weight the training losses, so they measure loss fit. The q-percentile anchor check is a diagnostic for the same reason.
5. **Ratios only.** All tests use log differences, so they are invariant to the global and per-channel scale ambiguity of R.
6. **No degenerate wins.** R = I (everything in reflectance) and R = const (everything in illumination) must both score badly on every split-sensitive headline number (§7).

## Part A · §2 Notation and measurement primitives

All quantities are natural-log differences between robust patch statistics, plus one excursion statistic on the path profile.

**Domain.** ℓI = log I, computed by the training data loader (same black level, scaling and ε clamp; imported, never reimplemented). ℓR = network output. ℓL = ℓI − ℓR. C channels (3 for RGB; 1 also supported). A pixel is invalid if I ≤ ε or within 2% of saturation in any channel, or if the loader masks it.

**Channel norm.** For a C-vector v, ‖v‖ = sqrt(mean over c of v\_c²). A grey step of size d has norm d. Diagnostics also report brightness b(v) = mean over c of v\_c, and chroma v − b(v)·1.

**Patch statistic.** A patch is a disc of radius r around the annotated point (r from the existing tool; default 4 px). P(ℓX) = per-channel median of ℓX over valid pixels in the disc. A patch is invalid if fewer than 70% of its pixels are valid. Per-channel standard error of the median: SE = 1.2533 × 1.4826 × MAD / sqrt(n).

**Endpoint differences.** ΔℓX = P\_B(ℓX) − P\_A(ℓX) for X ∈ {I, R, L}. ΔℓL = ΔℓI − ΔℓR exactly.

**Split fraction** (near i, m, b pairs): the share of the observed change attributed to illumination, by projection onto ΔℓI. The target is α = 1 on i pairs and α = 0 on m pairs. α is undefined (pair flagged) when ‖ΔℓI‖ < 2τ.

```latex
\alpha = \frac{\Delta\ell_L \cdot \Delta\ell_I}{\Delta\ell_I \cdot \Delta\ell_I}, \qquad \Delta\ell_R = (1-\alpha)\,\Delta\ell_I + \text{(orthogonal part)}
```

**Path profile.** Sample the segment between the two patch centres at 1 px steps, t ∈ \[0, T\]. At each step take the per-channel median across a perpendicular band of half-width w = 1 px. Then Gaussian-smooth along t with σ\_t = 1.5 px (raise σ\_t for textured materials). The result is p\_X(t) for X ∈ {I, R, L}.

**Excursion.** H\_X = max over t of the channel-norm distance from p\_X(t) to the line segment \[P\_A(ℓX), P\_B(ℓX)\] in C-space. H ≈ 0 for a flat profile and for a clean monotone step between the endpoint values. H > 0 for halos, overshoots, blur bumps and displaced edges. H\_I is the input's own excursion; H\_I > τ flags a structured path (a seam or thin dark line).

**Noise guard.** A pair is low-SNR if the RMS over channels of sqrt(SE\_A² + SE\_B²) on ℓI exceeds τ/3. Low-SNR pairs are scored, excluded from headline numbers, and counted.

**Threshold.** τ = 0.10, about a 10% ratio (the WHDR δ). Every threshold in §4 defaults to τ. Sweep τ ∈ {0.05, 0.10, 0.20}. All constants live in config.

## Part A · §3 Pair taxonomy and annotation schema

Five pair types. i, m and b exist; s and d are new. Near pairs (i, m, b, s) have segments ≤ 80 px and get profile tests. Distant pairs (d) have segments ≥ 150 px and are endpoint-only.

| Type | Segment crosses | Material | Illumination | Must stay flat | Status |
| --- | --- | --- | --- | --- | --- |
| i | one illumination edge | same | different | R | existing |
| m | one material edge | different | same | L | existing |
| b | one edge that is both | different | different | neither | existing |
| s | no edge | same | same | R (L mirrors it) | new |
| d | anything | labelled | labelled | R if same material; L if same illumination | new |

**Fields on every pair:** id (existing `t=<timestamp>` ids kept), image, type, A and B as (x, y) in full-resolution pixel coordinates, r, confidence ∈ {sure, unsure}, split ∈ {eval, train}, overlay\_seen (set by the tool), annotator, created, note, schema\_version.

**Type-specific fields:**

- i: lit\_end ∈ {A, B}. For migrated pairs, infer it as the end with the brighter b(P(ℓI)).
- s: lighting ∈ {lit, shadow}; pool ∈ {random, targeted}; gap\_ref (free text naming the known gap or failure-review pair, targeted pool only).
- d: material ∈ {same, different, unsure}; illumination ∈ {same, different, unsure}.

**Placement rules** (repeated in the annotation guide):

- Near pairs cross exactly one boundary; s pairs cross none. Patch discs must not touch the crossed boundary.
- m and s pairs must not cross a depth step or fold. Geometric crossings are also computed from geom\_v1\_1, for stratification only.
- Avoid fine meshes and patterns, specular highlights, and saturated or near-black patches.
- s, random pool: the tool proposes seed points uniformly over valid pixels. The annotator accepts, nudges ≤ 10 px, or rejects (rejections logged with a reason), then places the partner 20–80 px away with no edge crossed.
- s, targeted pool: the annotator places pairs near known boundary gaps (for example the recall-failure segments from the failure review), at varying distances from the gap. Overlays are allowed and expected.
- d pairs are annotator-chosen. Prioritize same-material pairs in different illumination (anchor test) and same-material pairs that fall in separate material regions.
- Overlays (detector edges, material regions) are off by default, except for targeted s. Model predictions are never shown in annotation mode.

**Splits.** Pairs that feed the planned lit-dark pair-annotation loss are `split=train` and are never scored. The metrics code refuses `split=train`. Default: split by image (§8).

**Migration of existing i/m/b pairs.** Lossless; the original files stay untouched. New fields get defaults: confidence=sure, split=eval, overlay\_seen=unknown. Apply the failure-review ruling: 070 m pair t=1794165897383 becomes b.

## Part A · §4 Per-pair metrics

A scored pair passes only if both its endpoint error E and its profile excursion H are ≤ τ. Failing i and m pairs are then binned by split fraction.

| Type | Endpoint error E (failure name) | Profile H (failure name) | Pass | Headline |
| --- | --- | --- | --- | --- |
| i | ‖ΔℓR‖ (R-leak: reflectance absorbed part of the shadow) | H\_R (L-edge fidelity: L blurs, halos or displaces I's edge) | E ≤ τ and H ≤ τ | yes |
| m | ‖ΔℓL‖ (L-leak: illumination absorbed part of the material change) | H\_L (R-edge fidelity) | E ≤ τ and H ≤ τ | yes |
| b | none | none | none | no |
| s | ‖ΔℓR‖ (leak offset; ΔℓL mirrors it) | H\_R (leak halo along the path) | E ≤ τ and H ≤ τ | random pool only |
| d, same material | ‖ΔℓR‖ | none (endpoint-only) | E ≤ τ | yes |
| d, same illumination and different material | ‖ΔℓL‖ | none | E ≤ τ | no (drift diagnostic) |

**Continuous companions.** For i and m: the leaked fraction E / ‖ΔℓI‖. For all types: E and H as raw values.

**Split bins for failing pairs.** For i pairs use α; for m pairs use β = 1 − α (target 1 in both cases):

- reversed: < 0
- full leak: 0 to 0.2
- partial: 0.2 to 1
- overshoot: > 1
- n/a: α undefined (for example chromatic m edges at equal brightness). The pass test still applies.

**b pairs (diagnostic).** Report the both-step rate (‖ΔℓR‖ > τ and ‖ΔℓL‖ > τ) and the α distribution.

**d pairs.** Only pairs whose relevant labels are both "sure" are scored. Same-material pairs are stratified by illumination label; "different" is the global-anchor test. Other label combinations are not scored.

**Per-pair flags** (all types):

- low\_snr (§2)
- invalid (a patch below 70% valid)
- structured\_path (H\_I > τ)
- geom\_cross (segment crosses a geom\_v1\_1 edge pixel)
- detector\_split (for i, s and same-material d: A and B fall in different material\_regions labels)

**Per-pair output.** id, image, type, the three Δ vectors, their norms, α, H\_I, H\_R, H\_L, pass, bin, flags. Profiles are saved for review mode.

## Part A · §5 Aggregation and headline numbers

v1 reports three headline numbers rather than one composite. An equal-weight composite over all pair types is won by R = const, which passes i, s and same-material d pairs and fails only m pairs (§7).

| Headline | Definition | Under R = I | Under R = const |
| --- | --- | --- | --- |
| Split error | mean(ER\_i, ER\_m) | ≈ 0.5 (all i pairs fail) | ≈ 0.5 (all m pairs fail) |
| Leak error | ER\_s, random pool | passes (ΔℓI ≈ 0); measures leaks, not the split | passes |
| Long-range error | ER\_d, same-material, sure labels | fails where illumination differs | passes; read alongside Split |

**Error rate.** ER\_type = the fraction of non-excluded pairs of that type that fail §4, micro-averaged over pairs across images.

**Exclusions.** Excluded from headline numbers, and counted in every report: low\_snr, invalid, confidence=unsure, split=train, all b pairs, targeted s pairs, and same-illumination d pairs.

**Continuous companions.** Mean and median of E and H per type; the mean leaked fraction for i and m.

**τ sweep.** Every headline number at τ ∈ {0.05, 0.10, 0.20}. The headline value uses τ = 0.10.

**Uncertainty.** 95% CIs from a cluster bootstrap over images: resample images with replacement, keep all their pairs, 2000 resamples, fixed seed.

**Stratified tables:**

- per image, for every type
- i by detector\_split, geom\_cross and structured\_path
- m by geom\_cross
- s by pool × lighting
- d same-material by illumination label
- split bins per type

**Optional composite.** The mean of the three headline numbers may be reported, but never alone and never as the headline.

## Part A · §6 Dense and anchor diagnostics

These are computed per image and are never headline numbers. Each one uses a map that also weights a training loss, so it measures fit to that loss. A2 is the exception: it is network-independent and checks the material loss's own assumption.

| ID | Name | Definition | Ideal |
| --- | --- | --- | --- |
| D1 | Off-edge L roughness | mean ‖∇ℓL‖ where the smoothness weight is ≥ 0.9 (away from edges; confirm the weight convention) | low |
| D2 | Shadow leak into R | Σ‖∇ℓR‖ / Σ‖∇ℓI‖ over illumination-edge pixels (current default detector) | 0 |
| D3 | Material leak into L | Σ‖∇ℓL‖ / Σ‖∇ℓI‖ over material-edge pixels not on illumination or geometric edges | 0 |
| D4 | Within-region R spread | area-weighted mean over material\_regions of 1.4826 × MAD of b(ℓR) | low; texture sets the floor |
| A1 | Anchor fit | per i pair: b(P\_lit(ℓR)) − Q\_q of b(ℓR) over the lit patch's region | 0 |
| A2 | Anchor validity | per i pair: b(P\_lit(ℓI)) − Q\_q of b(ℓI) over the lit patch's region | ≈ 0; < 0 means the percentile sits above the lit level (highlights, glossy surfaces) |

**Conventions.** Gradients are central differences on ℓ under the channel norm; invalid pixels are excluded. q ∈ {0.90, 0.95}, matching the material loss. Region labels come from the training's material\_regions. Once the linking layer exists, rerun A1, A2 and D4 with group labels.

## Part A · §7 Sanity checks

The code is accepted only when three degenerate baselines and a synthetic test suite behave as specified below. The baselines run alongside the model in every metrics run.

| Baseline | ℓR | Expected on real annotations |
| --- | --- | --- |
| all-R | ℓI | i: α = 0, all fail (full leak) except pairs with tiny ΔℓI. m: all pass. s: pass unless structured\_path. Split ≈ 0.5 |
| all-L | 0 | i: all pass. m: β = 0, all fail. s and same-material d: all pass. Split ≈ 0.5 |
| blur-L | ℓI − G\_σ ∗ ℓI, σ = 8 px | i: H\_R > τ on most pairs; E\_i small where patches lie beyond 2σ from the edge. Exercises the profile metric |

**Synthetic suite** (`tests/`). Generate an image with piecewise-constant RGB materials, a soft-edged shadow with a coloured (ISD-like) illumination ratio, mild smooth shading, and sensor-like noise. Place i, m, b, s and d pairs programmatically. Required behaviour:

- Ground-truth ℓR: every scored pair passes; α ≈ 1 on i pairs and α ≈ 0 on m pairs, within noise.
- all-R and all-L: the expected bins exactly.
- An injected ℓR step at a gap in the shadow edge: targeted s pairs near the gap fail, and E decays with distance from the gap.
- An injected halo (L blurred across the shadow): H\_i > τ while E\_i ≈ 0.
- Scale invariance: ℓR + c, global or per channel, gives identical results.
- Identity: ℓL = ℓI − ℓR to within 1e-6.
- Log-I parity: the metric's ℓI matches the training loader's to within 1e-6 on all 10 dev images.

## Part A · §8 Open decisions

v1 ships with the leaning defaults below. Each one is a config switch, so it can be revisited without code changes.

| Decision | v1 default (lean) | Alternative |
| --- | --- | --- |
| Headline structure | three numbers: Split, Leak, Long-range | equal weight per type (gameable by R = const) |
| Eval/train separation | split by image; pairs on train images may feed the pair-annotation loss and are never scored | split by pair within an image (correlated with the training signal) |
| b pairs | diagnostic only | add an L-step test to Split |
| Distant same-illumination pairs | diagnostic only | headline once inter-annotator agreement is shown to be high |
| Aggregation within a type | micro over pairs, cluster bootstrap by image, per-image table | macro over images (070 has dense far-clutter m pairs) |
| Overlays while annotating | off by default; required for targeted s; overlay\_seen recorded | always on |
| Random s seeds | tool-proposed | annotator-chosen (biased toward easy spots) |
| Pass criterion | endpoint and profile both ≤ τ | report separate rates only |
| Threshold | single τ = 0.10, noise guard at SE > τ/3 | per-pair τ\_eff = max(τ, k·SE) |
| Near/distant lengths | near ≤ 80 px, distant ≥ 150 px | fractions of the image diagonal |

## Deltas from Checkpoint 0

Owner rulings, 2026-11-18. They override Part A where they differ.

- **B1, ℓI source (interim).** No training repo exists yet.
  - ℓI comes from a config hook `loader.log_i`. Its default is `illumination_edge/common.py` (`load_rgb`, `log_rgb`
    = log(max(I, 1))), imported, not copied. No black-level subtraction, no scaling.
  - §7 log-I parity runs against `illumination_edge/outputs/annotate/frozen_v2_pairs_fullres.csv` at **5e-6** (its
    values have 5 decimals). Re-run at 1e-6 once a real loader exists.
  - Prediction format: `<pred>/<stem>.npz`, key `log_R`, float32, shape (H, W, 3), natural log, full resolution,
    stored TIFF frame. `log_L` is an optional key; if present, the identity ℓL = ℓI − ℓR is checked.
  - No network exists, so Phase 3 runs the baselines only.
- **B2, material regions.** The region source is pluggable. None exists, so detector_split, D4, A1 and A2 report
  "n/a: no region source". Regions are never derived from open edge contours.
- **B3, splits.** The eval image list lives in `config.yaml`. Every annotated image is `split=eval`; the train list
  is empty. Each image has a `seen_in_training` field, default `unknown`.
- **C1, saturation.** Default `sat_dn = 65000` (a pixel is invalid if any channel ≥ 65000), as a config value, for
  parity with the detectors and the frozen patch medians. This replaces §2's "within 2% of saturation". Pixels in
  [64224, 65000) are counted per image and reported.
- **C2, near-black.** The annotator's near-black floor (max(64, 5 σ_read)) is not part of the validity mask; the
  noise guard covers it.
- **C3, patch shape.** Square patches everywhere, (2h+1)², with the annotated half-size h per end (replaces §2's
  "disc of radius r, default 4"). Discs stay available as a config option. The new tool keeps the auto-sizer.
- **C4, near vs distant.** Pair type decides: i, m, b and s are near; d is distant. The 80 px / 150 px limits are only
  tool warnings and a `length_out_of_range` flag. Pairs of 80–150 px are scored as near, flagged, and kept in the
  headline.
- **C5, migrated sets.** All four existing sets are migrated, each with its source file and SHA-256, all `split=eval`:
  dev v2, held-out, glossy dev, glossy confirm.
  - The headline aggregates dev v2 and held-out only.
  - Glossy dev and glossy confirm are reported as separate strata, never in the headline.
  - Deleted pairs go into the legacy block and are never scored.
- **C6, annotators.** `annotator` is a per-pair field and a column in every CSV export (migration table, per_pair.csv).
  - Migrated pairs: annotator = "Yueren" (all four sets).
  - Tool: a name-tag selector (known names plus "Add new…"); names in `metrics/annotations/annotators.json`, seeded
    with "Yueren", trimmed and de-duplicated case-insensitively; the current tag persists per browser and is stamped
    on every new pair; no tag selected means pair creation is blocked; editing keeps the original annotator and
    records `last_edited_by`; the pair list filters and counts by annotator.
  - A merge command unions per-image files by pair id and reports conflicts instead of overwriting.
  - Metrics stratify by annotator. Agreement analysis waits for double-annotated pairs.
- **C7, environment.** `metrics/` is its own uv project, using only packages the repo already uses.
- **C8, tool.** A new annotation tool in `metrics/annotate/` (not an extension of `annotate_pairs.py`, which is frozen
  under `frozen/SCORER_v4.json`). It keeps the old display transform (99.5th percentile → 1, gamma 2.2) and keys.
- **C9, keys.** The old keys stay (`u` = undo). The confidence toggle moves to another key, documented in the guide.

## Deltas from Checkpoint 1

Owner rulings, 2026-11-18. They override Part A and earlier deltas where they differ.

- **Schema (implementation deviations 1–4, approved).**
  1. Patch size per end: `A = {x, y, r}`, `B = {x, y, r}` (r = square half-size, or disc radius under
     `patch.shape: disc`) instead of one pair-level r.
  2. Per-image file fields: `set`, `seen_in_training` (unknown / yes / no), `width`, `height`, `image_sha256`,
     `s_seed_log`, `legacy` (source set / file / SHA-256, deleted v0 records).
  3. Optional pair fields: `last_edited_by`, `edited`, `history`, `legacy` (v0 record verbatim, v0_index, lit_end
     inference record, relabel).
  4. The optional float32 `log_L` prediction key is checked at 1e-5; the metrics' own ℓL (float64) at 1e-6.
- **Noise and texture are handled separately (replaces §2's noise guard; amends §4).**
  - Sensor noise: `low_snr` uses the SE of the patch median from the repo's sensor-noise model
    (`illumination_edge/common.py:estimate_noise_robust`, imported): low_snr if RMS over channels of
    sqrt(SE_A² + SE_B²) > τ/3. It still excludes pairs from the headline.
  - Texture: `se_tex` = the MAD-based SE of §2 (1.2533 × 1.4826 × MAD / sqrt(n)) on ℓI, combined over both patches
    the same way. It is a reported column and defines a textured / smooth stratum. It never excludes a pair.
  - For the R-flat types only (i, s, same-material d), two patches of one textured material can differ in median R
    under a perfect decomposition, so:
    - endpoint pass: E ≤ τ_eff, τ_eff = max(τ, 2 · se_tex);
    - profile pass: H_R ≤ max(τ, H_I) (R may wiggle as much as the input does, not more).
  - m pairs keep plain τ for both tests (L carries no texture).
  - The synthetic suite includes a textured material; ground-truth ℓR must pass under these tolerances.
  - Reports show per-type counts and pass rates under plain τ and the widened tolerances side by side.
- **Empty types.** A rate over zero pairs is reported as "n/a: 0 pairs", never 0.
- **lit_end.** Inferred per §3; it only feeds A1/A2 (n/a for now). The Phase 3 report lists the 13 corrected A/B
  swaps and the 2 marginal pairs.
- **Version control.** metrics/ (including annotations/) is committed on branch `metrics-v1`; runs/ and caches are
  ignored. No writes outside metrics/.

## Deltas from Checkpoint 2

Owner rulings, 2026-11-18. They override Part A and earlier deltas where they differ.

- **Synthetic shadow.** The synthetic penumbra mixes lit and shadow light in linear space (physical), not in log space.
- **D1, texture tolerances (R-flat kinds i, s, same-material d).**
  - k_tex = 2.5: endpoint pass E ≤ τ_eff = max(τ, 2.5 · se_tex).
  - Profile pass H_R ≤ max(τ, H_I), with σ_t = 3 px (instead of 1.5) for textured pairs (se_tex > 0.0333), per
    §2's "raise σ_t for textured materials".
  - Tested and not adopted (pre-declared rule): H_ref = max_t ‖p_I(t) − iso_I(t)‖ (per-channel isotonic fit). On the
    30-seed Monte Carlo it gave 57.5% ground-truth false fails on strong texture vs 16.7% for σ_t = 3 px (moderate:
    4.1% vs 1.5%). `texture.profile_ref: iso` remains as a config option only.
  - Textured pairs stay in the headline. Reports show the textured stratum separately, and the synthetic ground-truth
    false-fail rate (`metrics/reference/texture_mc_d1.json`) as the error floor.
- **D2.** §4's `structured_path` is renamed **`input_excursion`** (H_I > τ). It is a stratum only; those pairs stay in
  the headline. Reports show m error with and without input_excursion pairs, and a per-type cross-tabulation of
  input_excursion × textured × geom_cross on the headline sets.
- **D3.** geom_cross = the A–B segment passes a geom_v1_1 pixel with w < 0.5 (repo convention).
- **D4, distant pairs (corrects §5 and §8 wording).** Every same-material d pair (material and illumination labels
  both sure) is in Long-range, whatever its illumination label, stratified by it (§4 table). Only same-illumination,
  different-material d pairs are diagnostic (L drift). §5's exclusion "same-illumination d pairs" and §8's
  "distant same-illumination pairs: diagnostic only" mean these different-material pairs only.
- **D5.** P(ℓL) := P(ℓI) − P(ℓR) and p_L := p_I − p_R, so ΔℓL = ΔℓI − ΔℓR holds exactly.
- **D6.** se_tex is computed on ℓI only, never on the prediction.
- **D7.** Bin edges: reversed < 0 ≤ full leak < 0.2 ≤ partial ≤ 1 < overshoot. Textured stratum: se_tex > 0.0333,
  fixed across the τ sweep.
- **low_contrast (new).** For i and m pairs, low_contrast = ‖ΔℓI‖ ≤ τ, evaluated at each τ of the sweep. Such pairs
  pass under every decomposition and carry no split information: they are excluded from Split and counted. Not
  applied to s pairs.

## Deltas from Checkpoint 3

Owner rulings, 2026-11-18.

- **E1.** `thresholds.bin_eps` = 1e-4 (config): numerical slack at the 0 and 1 split-bin edges (reversed < −eps;
  partial ≤ 1 + eps). Float32 predictions otherwise flip α ≈ 0 into "reversed".
- **E2.** Dense diagnostics stay "n/a: map not available" where the default-arm maps were never exported (glossy_dev:
  D1–D3; glossy_confirm: D1). The exporter is not run for metrics.
- **E3.** D3 uses material-edge pixels farther than remove_px = 2 from illumination pixels and off the geom_v1_1
  w < 0.5 band.
- **τ sweep.** Reported on two populations: per τ (exclusions such as low_contrast and low_snr re-evaluated at each
  τ) and fixed (the pairs eligible at τ = 0.10, judged at each τ), which separates the threshold effect from the
  population change.
- **Perfect-decomposition floor.** Each Split number is shown with an approximate ground-truth Split floor (synthetic
  texture): the Monte Carlo false-fail rates in `metrics/reference/texture_mc_d1.json` (the option matching the
  config) weighted by the group's eligible i and m counts per texture stratum (smooth se_tex ≤ 0.0333; moderate;
  strong se_tex > 0.065, the midpoint of the Monte Carlo moderate and strong mean se_tex). Range: smooth i pairs
  at 0 to the not-textured-on-textured-material rate; m pairs have a 0 floor in the Monte Carlo.

## Deltas from Checkpoint 4

Owner rulings, 2026-11-18.

- **Tool input handling (trial bug).** The global hotkey handler ignores events from input, textarea, select and
  contenteditable elements and is off while any dialog is open; dialogs handle their own keys. Every text field
  (name tag, note, gap_ref, rejection reason) receives typed text. A new name tag is POSTed to annotators.json,
  appears in the selector and becomes the current tag; a failed save shows the server's error in the dialog. Every
  non-credential input carries `autocomplete="off"`, `data-1p-ignore`, `data-lpignore="true"`. The headless
  self-test types into every text field with real input events (CDP `Input.insertText` / `Input.dispatchKeyEvent`)
  and adds a name tag by typing before placing a pair under it.
- **Approved as leaned:** the keys (old keys kept; z also undoes; q confidence; t lit end; r random s; 1–3
  overlays); a click-click s pair is targeted, random s only from the seed button; the brighter end is pre-selected
  for lit_end; the auto-sizer uses the metrics validity mask; the schema extension (`deleted_pairs` tombstones,
  `s_seed_log` entry schema); no live geometric-crossing warning.
- **Sets and groups (config only; annotation files unchanged).** The former held-out 15 join the former dev v2 10 as
  the **dev** group (25 images), the headline. Each annotation file keeps its original set name (`dev_v2`,
  `heldout`) as provenance. A **confirmation** set of about 15 non-glossy val_set images (picked by the owner from
  25 seeded candidates, excluding the 46 annotated images and the glossy pool) is reported as its own block once
  annotated and becomes the held-out headline. prepare.py runs on the picked images only.
- **Blind relabel mode.** An existing pair's geometry is shown with its type, d labels and annotator hidden; the
  current annotator assigns the type and labels. Results go to a separate relabels file, never to the pair.
  Reports: per-type agreement and Cohen's κ for type and for the d material and illumination labels.
- **Per-image targets** (config, shown as progress in the pair list): i / m / b = the per-type means over the dev
  images; random s 12–15; targeted s as gaps are found; d 12–15, at least half same material across different
  illumination.
- **Annotation guide** (`docs/ANNOTATION_GUIDE.md`): written for a new annotator, numbered single-action steps, no
  spec notation, real labelled screenshots from the headless-Chrome harness, the ten required sections.

## Deltas from Checkpoint 5

Owner rulings, 2026-11-18. They override Part A and earlier deltas where they differ.

- **Trial data.** The owner's trial seed entries on acharya_mrudula_048 were removed (file restored from its pre-trial
  backup), so they cannot distort the random-s acceptance and rejection statistics. Annotation-file changes and code
  changes are always committed separately.
- **Pending seeds.** An image reopened with a pending random-s seed offers to resume or abandon it. Abandoning is
  always logged with a reason (schema: `reason` is required for `abandoned` as for `rejected`). No orphaned pending
  entries.
- **Draws** always exclude the FRESH_PROTOCOL span (positions 17–109 of the held-out draw sequence).
- **Blind relabel** of one's own pairs is opt-in and reported separately as test–retest.
- **Targets** per image are the dev medians: i 33, m 33, b 3 (random s and d 12–15; targeted s as gaps are found).
  Targets are guidance: a type can be marked **exhausted** on an image, with a note, and then counts as met.
- **The confirmation set is dropped**; the 100-image split replaces it.
- **Fresh-clone setup.** Machine-specific paths live in the untracked `metrics/config.local.yaml` (committed example:
  `config.local.example.yaml`); tracked files carry no machine paths. prepare.py checks each TIFF against its
  annotation file's `image_sha256` and refuses on a mismatch. `pyproject.toml` and `uv.lock` are committed (Python
  3.12). Tests that need local files skip with a clear message. `metrics/run.sh doctor.py` checks the environment,
  dataset, image hashes, cache and overlays. Overlay maps travel as an export / import archive with a hash manifest;
  without it overlays are n/a and targeted s pairs are disabled. Nothing built goes into git.
- **100-image split, train / val / test = 60 / 20 / 20.**
  - 54 new images drawn from val_set with seed 20261118 under the confirmation-candidate eligibility rules (excluding
    the 46 annotated images, the glossy pool and the FRESH_PROTOCOL span); no hand-picking.
  - All 100 assigned with a fixed seed, stratified by annotator group (A: the 46 migrated images; B: the 54 new) and
    glossy-pool membership (`metrics/splits/split100.json`).
  - Each image file has `split` ∈ {train, val, test}. It replaces the dev headline. The old set and group names stay as
    provenance and as strata.
  - Every run reports train and val as separate blocks plus the train − val gap per headline number. Test is reported
    only with `--test`. Splits are never pooled.
- **Pair field `used_in_loss` (bool) replaces the pair-level split (eval / train).** It is true when a training loss
  consumes the pair (e.g. the future pair-annotation loss). Such pairs are still scored, in a separate "supervised
  fit" block, never pooled with unused pairs. All current pairs are `used_in_loss = false`. This changes §3, §5, §8:
  - §3 "split ∈ {eval, train}" on every pair is replaced by `used_in_loss`; "Pairs that feed the planned lit-dark
    pair-annotation loss are split=train and are never scored. The metrics code refuses split=train" becomes: such
    pairs have `used_in_loss = true` and are scored only in the supervised-fit block. "Default: split by image" now
    means the per-image data split.
  - §5 exclusion "split=train" is replaced by the separate supervised-fit block.
  - §8 "Eval/train separation: split by image; pairs on train images may feed the pair-annotation loss and are never
    scored" becomes: split by image (train / val / test); `used_in_loss` pairs are scored separately as supervised fit.
- **Annotation scope.** Every image gets every pair type, with the same per-image targets on every split. Val and
  test images are annotated first, then train. The pair list panel shows each image's split and its progress.
- Schema 1.1 records these changes (per-image `split`, optional `exhausted`, pair `used_in_loss`).

## Deltas from Checkpoint 6

Owner rulings, 2026-11-18. They override Part A and earlier deltas where they differ.

- **Q8, Q9, Q10.** The inert placeholder for the git-ignored `log_chromaticity/material_edges.py` import stays, and
  any use of it raises an ImportError naming the missing file; doctor keeps reporting it. The four absolute
  `image_dir` paths in `migration_v0/manifest.json` stay as provenance; `migrate_v0.py verify` is maintainer-only. The
  "Yueren_test" tag stays removed.
- **Image identity is the pixels.** Every image file carries `image_pixel_sha256` = SHA-256 of
  `"{shape}|{dtype}|"` + the decoded array's bytes (the loader's `read` hook, uint16). `image_sha256` (file bytes)
  stays. prepare and doctor accept a file whose bytes differ but whose pixels match, with a warning, and refuse a
  pixel mismatch. Dataset images are never rated, tagged or edited in Explorer or Photos.
- **`reflective_advisory: true`** on draw54 ranks 1, 5, 10, 12, 16, 20, 28, 31, 33, 39, 45 (reflective on the
  contact sheet). It is reported as a stratum; the split is not redrawn.
- **Name tags have stable ids.** `annotations/annotators.json`: `{id, name, created, history[{from, to, at, by}]}`.
  Pairs, tombstones, seed-log entries, exhausted marks and relabels store the id (`^[a-z0-9][a-z0-9_]*$`). Every view
  and export resolves it to the current name; CSV exports carry `annotator` (current name) and `annotator_id`. A
  rename (name-tag menu: pair count, confirmation; refuses empty names and case-insensitive collisions) changes only
  `annotators.json`. merge.py unions tags by id and reports name conflicts (ours kept). Schema 1.2.
- **Assignments.** Every image file has `assigned_to` (annotator id): the 46 migrated images → `yueren`, draw54 → the
  teammate's tag, pre-created as **"Annotator B"** (id `annotator_b`; she renames it). The image list defaults to the
  current tag's images; opening another annotator's image warns and needs confirmation (blind relabel exempt).
  `docs/ASSIGNMENTS.md` (tracked, generated by `tools/assignments.py`) lists them in annotation order, with split and
  reflective flag, plus the draw54 contact sheet.
- **Overlays and targeted s are maintainer-only.** doctor reports READY without maps; without overlays the targeted s
  control is disabled with a one-line reason; annotator targets exclude targeted s. SETUP.md separates the annotator
  path (no archives, no maps) from a maintainer part (overlay export / import, map building, targeted s, check_pr).
- **No GPU for annotators.** The annotator workflow (uv sync, doctor, prepare, tool, merge, commits) imports no torch
  or DA3. `tools/build_maps.sh` (maintainer) fails early without a GPU or a needed environment. `uv.lock` resolves to
  wheels on macOS arm64 and Linux x86_64 (WSL2).
- **Teammate workflow.** Her own branch off `metric-pair-annotation` (the published branch, below); annotation-only commits; merge `metrics-v1` before each
  session; push at the end; one PR per finished batch; only her assigned images, val / test first.
- **Pre-merge check.** `tools/check_pr.py BRANCH` (read-only, against the merge base on `origin/metric-pair-annotation`): only the
  branch annotator's assigned annotation files, `annotators.json` and relabels may change; every changed file passes
  the schema with an unchanged image identity; new and edited pairs carry the branch annotator's id and authorship
  never changes; no tombstone is resurrected or removed; `annotators.json` only gains ids or records renames. It
  prints per-image progress (exhausted counts as met) and exits non-zero on any error.
- **Published branch (owner, after Checkpoint 6).** Local `metrics-v1` carries unpublished study history and is
  never pushed. GitHub gets `metric-pair-annotation`: origin/main plus one snapshot commit per publish containing only
  `metrics/` and `illumination_edge/common.py` (`tools/publish.py build`; trailer `Published-from: metrics-v1 <sha>`).
  Annotations merged on GitHub come back into metrics-v1 first (`publish.py pull --apply`, then an annotations
  commit); build refuses while GitHub has annotations metrics-v1 does not contain, or other changes under metrics/.
  The migrate-verify tests skip when the old pair files are absent (they are not published).
