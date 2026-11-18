# Metrics v1 — Design + Claude Code Prompt

Nov 17, 2026 · @Yueren Xing

## How to use this doc

Paste the whole doc into Claude Code as one prompt. Part A (§1–§8) is the normative metrics spec. Part B is the implementation task; it cites Part A by section and saves it as `metrics/DESIGN.md`.

v1 scope: sparse annotated-pair metrics are the headline; dense metrics are diagnostics.

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

## Part B · Claude Code prompt

Implement the evaluation metrics in Part A (§1–§8) for the Understanding Illumination project, plus a dedicated annotation tool and a user guide, in `metrics/` at the repository root. Work phase by phase. At every checkpoint, stop, report, and wait for approval. When a decision is borderline, state it with your lean instead of resolving it silently.

### Context

- Inputs are linear raw .tif images at 1023×768 or 768×1023. The network takes ℓI = log I and outputs ℓR = log R at full resolution; L = I/R.
- Existing annotations: i, m and b pairs on the 10 dev images, made with an existing HTML annotation tool. Pair ids are timestamps (`t=…`).
- Existing maps:
  - geom\_v1\_1 geometric edges
  - the material detector (log-chromaticity + ISD) and its material regions
  - the default illumination edge detector (step − material + H29 + H35-SB)
  - the unified smoothness weights
- A material-detector upgrade (Study 1) is in progress. Use the version the training config currently points to, and record which.

### Hard constraints

- Do not modify training code, losses, detectors or existing annotation files. Outside `metrics/`, everything is read-only unless I approve a change.
- Compute ℓI by importing the training data loader; never reimplement preprocessing.
- No off-the-shelf models. Use only dependencies the repo already has; the tool server uses the stdlib only.
- All thresholds and paths go in `metrics/config.yaml`, with defaults from Part A.
- Everything is deterministic (fixed seeds).
- Model predictions are never shown in annotation mode.

### Target layout (adapt names to repo conventions; flag deviations)

```
metrics/
  DESIGN.md               # Part A verbatim
  README.md               # running metrics, outputs, config
  config.yaml
  schema/pairs.schema.json
  core/                   # io, patches, profiles, pair_metrics, dense, aggregate, baselines
  run_metrics.py          # CLI
  migrate_v0.py           # existing i/m/b -> schema v1
  annotate/               # server.py, index.html, app.js, style.css, review.html
  annotations/            # one JSON per image, schema v1
  docs/ANNOTATION_GUIDE.md
  tests/
  runs/                   # outputs, gitignored
```

### Phase 0 — Recon (no code changes)

Find and report:

1. The existing annotation tool and its data format: fields, coordinate convention, patch radius, display transform, and where the JSON files live.
2. The training loader's ℓI computation: black level, scaling, ε, invalid masks.
3. The network's output format, and where predictions are saved or can be saved: shape, channels, dtype, log or linear.
4. Locations and formats of geom\_v1\_1, material edges, material\_regions, illumination edges and the smoothness weights. Confirm whether the weight is high away from edges.
5. The train/eval image lists.

Then propose the mapping from the existing annotation format to the §3 schema, and say whether you will extend the existing tool or build a new one, and why.

**Checkpoint 0.**

### Phase 1 — Schema, I/O, migration

1. Save Part A verbatim as `metrics/DESIGN.md`.
2. Write `pairs.schema.json` per §3, including schema\_version.
3. Write `migrate_v0.py`: a lossless migration of the existing pairs into `metrics/annotations/`, with original ids kept and §3 defaults applied, including the 070 reclassification. Print per-image, per-type counts before and after.
4. Write the I/O module: load an image and compute ℓI via the loader, load ℓR predictions, derive ℓL, build the validity mask.
5. Add the parity and identity tests from §7.

**Checkpoint 1:** the counts table and parity results.

### Phase 2 — Pair metrics and tests

1. Implement patches, profiles and pair metrics exactly per §2 and §4, including every per-pair flag.
2. Implement the three baselines from §7.
3. Implement the full synthetic suite from §7.

**Checkpoint 2:** test results, plus per-type pass rates and split-bin tables for the three baselines on the real annotations, each compared with the expected behaviour in §7. Explain any mismatch before going on.

### Phase 3 — Aggregation, diagnostics, CLI

1. Implement aggregation per §5: headline numbers, counted exclusions, τ sweep, cluster-bootstrap CIs and the stratified tables.
2. Implement D1–D4 and A1–A2 per §6, for q ∈ {0.90, 0.95}.
3. Build the CLI: `run_metrics.py --pred <dir> --images <list> --out runs/<name>`.
   - It writes per\_pair.csv, profiles.npz, per\_image.csv, summary.json and summary.md.
   - It always runs the three baselines alongside the model.
   - summary.json records the config, git commit, detector versions and annotation file hashes.
4. Run it on the current checkpoint if predictions exist; otherwise run the baselines only.

**Checkpoint 3:** summary.md.

### Phase 4 — Annotation tool (annotation mode)

- **Server.** `annotate/server.py` (stdlib) serves the tool, lists images, serves display images, and reads and writes `annotations/<image>.json`. Writes are schema-validated and atomic, with timestamped backups. Launch with one command that prints the URL.
- **Display.** Tone-mapped view of the linear raw: exposure slider, gamma 2.2 by default, a log-view toggle. Zoom and pan at full resolution. Patch discs are drawn at true radius.
- **Placing pairs.** Click A, then B. A hotkey chooses the type (i, m, b, s, d), and type-specific fields are prompted per §3. `u` toggles confidence. Notes, drag-to-edit, delete and undo (`z`).
- **Live warnings, never blocks:**
  - segment length outside the near/distant limits
  - a patch below 70% valid
  - low-SNR
  - a point out of bounds
- **Random s pool.** A seed button proposes uniform random valid points from a logged, seeded RNG. Accept, nudge ≤ 10 px, or reject with a reason; rejections are logged.
- **Overlays.** Toggles for illumination edges, material edges, geometric edges and material regions, all off by default. Any pair created while an overlay is visible gets overlay\_seen = true. Targeted s requires overlays on.
- **Pair list panel.** Filter by type, confidence and split; click to jump; show counts per type per image.
- **Defaults.** Keyboard-first, autosave on every change. The tool never loads predictions.

**Checkpoint 4:** I try the tool on one image.

### Phase 5 — Review mode and docs

1. Build `review.html` as a separate page from annotation mode.
   - It loads a run (per\_pair.csv and profiles.npz, served as JSON) and colours pairs pass/fail on the image.
   - It filters by type, bin and flag.
   - For a selected pair it shows the patches and the p\_I, p\_R and p\_L profile plots, with the endpoint segments and H marked.
2. Write `docs/ANNOTATION_GUIDE.md`, easy to follow for a new annotator. Use annotated screenshots if you can generate them, otherwise ASCII sketches. It must cover:
   - a 5-minute quick start: launch, open an image, place a first pair, confirm the save
   - each pair type in one paragraph, with one do and one don't, plus the §3 placement rules
   - the random vs targeted s workflow
   - d labelling, and when to mark a pair unsure
   - overlays, and why they are off by default
   - a keyboard shortcut table
   - where files are saved, backups, and migration of old files
   - troubleshooting
3. Write `README.md`: running metrics, reading the outputs, editing the config.

**Checkpoint 5:** final review.

### Checkpoint report format

Keep it short: what was built, test results, deviations from Part A with reasons, borderline decisions with your lean, and what comes next.
