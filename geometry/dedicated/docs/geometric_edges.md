# Geometric edges (geom_v1): the combined smoothness-loss weight

Status 2026-11-03: **final (TASK_GEOM_INTEGRATION G6)**. This is the consumer-facing description of `geom_v1`: what the
weight means, its frame and resolution, how to load it, the strengths, the alignment, the limitations and how to re-run.
Channel math: `docs/depth_channel.md`, `docs/fold_channel.md`, `docs/pipeline_tri_veto4.md` (the Laplacian detector);
shared architecture and the status of the integration items: `docs/geometry_channels.md` §6.1; the integration record
(design, evaluation, audit, G2–G5 results) is Appendix A below. **geom_v1 is the pipeline default** since 2026-11-03
(`configs/default.yaml` = `configs/candidate_geom_v1.yaml`; section 7).

## 1. What w is

`w` is a per-pixel weight for the smoothness term of a shading / illumination loss: **w = 1** where the surface is
smooth and the shading may be assumed continuous, **w → w_min = 0.05** at geometric edges, where the illumination can be
discontinuous (depth steps and silhouettes, limbs, folds between faces, creases). It is relaxation, not a label: the loss
should multiply its neighbour differences by the weight (section 2), never divide by it.

geom_v1 combines three channels by the **elementwise minimum** (a pixel is relaxed if any channel relaxes it):

| channel | fires on | weight per candidate | settings in geom_v1 |
|---|---|---|---|
| depth | occluding depth steps in DA3's depth (two-sided plane fits in disparity) | w_c = max(0.05, ΔZ₀/(ΔZ + ΔZ₀)), ΔZ₀ = 0.10 m; band max(2, ⌈σ_w⌉ + 1) px | additive null; `sky_band: depth_and_mask` (the sky-mask boundary ring within 4 px of an accepted sky edge) |
| fold | surface-normal discontinuities (two-sided plane normals, capped veto) | w_c from the shading jump J; band max(2, ⌈width/2⌉) px | fold v2 (multiplicative gate, capped veto); `crease_rescue: laplacian_corroborated` (8°, k_L 3) |
| laplacian | tri_veto4's detector: \|LoG(ρ_f)\| / ρ with the scale-ratio veto | w = max(w_min, ε/(q_pool + ε)) | `pool_radius: 2` (tri_veto4 pools with 3) |

```
Stage A cache: D (DA3 metric depth, canonical focal), sky_prob, meta          [no new DA3 pass; D never shifted]
   └─► GeometryContext (network res, upright, float64): S, Ω, ρ = 1/D, ρ_floor, ρ_f (trilateral), dense fit, n̂
        ├─► depth      → w_depth, r = ΔZ/Z_near (∞ at sky edges), ΔZ [m]
        ├─► fold       → w_fold, J
        └─► laplacian  → w_lap, q_pool/ε
   w_net = min(w_depth, w_fold, w_lap)                (network res, upright)
   w     = clip(bilinear upsample of w_net to the original size, half-pixel centres, [w_min, 1]), to the stored frame
```

**Frame and resolution.** Per image, `stage_b/geom_v1/<stem>.npz`:

| key | resolution | frame | dtype | meaning |
|---|---|---|---|---|
| `w` | original (the TIFF's size) | **stored** (as the TIFF is stored; the EXIF orientation is not applied) | float16 | **the deliverable** |
| `w_net` | network (504 × 378 or 378 × 504) | stored | float16 | combined weight before upsampling |
| `depth_w_net`, `fold_w_net`, `laplacian_w_net` | network | stored | float16 | per-channel weights |
| `depth_strength`, `fold_strength`, `laplacian_strength`, `depth_dZ` | network | stored | float16 / float32 | strengths (section 3) |
| `D`, `sky_mask`, `n_hat`, `normal_valid`, `sigma_theta`, `q`, `L_over_rho`, … | network | stored (n_hat components in the upright camera frame) | | diagnostics |

`<stem>.json` holds the per-image scalars (channel counts, null scales, frames: `geometry.orientation_applied`,
`upright_WxH`, `stored_WxH`); `<stem>_{depth,fold}_candidates.parquet` the candidate tables (upright network pixel
coordinates); `manifest.csv` one row per image. Every image of batch1 and batch2 has orientation 1 (stored = upright).

**Loading.**

```python
import numpy as np
from illum_baseline.outputs import pair_weights

w = np.load("baseline_output/stage_b/geom_v1/<stem>.npz")["w"].astype(np.float32)   # (H, W), original res, stored
w_x, w_y = pair_weights(w)                                                            # (H, W-1), (H-1, W)
# smoothness term, e.g.  sum(w_x * |dI_x|^p) + sum(w_y * |dI_y|^p)   on the image in the same (stored) frame
```

To get arrays in the upright frame use `illum_baseline.stage_b.load_upright(cfg, run_dir, stem)` (cfg from the run's
`config_resolved.yaml`). Do not differentiate or re-filter an upsampled weight map: upsample once (as saved) and take
neighbour minima.

## 2. `pair_weights`

`pair_weights(w) -> (w_x, w_y)` with `w_x[i, j] = min(w[i, j], w[i, j+1])` (shape H × (W−1)) and
`w_y[i, j] = min(w[i, j], w[i+1, j])` ((H−1) × W): the weight of a neighbour difference is the smaller of its two pixels,
so a difference that touches a relaxed pixel is relaxed. Compute it at load time on the full-resolution `w`, in the same
frame as the image the loss runs on. It is the only pairing implemented; it is symmetric and needs no orientation.

## 3. Strengths and units

The weights are what geom_v1 delivers; the strengths are saved for consumers that want the geometric magnitude (and for a
future fusion other than min, which is not implemented):

| array | definition | range, units |
|---|---|---|
| `fold_strength` | the fold's shading jump J (worst case over light directions) at rasterized pixels (max over overlapping bands), 0 elsewhere | J ∈ [0, 1], dimensionless |
| `depth_strength` | r = ΔZ / Z_near at rasterized pixels (max over bands), 0 elsewhere | r ∈ (0, ∞], dimensionless; **∞ at sky edges** and unbounded far sides: treat as saturated |
| `depth_dZ` | ΔZ at rasterized pixels (max over bands), NaN elsewhere | metres (canonical-focal DA3 depth scaled by f_n/300), ∞ at sky edges |
| `laplacian_strength` | q_pool / ε, so that w_lap = max(w_min, 1/(1 + strength)) where q_pool > 0 | dimensionless, ≥ 0 |

Under min with the Laplacian channel the depth channel sets w on ≤ 0.03 % of the pixels (section 5): its contribution
to the deliverable is the strengths r and ΔZ (which depth edge is an occlusion, and how large), not the weight.

## 4. Alignment

None is applied. DA3's edges sit a fraction of a pixel off the RGB edges, but the shift is **not stable** across images
(G2: pooled (−0.07, −0.47) px over 20 images, dy CI excludes 0, yet only 39 % of the images lie within 0.3 px of the
pooled shift; per-image dx from −0.54 to +0.54 px). No `deliverable_shift_px` option exists; `w` is in DA3's frame,
upsampled with half-pixel centres. The bands (≥ 2 px at network resolution, ≈ 4 px at original resolution for 1000-px
images) absorb sub-pixel offsets; 74 % of strong depth edges have their RGB edge within 2 px.

## 5. Limitations

Numbers from the labelled audit (`audit_split/`, 190 units, stratified pps estimator, 95 % stratified-bootstrap CIs;
`baseline_output/diag/geom/g6/limits.json`), the pinned batch1 ROIs and the G5 runs.

- **The depth channel rarely sets w.** Every depth w < 0.5 pixel lies inside the Laplacian's w < 0.5 except 2 (batch1)
  / 3 (batch2) px; depth is the binding minimum on 0.011 % / 0.029 % of the pixels (median lowering 0.03 / 0.02). Its
  deliverable role is the strengths (section 3). The sky ring (`sky_band: depth_and_mask`) likewise adds no area beyond
  the Laplacian's sky band.
- **The Laplacian's band excess and texture** (geom_v1, pooling radius 2), as shares of the tri-only area (665 801
  network px over 20 images, 17.5 % of all pixels) still relaxed per pixel: band excess (relaxation wider than the edge)
  **0.214 [0.159, 0.271]** (142 400 px; 0.86 of the band-excess area), tri_veto4 false positives on texture (tri_fp)
  **0.069 [0.038, 0.104]** (45 700 px), all units labelled texture **0.159 [0.114, 0.208]** (106 100 px). Texture ROIs
  (share w < 0.5): 070 floor 0.008, 079 grass 0.219, 079 sunlit grass 0.383, 083 bollard interior 0.002; texture class
  0.198 (tri_veto4 0.254, geom_v0 without the Laplacian 0.003). Geom-only false positives (DA3 bumps on foliage and
  surface marks): 0.052 [0, 0.104] of the geom-only area (39 123 px), band excess 0.033 [0.010, 0.064]. The price of
  keeping the Laplacian (G4 decision (c)): without it geom_v0 missed 0.584 [0.512, 0.656] of the tri-only area as true
  geometry; geom_v1 misses 0.007 [0, 0.021] per unit (0.079 per pixel).
- **Fold placement.** The fold channel sites a fold at the NMS ridge of the normal rotation rate. Signed offset of the
  RGB edge from it along the fit direction (15 audit units labelled geom_offset): vertical corners −2.02 px (toward the
  oblique face), nosings +2.59 px (up), contact folds +0.81 (mixed). At the intersection of the two fitted side planes
  the offsets are −1.05 / +1.21 / +0.32 px; all convex candidates −1.75 → −0.82 px toward the frontal face (the NMS peak
  sits toward the more camera-facing face; the plane site halves it). The plane site is unusable on narrow plinths: the
  planes cross within ±k₁ for only 34 % of the contact-target candidates. RGB caveat: the luminance maximum can be a
  shadow edge, and 12 % of the contact-fold samples sit at the ±8 px search limit. Not corrected in geom_v1 (geom_v2
  candidate).
- **Items 6 / 7 (fold channel), reopened, not fixed.** Folds on textured surfaces (audit units 97, 124, 125) and narrow
  faces narrower than the fold windows (15, 37) are missed by the fold channel (w < 0.5 on 0.00–0.32 of each unit); in
  geom_v1 the Laplacian covers them (0.70–0.89 of each unit, median w 0.17–0.28). Unit 92 (076 plinth contact fold)
  is covered by the crease rescue.
- **Thin-contour depth components.** Of the 45 step gaps the depth channel leaves open, 25 are dropped by the component
  minimum (contour pieces < 5 px; 82 % thin doublet fits, r above τ_high): fences and railings. The Laplacian covers
  them in geom_v1.
- **Alignment shift**: unstable across images, not applied (section 4).
- **Moderate creases**: the crease rescue relaxes the 088 rock ridge (coverage 0.76) but not the 076 façade ledge
  (blocked by the capped veto; documented gap, item 9).
- **Clutter**: fences, railings, balusters and foliage carry dense thin depth rings; the relaxation is kept there (the
  illumination is genuinely discontinuous); no audit unit was labelled artifact or clutter.
- **Runtime.** 7.00 / 7.05 s per image for Stage B on the CPU (median, batch1 / batch2; tri_veto4 0.55 s), 65 % of it
  the fold channel's dense fit and normals (4.5 s; breakdown in section 7); the GPU backend is a separate step.
- **DA3 dependence.** All geometry comes from one DA3 metric-depth pass at network resolution: edges below its resolution
  (gap_subscale: siding laps, ~1–2 cm trims; 0.034 of the tri-only area) are not resolved by the depth or fold channel.

## 6. How to re-run

From `geometry/dedicated/` (ROS / other Python paths off: `env -u PYTHONPATH`):

```bash
# Stage A (DA3, GPU) if the cache is missing; then Stage B geom_v1 (CPU), both batches
env -u PYTHONPATH uv run python -m illum_baseline stage-a --config configs/candidate_geom_v1.yaml
env -u PYTHONPATH uv run python -m illum_baseline stage-b --config configs/candidate_geom_v1.yaml
env -u PYTHONPATH uv run python -m illum_baseline stage-b --config configs/candidate_geom_v1.yaml -i data/batch2 -o baseline_output_batch2
# sheets only (viz, viz_depth, viz_fold, viz_geom)
env -u PYTHONPATH uv run python -m illum_baseline viz --config configs/candidate_geom_v1.yaml
# evaluation and audit reports (read RGB; diagnostic only)
env -u PYTHONPATH uv run python scripts/geom_g5.py          # G1 table, audit re-score, marginals, r sweep
env -u PYTHONPATH uv run python scripts/geom_g5_radius.py   # pooling radius report
env -u PYTHONPATH uv run python scripts/geom_g5_fold_site.py
env -u PYTHONPATH uv run python scripts/geom_g6_limits.py
env -u PYTHONPATH uv run python scripts/geom_g6_default.py  # default-proposal numbers (times Stage B)
# bit-for-bit check of every reference run (incl. geom_v1, geom_v1_r3)
env -u PYTHONPATH uv run pytest tests/test_baseline_repro.py
```

A run directory refuses a different detector / sky / preprocess / geometry section unless `--force`; any change of
settings needs a new `run.name`. Options are listed in section A.4.

## 7. The default (applied 2026-11-03)

geom_v1 is the pipeline default: the proposal commit (`configs/default.yaml` = `configs/candidate_geom_v1.yaml`, the two
tests that pin the default, and the docs lines that name it; branch `proposal/geom_v1_default`, `ff6f443`) was merged
into `yx_dev`. The previous default, tri_veto4, is `configs/candidate_tri_veto4.yaml`. What changed for a default run
(`python -m illum_baseline infer` / `stage-b` without `--config`), per batch
(`baseline_output/diag/geom/g6/default_proposal.json`, `timing.json`):

| | batch1 | batch2 |
|---|---|---|
| run directory | `stage_b/tri_veto4` → `stage_b/geom_v1` | same |
| w < 0.5, deliverable `w` (original res) | 0.2262 → 0.2156 (−0.0105) | 0.2741 → 0.2765 (+0.0025) |
| w < 0.5, `w_net` (network res) | 0.2274 → 0.2174 (−0.0100) | 0.2753 → 0.2788 (+0.0035) |
| marginal w < 0.5 area (px; share of geom_v1's w < 0.5) | laplacian 253 475 (61.2 %), fold 34 579 (8.3 %), crease rescue 7 222 (1.7 %), depth 2, sky ring 0 | laplacian 286 907 (54.0 %), fold 64 832 (12.2 %), crease rescue 13 894 (2.6 %), depth 3, sky ring 0 |
| runtime per image (Stage B, CPU, median, no sheets) | 0.54 s → 7.00 s | 0.55 s → 7.05 s |
| saved data per image | 1.6 → 6.2 MB | 1.5 → 8.9 MB |
| saved sheets per image | 1.0 → 3.6 MB | 1.1 → 4.6 MB |

New saved outputs per image: in `<stem>.npz` the keys `depth_w_net`, `depth_strength`, `depth_dZ`, `depth_q_over_s`,
`depth_edge_mask`, `fold_w_net`, `fold_strength`, `fold_r_n`, `laplacian_w_net`, `laplacian_strength`, `laplacian_q`,
`laplacian_L_over_rho`, `n_hat`, `normal_valid`, `sigma_theta` (kept: `w`, `w_net`, `q`, `L_over_rho`, `D`,
`sky_mask`); the tables `<stem>_depth_candidates.parquet`, `<stem>_depth_null.parquet`, `<stem>_fold_candidates.parquet`,
`<stem>_fold_null.parquet`; the sheets `viz_depth/`, `viz_fold/`, `viz_geom/` (besides `viz/`); the JSON gains
`channels`, `combined`, `normals_computed`; the manifest gains `depth_*` columns.

Config keys that change (effective values; `configs/default.yaml` relies on the built-in geometry defaults today):
`geometry.enabled` false → true; `geometry.channels` [laplacian] → [depth, fold, laplacian]; `geometry.fold.veto_mode`
fixed → capped; `geometry.fold.crease_rescue` off → laplacian_corroborated; `geometry.depth.sky_band` depth →
depth_and_mask; `geometry.laplacian.pool_radius` null → 2; `run.name` tri_veto4 → geom_v1. The YAML additionally spells
out the `geometry.dense_fit`, `geometry.fold`, `geometry.depth` and `geometry.laplacian` sections (at their built-in
values otherwise). Detector, sky, preprocessing, DA3 and paths are unchanged, so the Stage A cache is reused.

**Runtime breakdown** (Stage B `process_one`, median s per image, CPU, no sheets; `scripts/geom_g6_timing.py`; IO timed
inside the same call, the parts in a second pass in Stage B's order; tri_veto4 from `scripts/geom_g6_default.py`):

| part | batch1 | batch2 |
|---|---|---|
| total | 7.00 | 7.05 |
| IO (Stage A cache in; npz, parquet, JSON out) | 0.28 | 0.28 |
| shared: disparity, sky mask | 0.004 | 0.004 |
| shared: ρ_f trilateral (depth and fold) | 0.31 | 0.32 |
| depth channel (incl. rasterization 0.005, sky ring ≈ 0) | 0.57 | 0.68 |
| fold context: dense fit + normals | 4.53 | 4.57 |
| fold channel (incl. crease rescue 0.11 / 0.12, rasterization 0.03, L/ρ_f record 0.01) | 0.82 | 0.81 |
| laplacian channel (tri_veto4's detector, its own trilateral) | 0.37 | 0.38 |
| combine (min) + deliverable upsample | 0.002 | 0.002 |
| other (array conversion, records) | 0.11 | 0.01 |

**Why the depth channel stays.** The report-only variant `geom_v1_fold_lap` (`configs/candidate_geom_v1_fold_lap.yaml`,
[fold, laplacian], geom_v1's other options; `scripts/geom_g6_variant.py`, `g6/variant.json`) has the same G1 ROI table
to 4 decimals and bit-identical fold and Laplacian weights; dropping depth raises w on 0.011 % / 0.029 % of the network
pixels (batch1 / batch2; > 0.1 on 0.0008 % / 0.0035 %, max 0.34 / 0.54; the w < 0.5 status of 2 / 3 network px), and
saves 0.62 / 0.69 s per image (6.38 / 6.36 s). Depth stays in the default because its accepted edges and tables (r, ΔZ,
the candidate table) are the input of the geom_v2 candidates (`docs/geometry_channels.md` §7.1: thin-contour depth
components, the depth strengths) and the consumers' occlusion information, at ≤ 0.03 % of the weights and 0.65 s.

**Sheets.** A default run renders the diagnostic sheets: `infer` and `stage-b` write `viz/`, `viz_depth/`, `viz_fold/`
and `viz_geom/` per image (3.6 / 4.6 MB per image). Rendering them takes about 24 s per image on top of the Stage B
times above (median over 4 images: `viz` 4.0 s, `viz_depth` 7.6 s, `viz_fold` 6.8 s, `viz_geom` 5.5 s), i.e. a default
run with sheets is about 4× slower than without. For dataset-scale runs pass **`--no-viz`** (a CLI flag of `infer` and `stage-b`; there is no config key); sheets can be
rendered later for chosen images with `python -m illum_baseline viz [stems ...]`.

## 8. geom_v2 design (TASK_GEOM_V2 V0; nothing built yet)

**Goal.** geom_v2 = geom_v1 with the Laplacian channel demoted to a **fallback**: it relaxes only away from accepted
geometric edges, so near them the depth and fold channels alone set the band. The fusion stays the elementwise min; the
fallback is a mask on the Laplacian's w before the min. Retirement is not the goal: gap_unresolved alone is 0.134
[0.089, 0.186] of the tri-only area (DA3 shows that geometry only as a smooth gradient), above G4's 10 % threshold.

**Definitions** (all milestones). Combined w = the run's deliverable weight. Core w = min(w_depth, w_fold) from the saved
per-channel arrays (geometry without the Laplacian). Per-unit closure = w < 0.5 on ≥ 50 % of the unit; per-pixel share =
the unit's fraction of pixels with w < 0.5. Audit estimates: the stratified pps estimator with stratified-bootstrap CIs,
scaled to the G1 totals (`scripts/geom_g4.py`). Labels: `audit_split/audit_labels.csv`, final.

### 8.1 The four parts and how each is scored

| part | milestone | mechanism | scored by | proposal rule |
|---|---|---|---|---|
| scorer + fold-gap map | V1 | `scripts/geom_score.py`; stage of every open fold / crease gap unit | reproduces the G5 / G6 numbers on geom_v1 (test) | V4 design: one option at the dominant addressable stage |
| fold re-siting | V2 | band centreline at the side planes' intersection | signed RGB-edge offsets (NMS vs re-sited) per sub-type, validity fraction, fold-target RGB coverage, geom_gain / geom_offset closure, texture | convex median \|offset\| −0.5 px or more; fold-target coverage −0.01 at most; no geom_gain unit loses closure |
| short thin depth components | V3 | accept short thin pieces now dropped by the component minimum | under core w: the 25 component-stage step units, all step gaps (per unit and per pixel), relaxed shares of tri_fp / geom_fp / texture units, texture ROIs (core and combined), overlays | most step-gap px closed, with texture ROIs (combined) ≤ geom_v1 + 0.01 and texture-unit relaxed share (core) + 0.02 at most |
| fold recall | V4 | the option approved at V1 | its V1 targets; texture ROIs ≤ geom_v1 + 0.01; 083 interior unchanged | as approved |
| fallback Laplacian | V5 | w_lap′ = 1 within d_excl of an accepted edge | every metric of the scorer, per setting; held-out batch1 → batch2 check; retirement metric (core, gap_unresolved apart) | lowest band excess among settings with missed geometry (per pixel) ≤ geom_v1 + 0.02, step / limb / fold ROI coverage each ≥ geom_v1 − 0.02, RGB-edge coverage of depth edges and fold targets each ≥ geom_v1 − 0.02, and (V2b review) RGB-edge coverage of all accepted fold candidates (reliable edges) ≥ geom_v1 − 0.02; ties by texture |

**V1 scorer** (`scripts/geom_score.py`, any list of runs; combined w and core w): G1 ROI table (classes, coverage), audit
per bucket per unit and per pixel, RGB-edge coverage of strong depth edges and fold targets (the G2(c) measure), texture
ROIs, w < 0.5 per batch, Stage B runtime per image (`scripts/geom_g6_timing.py`). Baselines tri_veto4, geom_v1_r3,
geom_v1. It **imports** the existing functions (`geom_eval.roi_list / roi_metrics`, `geom_g4.Estimator`,
`geom_g2_coverage.raster / covered / to_full / poly_normals`, `geom_g2_align.strong`, `rgb_edges`) and moves the
coverage loop of `geom_g2_coverage.main` into a function; the old scripts keep their outputs.

**V1 fold-gap stages** (units labelled fold or crease, bucket gap_fixable or gap_unresolved, open under core w; fold
candidates of geom_v1 within 2 px; furthest stage reached, in this order):

| stage | from the saved run |
|---|---|
| no candidate (below the NMS floor) | no row; max `fold_r_n` in the unit / the run's `rate_threshold` reported |
| window validity / contamination | reason `invalid_window` |
| planarity | valid, a side on the dense-normal fallback (`fallback_p/m`), gate failed |
| χ² gate | both sides planar, reason `gate` |
| capped concentration veto | reasons `veto`, `veto_undecidable` |
| component minimum | reason `component` |
| J_min | accepted, but w_c ≥ 0.5 (J < J_min + J0, Δθ ≲ 20°): J_min enters only the weight, not acceptance |
| rescue ineligible | gate failures not rescued, split by the rescue condition that fails (Δθ < 8°, J < J_min, veto, component, support) |

(The listed order puts J_min before the component minimum; in the channel the component minimum acts on acceptance and
J_min only on the weight, so J_min is the later stage. *Flag; lean:* the order above.)

### 8.2 New options (each defaults to the old behaviour; recorded in the JSON and the candidate tables only when on)

| option | values (default first) | where | records when on |
|---|---|---|---|
| `geometry.fold.site` | `nms` \| `plane_intersection` | fold channel, before rasterization | columns `site_t`, `site_valid`, `site_t_smooth`; scalar `n_site_valid` |
| `geometry.fold.site_max_t` | 3.0 px | | |
| `geometry.fold.site_smooth_n` | 5 | | |
| `geometry.depth.short_thin` | `off` \| `strong` \| `linked` | depth channel, D-5 component step | column `short_thin` (accepted by the waiver / the link); `link_id`; scalars |
| `geometry.depth.short_thin_min_px` | 2 | `strong` | |
| `geometry.depth.short_thin_gap_px`, `..._angle_deg` | see flag 2; 20° | `linked` | |
| V4 option | approved at V1 | fold channel | |
| `geometry.laplacian.role` | `channel` \| `fallback` | Stage B `run_geometry`, after the channels, before the min | arrays `laplacian_w_fallback`, `laplacian_exclusion`; scalar `exclusion_share` |
| `geometry.laplacian.d_excl` | 3.0 px | `fallback` | |

- **Site.** The k₁ two-sided fit already gives the root of ρ̂₊ − ρ̂₋ along e1 (`s_gap` when `planes_cross`, both sides
  planar), so t needs no refit: valid if planes cross with |t| ≤ 3 px, else t = 0 (x₀ kept). Smoothing: running median
  of t over 5 candidates along `geom.depth.chains` (mutual-nearest ordering along the fold line) within a component;
  candidates in chains shorter than 5 keep their own t. Δθ, J, the gate and every acceptance decision are unchanged;
  only the rasterized centreline moves to round(x₀ + t e1).
- **Fallback.** The exclusion zone is a Euclidean distance transform (network resolution) from the rasterized
  centrelines of the accepted fold candidates (rescued included; at the V2b site) and of the accepted depth candidates
  **except thin doublets**, sky edges kept although they carry the thin flag (settled at the V3 review: excluding around
  thin fragments would break the Laplacian's continuous relaxation on fences and railings; the all-edges variant is
  reported once at the chosen setting as a check). `role: fallback` needs depth or fold in `channels` (validated). The saved `laplacian_w_net`
  keeps its meaning (the channel's own output); the masked weight that enters the min is saved separately.

### 8.3 Runs and configs

| milestone | config / run | role |
|---|---|---|
| V1 | none (scorer on tri_veto4, geom_v1_r3, geom_v1); outputs `baseline_output/diag/geom/v2/` | report |
| V2 | `configs/candidate_geom_v1_site.yaml` → `geom_v1_site` (both batches) | diagnostic, not a reference |
| V3 | depth channel rerun in-script from the Stage A cache for the variants (0.6 s per image), the rest re-rasterized from the geom_v1 tables; `geom_v1_thin_<variant>` run for the proposed variant only | diagnostic |
| V4 | `configs/candidate_geom_v1_<V4 option>.yaml` → `geom_v1_<V4 option>` | diagnostic |
| V5 | in-script sweep: re-rasterization (band_min, exclusion) and Laplacian reruns (pool_radius 2 / 3) on the V2–V4 runs' tables | report |
| V6 | `configs/candidate_geom_v2.yaml` → `geom_v2` (both batches); branch `proposal/geom_v2_default` | **new bit-for-bit reference**; default proposal (not applied) |

### 8.4 What stays bit-for-bit

Every existing reference, with every new option at its default: `baseline_v0`, `tri_veto4` (batch1, batch2), the
`[laplacian]` ≡ tri_veto4 test, `fold_only`, `tri_veto4_fold`, `fold_v2`, `tri_veto4_fold_v2`, `geom_v1` and
`geom_v1_r3` (batch1, batch2): arrays, JSON blocks, candidate / null tables (new columns only when an option is on) and
channel scalars. Plus option-off tests against geom_v0 / geom_v1 outputs, as in G2 / G3. `configs/default.yaml` (geom_v1)
is not changed.

### 8.5 Missing tools and annotations

- **Tools to build:** the scorer and its geom_v1 reproduction test (V1); the fold-gap stage diagnostic (V1); an overlay
  script for newly accepted depth pieces (V3); the exclusion-zone rasterizer, shared by Stage B and the V5 sweep (V5);
  a sampler that draws audit units from an arbitrary mask (geom_audit.py takes only the tri-only / geom-only maps; V6's
  supplementary audit); viz_geom panels 7 / 8 for the fallback and for geom_v2 vs geom_v1 (V6).
- **Not available, so proxies are used:** fold-site ground truth (the RGB luminance edge stands in, with its
  shadow caveat and the ±8 px search limit); labels for the newly accepted V3 pieces and for any area V2–V5 relax outside
  the labelled units (overlays, then V6's conditional `audit_v2_labels.csv`); an independent test set (the audit is
  also the selection set; V5's batch1 → batch2 held-out check is the only guard).

### 8.6 Flags (lean in each)

1. **Stage order J_min vs component minimum** (8.1): lean, J_min after the component minimum, as the channel applies them.
2. **`linked` gap.** Depth candidates within `link_radius` = 3 px already form one component (D-5), so separate pieces
   are > 3 px apart and a gap ≤ 2 or ≤ 3 px links nothing. Lean: endpoint gap ∈ {4, 6} px (the sweep), keeping the 20°
   tangent and same-polarity conditions; endpoints and tangents from `chains`.
3. **Site smoothing** along `chains` within a component; chains shorter than 5 keep their own t. Lean: as stated.
4. **Exclusion edges:** sky edges included (lean, as you leaned); the sky ring (`depth_and_mask`) is a painted weight,
   not an edge centreline: lean, not included.
5. **Saved arrays under the fallback:** `laplacian_w_net` stays the channel's own output, the masked weight is
   `laplacian_w_fallback`; so combined w = min(depth, fold, laplacian_w_fallback), and the scorer's core w is unaffected.
6. **Scorer refactor:** import from the existing scripts (and move the G2(c) loop into a function) rather than rewrite
   them, so their committed outputs stay reproducible.
7. **V3 / V5 by re-rasterization:** exact for band_min and the exclusion (the G2 / G5 re-rasterization reproduces the
   saved w bit for bit); the depth variants need the depth channel rerun (cheap), not the fold channel.

### 8.7 V1 results: scorer and fold-gap map

**Scorer** (`scripts/geom_score.py`; outputs `baseline_output/diag/geom/v2/score.json`, `score_summary.csv`; the fixed
RGB-edge set cached in `rgb_edge_set.npz`; test `tests/test_geom_score.py` reproduces the task-file geom_v1 numbers
within 0.0008):

| metric | tri_veto4 | geom_v1_r3 | geom_v1 | geom_v1 core |
|---|---|---|---|---|
| missed true geometry per pixel / per unit | 0 / 0 | 0 / 0 | 0.079 / 0.007 | 0.568 / 0.584 |
| gap_unresolved missed per pixel | 0 | 0 | 0.017 | 0.128 |
| band excess still relaxed (per pixel) | 0.256 | 0.256 | 0.214 | 0.003 |
| tri_fp still relaxed | 0.083 | 0.083 | 0.069 | 0 |
| texture-labelled units relaxed | 0.193 | 0.193 | 0.159 | 0 |
| ROI coverage step / limb / fold / crease | 0.909 / 1 / 0.962 / 0.495 | 0.909 / 1 / 0.999 / 0.585 | 0.900 / 1 / 0.982 / 0.574 | 0.850 / 1 / 0.817 / 0.399 |
| texture: class / 079 sunlit / 079 grass | 0.254 / 0.504 / 0.267 | 0.254 / 0.504 / 0.267 | 0.197 / 0.383 / 0.218 | 0.003 / 0.006 / 0.003 |
| RGB-edge coverage: depth edges / fold targets | 0.990 / 0.847 | 0.991 / 0.908 | 0.977 / 0.869 | 0.826 / 0.698 |
| w < 0.5 deliverable, batch1 / batch2 | 0.226 / 0.274 | 0.240 / 0.304 | 0.216 / 0.277 | 0.081 / 0.124 |
| Stage B s per image, batch1 / batch2 | 0.56 / 0.56 | 7.96 / 9.25 | 8.78 / 9.10 | – |

(Runtimes from the scorer's own pass; this machine varied by about 25 % against G6's clean 7.00 / 7.05 s; compare
within a pass only.)

**Fold-gap map** (`scripts/geom_fold_gaps.py`; `fold_gaps.json`, `fold_gaps_units.csv`, `fold_gaps_candidates.csv`).
33 fold / crease units in gap_fixable or gap_unresolved; 4 closed and **29 open** under geom_v1's core w. Furthest stage
reached per unit (fold candidates of geom_v1 within 2 px):

| stage | units | px | share of px |
|---|---|---|---|
| no candidate (below the NMS floor) | 2 | 1 127 | 0.09 |
| window validity / contamination | 2 | 1 000 | 0.08 |
| planarity | 0 | 0 | 0 |
| χ² gate | 7 | 3 352 | 0.26 |
| capped concentration veto | 5 | 2 304 | 0.18 |
| component minimum | 6 | 2 188 | 0.17 |
| accepted, w ≥ 0.5 (J ≤ J_min + J0) | 1 | 944 | 0.07 |
| accepted, w < 0.5 (band misses the unit) | 6 | 2 064 | 0.16 |

The furthest stage hides where most candidates stop: the units past the gate reach it with 1–7 candidates in components
of 1–3 px (fold minimum 5) or 1–7 px (rescue minimum 10). Candidate level (932 candidates of the open units): χ² gate
455, planarity 197, window 108, band misses 97, veto 42, component 25, accepted w ≥ 0.5 8; modal stage per unit: χ² gate
12 units (5 273 px), planarity 9 (4 418 px), so the **gate** stops most candidates in 21 of the 29 units (75 % of the
px). The χ²-gate failures are not weak folds (Δθ P10 / P50 / P90 4.1 / 24.8 / 71.1°; T/λ̂ P50 0.68 against 13.8). What
keeps those gate failures out of the crease rescue (candidate level): **rescue veto 428 candidates in 21 units, every
one undecidable** (none decided as fail; Δθ P50 43°), Δθ < 8° 122 (7 units), rescue component < 10 px 71 (14 units),
support < 0.7 31 (2 units).

- **"accepted, w ≥ 0.5" does not dominate** (1 unit, 113, the 083 bollard collar crease; 4 candidates, Δθ 13.4–14.4°),
  so V4 does not target the J0 knee.
- **Textured-surface units:** 97 (079 fold) χ² gate, rescue blocked by the veto; 124 (088 crease) χ² gate, rescue
  component < 10 px; 125 (088 crease) accepted, band misses the unit (core 0.14).
- **Narrow-face units:** 15 (067 pickets) capped concentration veto; 37 is labelled step (not in this map; G4: no
  depth candidate above the gradient floor).

**V4 design (proposed, not coded).** One option, aimed at the dominant addressable stage (gate failures kept out of
the rescue by an undecidable veto):

- `geometry.fold.rescue_undecidable: reject | corroborate`, default `reject` (today's behaviour).
- `corroborate`: in the crease rescue, an eligible gate failure (Δθ ≥ 8°, J ≥ J_min) whose second-pass capped veto is
  undecidable counts as "decided as pass" when its rescued component meets the existing size (≥ 10 px) and support
  (≥ 70 %) conditions **on the scale-ratio-vetoed Laplacian** (q′ = min(|L|, g |L_σ − L_4σ|) / ρ, the tri_veto4 veto,
  computed on ρ_f as a new context record field) instead of the un-vetoed L/ρ_f. The concentration veto exists to
  reject smooth curvature; the scale-ratio veto removes exactly that from the corroborating response, so it stands in
  where the concentration ratio cannot be decided. Decided-fail candidates stay rejected; nothing else changes.
- **Targets:** ≥ 7 of the 21 veto-blocked open units closed under core w; texture ROIs (combined) ≤ geom_v1 + 0.01;
  the 083 bollard interior unchanged; texture-labelled units relaxed (core) + 0.02 at most; geom_fp units relaxed
  (core) not increased. Reported: the 076 façade ledge and 088 ridge coverage (the ledge was blocked by the same veto
  in G3), units 15, 97, 124.
- It reopens item 7 (the undecidable capped veto is the narrow-face mechanism): the outcome goes to fold_channel.md §7.
- Not proposed: the χ² gate itself (the failures come from noisy normals, T/λ̂ P50 0.68 at Δθ P50 25°; a local σ_geo,
  item 6, which is open (reopened in G4, not fixed), would make the gate stricter on noisy surfaces) and the rescue's
  10-px minimum (second, 14 units; an existing parameter).
- **Approved at the V1 review** with these targets, plus: rescued candidates split by whether another accepted edge lies
  within 10 px (narrow-face cases); overlays on the 075 kiosk cylinder and the 083 bollard (smooth-curvature controls);
  a report-only rescue component minimum of 6 px. Order V2 → V3 → V4 → V5.

### 8.8 V2 results: fold re-siting

Option `geometry.fold.site: nms | plane_intersection` (default nms; `site_max_t` 3 px, `site_smooth_n` 5; recorded only
when on). The site is the k₁ fit's plane-gap root (`s_gap` when `planes_cross`), valid if |t| ≤ 3 px; t is smoothed by a
running median over 5 candidates in chain order (`geom.depth.chains`; chains shorter than 5 keep their own t); a
candidate without a valid t keeps its NMS pixel. Acceptance, Δθ, J and w_c are unchanged (test); only the rasterized
centreline moves. Columns `site_t`, `site_valid`, `site_t_applied`, `u_site`, `v_site`; scalars `n_site_valid`,
`site_valid_share`. Run `geom_v1_site` (`configs/candidate_geom_v1_site.yaml`, both batches, diagnostic); report
`scripts/geom_v2_site.py` → `baseline_output/diag/geom/v2/site.json`, `site_samples.csv`.

**Validity:** 0.50 (batch1) / 0.45 (batch2) of the accepted fold candidates have a valid site (convex 0.51 / 0.44); geom_offset
units 0.80–0.99 per sub-type; fold targets: vertical corners 0.69, contact folds 0.13, nosings 0.75.

**Signed RGB-edge offset along e1** (median, px; NMS → re-sited; G5 conventions):

| set | n | NMS (median \|·\|) | re-sited (median \|·\|) |
|---|---|---|---|
| geom_offset vertical corners (5 units) | 585 | −2.00 (2.00) | −1.05 (1.05) |
| geom_offset nosings (5) | 213 | +2.90 (2.99) | +1.50 (1.61) |
| geom_offset contact (5) | 281 | +0.81 (3.48) | +0.28 (3.36) |
| fold targets, vertical corners | 459 | −0.94 (1.31) | −0.07 (1.18) |
| fold targets, contact | 364 | +2.55 (2.75) | +2.56 (2.78) (valid 0.13) |
| fold targets, nosings | 20 | +8.00 (65 % at the search limit) | +6.17 |
| all accepted convex candidates, 20 images (frontal convention) | 11 515 | −1.69 (**2.84**) | −1.19 (**2.63**) |
| the same, valid sites only | 5 437 | −1.45 (1.93) | −0.80 (1.47) |

Per unit: 143 −3.22 → −0.83, 153 +1.73 → −0.11, 154–156 +2.3–2.9 → +0.8–1.6, 183 / 189 halved; 178 / 179 slightly
worse (−1.2 → −1.4, −2.2 → −2.5), 182 / 187 worse (+1.9 → +3.0, −2.4 → −3.0), 180 no valid site.

**Coverage and closure:** fold-target RGB-edge coverage combined 0.869 → 0.870, core 0.698 → 0.701; split: targets the
fold channel detects (8) combined 0.899 → 0.899, core 0.744 → 0.748; not detected (5) combined 0.828 → 0.828, core
0.634 → 0.635. ROI coverage combined step / limb / fold unchanged, crease 0.574 → 0.553 (core 0.399 → 0.367). geom_gain:
31 / 31 units still closed (combined and core), but the per-pixel relaxed share falls 0.664 → 0.573; geom_offset units
15 → 13 closed (143, 182: the band leaves the displaced area the unit was drawn on), relaxed 0.250 → 0.190. Texture ROIs
unchanged (combined and core).

**Rule: not met.** The convex-candidate median |offset| drops by **0.21 px** (0.46 px on the valid sites alone),
against ≥ 0.5; fold-target coverage +0.0004 (combined) / +0.003 (core) and no geom_gain unit losing closure pass. So
`site` stays `nms` (not proposed). Residual offset with the option on: convex −1.19 px median (frontal convention),
vertical corners −1.05, nosings +1.50; more than half of the convex candidates have no valid site (planes not crossing
within ±3 px), and those keep the full NMS offset. No further correction in this task.

### 8.9 V2b results: Hessian-peak siting (proposed)

Third mode `geometry.fold.site: hessian_peak` (with `site_window_px` 4; recorded only when on). For each accepted fold
candidate: y(s) = −convexity · q(x₀ + s e1), s ∈ [−4, 4] px (step 0.25), q = e1ᵀ H e1 / max(ρ_σ, ρ_floor) from the depth
channel's kernels on ρ_f (built-in DepthParams: σ 1.25, truncate 4); the site is the parabola-refined maximum. Invalid
(x₀ kept) if the convexity is 0, the peak is at a window end, the peak fails the depth channel's height test
(y > k_h ŝ_q, k_h 3) or the opposite-sign extremum exceeds half the peak (a doublet: a step component). Same 5-candidate
chain-order median as plane_intersection. Sign: the recorded convexity is −sign(L/ρ_f) (+1 = disparity ridge, where
nᵀHn < 0), so the fold's own bump is −convexity · q > 0. Synthetic check (test): on an asymmetric kink (face slopes
+0.002 / −0.0005 per px) the peak lies within 0.3 px of the kink from x₀ offsets of +1.5, −2 and 0 px; a step is
rejected as a doublet. Run `geom_v1_hessian` (`configs/candidate_geom_v1_hessian.yaml`, both batches, diagnostic).

**Signed RGB-edge offset** (median, in brackets the median |·|; px) and validity, nms / plane_intersection / hessian_peak:

| set | n | nms | plane_intersection (valid) | hessian_peak (valid) |
|---|---|---|---|---|
| geom_offset vertical corners | 585 | −2.00 (2.00) | −1.05 (1.05) (0.99) | **−0.27 (0.66)** (1.00) |
| geom_offset nosings | 213 | +2.90 (2.99) | +1.50 (1.61) (0.80) | +2.71 (2.81) (0.65) |
| geom_offset contact | 281 | +0.81 (3.48) | +0.28 (3.36) (0.98) | +0.30 (2.38) (0.92) |
| fold targets vertical corners | 459 | −0.94 (1.31) | −0.07 (1.18) (0.69) | −0.07 (1.02) (0.98) |
| fold targets contact | 364 | +2.55 (2.75) | +2.56 (2.78) (0.13) | +1.96 (2.22) (1.00) |
| fold targets nosings | 20 | +8.00 (65 % at the limit) | +6.17 (0.75) | +8.01 (0.95) |
| **all accepted convex candidates** (frontal) | 11 515 | −1.69 (**2.84**) | −1.19 (2.63) (0.47) | −0.77 (**1.70**) (0.74) |
| the same, valid sites only | | −1.45 (1.93) | −0.80 (1.47) (n 5 437) | −0.72 (1.22) (n 8 540) |

Validity of all accepted candidates: plane_intersection 0.50 / 0.45 (batch1 / batch2), hessian_peak 0.76 / 0.79 (convex
0.72 / 0.76). Per unit (hessian_peak): vertical corners 143 −3.22 → −0.29, 178 −1.23 → −0.17, 179 −2.20 → −0.97, 183
−1.46 → −0.02, 189 −0.76 → +0.16 (every one within 1 px); nosings 153 +1.73 → −0.05, 155 / 156 halved, 154 no valid
site, 180 +6.91 → +8.18 (worse; RGB edge at the limit); contact 165 +5.65 → +2.93, 161 / 173 / 182 / 187 about
unchanged.

**Coverage, closure, texture** (geom_v1 → hessian_peak): fold-target RGB-edge coverage combined 0.869 → 0.869, core
0.698 → 0.698 (detected targets 0.899 → 0.899 / core 0.744 → 0.744; not detected 0.828 → 0.828 / 0.634 → 0.634; one
sample of 2 599 moves each way); RGB coverage of depth edges 0.977 → 0.976 (core 0.826 → 0.824); ROI coverage combined
fold 0.982 → 0.981, step / limb / crease unchanged; geom_gain 31 / 31 closed (combined and core), per-pixel relaxed
0.664 → 0.589 (the units were drawn on geom_v0's band, which moves); geom_offset 15 → 12 closed (143, 178, 183: the band
moves off the displaced area onto the edge), relaxed 0.250 → 0.183; texture ROIs unchanged (combined and core);
w < 0.5 deliverable 0.216 / 0.277 → 0.215 / 0.275.

**Rule: met.** Convex median |offset| −1.15 px (≥ 0.5), fold-target coverage change 0.000 (combined and core; ≥ −0.01),
no geom_gain unit loses closure. **`site: hessian_peak` is proposed, and V5 uses it** (approved, fixed, not swept).

**Reading the geom_gain / geom_offset drop.** Both buckets were drawn on geom_v0's band, i.e. on the NMS band position;
their relaxed shares measure how much of that position stays relaxed, so the drop (geom_gain 0.664 → 0.589,
geom_offset 0.250 → 0.183, 3 geom_offset units no longer closed) is the band moving toward the RGB edge, not lost
geometry. **Nosings** stay at about +2.7 px: narrow treads (the peak window meets the next edge), DA3's smoothing of
stairs, and an RGB edge often at the search limit.

**Same-sign neighbours (V2b review).** Synthetic: with two convex kinks 5 px apart (the second stronger) and the first
one's candidate 1.5 px toward the second, the unrestricted peak jumps to the neighbour (site at +4.8 px from the first
kink). So each side of the window is now limited to half the distance to the next accepted fold centreline on that side
along e1 (accepted candidates within 1 px of the line x₀ + s e1, at |s| ≥ 1.5 px); tested: the stronger kink's site
stays on its kink, the weaker one (only a shoulder of the stronger bump after the trilateral filter and σ 1.25) is
invalid rather than displaced. Real images (`scripts/geom_v2_site_neighbours.py`): of the valid sites 18.2 % have
|t| > 2 px; of those, 12 (0.3 %) had another accepted fold centreline within 6 px along n before the limit, and 1 landed
on it; with the limit 5, and 0 land on it. The table above is from the rerun with the limit (geom_v1_hessian rerun on
both batches): validity 0.76 / 0.79, convex median |offset| 2.84 → 1.70 px (−1.15), every other number as above to the
digits shown.

**RGB-edge coverage of all accepted fold candidates** (V2b review; scorer `rgb.fold_candidates`): the fixed set of
geom_v1's 26 466 accepted fold candidates (NMS x₀ along e1), counted only where the RGB edge is reliable: inside the
±8 px window (93.7 %) and with an edge strength max |∇Y · e1| ≥ 3 × the image's median |∇Y| (17 950 kept, 67.8 %; at
1 / 2 / 5 / 10 × the median 83.9 / 74.4 / 56.7 / 44.3 %; the median candidate's edge is 8.6 × the image median).
*Flag: the 3× threshold; lean: 3× (it drops the flat-shading folds whose luminance maximum is noise, and keeps two thirds
of the sample).* Coverage combined / core: geom_v1 0.800 / 0.671, plane_intersection 0.804 / 0.671, hessian_peak
0.809 / 0.696. It is the fourth constraint of the V5 selection rule (≥ geom_v1 − 0.02). Residual: convex −0.77 px median
(frontal convention), vertical corners −0.27; nosings are not corrected (+2.71: a third of them have no valid peak,
and their RGB edge is often a shadow or at the search limit). (plane_intersection fails the same rule: −0.21 px.)

### 8.10 V3 results: short thin depth components (documented gap)

Option `geometry.depth.short_thin: off | strong | linked` (default off; recorded only when on; columns `short_thin`,
`short_thin_link`, scalar `n_short_thin_accepted`). Pieces = D-5 components below `min_component_px` (5) whose every
member is `thin`. `strong`: a piece of ≥ 2 px whose every member has r > τ_high (and, as a survivor, passed the
significance gate) is accepted. `linked` (V0 ruling): pieces are linked end to end when an endpoint gap is ≤ 4 or 6 px
(the sweep), their tangents differ by ≤ 20°, their mean normals point to the same side, and each piece's endpoint lies
≤ 1.5 px from the other's tangent line; a linked chain is accepted if its summed pixels reach 5 and its maximum r
exceeds τ_high. Report `scripts/geom_v3_thin.py` (the depth channel rerun from the Stage A cache per variant, geom_v1's
saved fold and Laplacian weights; `off` reproduces the saved depth w bit for bit) → `v2/thin.json`,
`v2/thin_overlays/<stem>.png`.

| variant (core w unless stated) | off | strong | linked 4 px | linked 6 px |
|---|---|---|---|---|
| newly accepted candidates, batch1 / batch2 | 0 / 0 | 338 / 633 | 0 / 92 | 10 / 144 |
| 25 component-stage step units: closed / relaxed px (of 15 733) | 0 / 66 | 0 / 905 | 0 / 142 | 0 / 197 |
| all 45 open step gaps: closed / relaxed px (of 25 089) | 0 / 465 | 0 / 1 304 | 0 / 541 | 0 / 596 |
| missed true geometry per pixel | 0.568 | 0.558 | 0.568 | 0.566 |
| tri_fp / geom_fp / texture-label units relaxed | 0 / 0.052 / 0 | 0.0006 / 0.052 / 0.0008 | 0 / 0.052 / 0 | 0 / 0.052 / 0 |
| texture ROIs, core and combined | unchanged (combined 0.008 / 0.218 / 0.383 / 0.002) | unchanged | unchanged | unchanged |
| w < 0.5 deliverable, batch1 / batch2 | 0.216 / 0.277 | same | same | same |

**Rule: no variant closes a step-gap pixel**, so none improves on `off` and none is proposed; item recorded as a
documented gap. Every variant meets the texture constraints; `strong` relaxes the most step-gap px (+839 over off,
3.3 % of the open step-gap area), but the rule ranks by closure. Why: the pieces are 1–4 px, so even accepted they add
short fragments of a 2-px band inside units that are tri_veto4's wide relaxation strips (median about 600 px); the
overlays show them on real geometry (073 railing bars and top rail, filling gaps between geom_v1's contours; 090 the
chain-link mesh, as dots; 080 / 075 / 082 / 067 sparse dots in foliage). `linked` finds few chains (pieces separated by
more than 3 px rarely line up within 1.5 px laterally). The open step gaps stay covered by the Laplacian (geom_v1) or,
in geom_v2, by its fallback away from accepted edges.

### 8.11 V4 results: fold recall through undecidable vetoes (targets not met)

Option `geometry.fold.rescue_undecidable: reject | corroborate` (default reject; capped veto only; recorded only when on;
column `rescue_undecidable`, scalar `n_rescued_undecidable`). With `corroborate`, eligible gate failures (Δθ ≥ 8°,
J ≥ J_min) whose second-pass capped veto is undecidable join the crease rescue's components; their support is measured on
the scale-ratio-vetoed response `ctx.L_veto_over_rho_f` (sign(L) min(|L|, g |L_σ − L_4σ|)/ρ on ρ_f, the tri_veto4 veto)
against the same k_L ŝ_L; an undecidable candidate is rescued when its joint component (with the decided-pass
candidates) is kept; the decided-pass components are evaluated exactly as without the option, so it only adds (test);
decided-fail candidates stay out; rescued component ≥ 10 px and support ≥ 70 % unchanged.
Runs with `fold.site: hessian_peak` (V3 review): base `geom_v1_hessian` (reject), `geom_v1_v4` (corroborate), report-only
`geom_v1_v4_c6` (corroborate + rescue minimum 6 px), both batches. Report `scripts/geom_v4_recall.py` → `v2/recall.json`,
`v2/recall_overlays/`.

| metric | geom_v1 | base (hessian) | V4 corroborate | V4 + min 6 px (report) |
|---|---|---|---|---|
| 21 veto-blocked units closed (core; target ≥ 7) | 0 | 0 | **1** (41) | 1 (41) |
| their per-pixel relaxed share (core) | 0.047 | 0.063 | 0.129 | 0.201 |
| open fold / crease gaps closed (core) | 4 | 4 | 5 | 5 |
| core missed true geometry per pixel | 0.568 | 0.563 | 0.508 | 0.486 |
| fold-candidate RGB coverage, combined / core | 0.800 / 0.671 | 0.809 / 0.696 | 0.809 / 0.700 | 0.810 / 0.702 |
| fold-target RGB coverage, combined / core | 0.869 / 0.698 | 0.869 / 0.698 | 0.878 / 0.767 | 0.878 / 0.767 |
| ROI coverage fold / crease, core | 0.817 / 0.399 | 0.817 / 0.399 | 0.907 / 0.399 | 0.907 / 0.410 |
| 076 ledge / 088 ridge coverage, core | 0.130 / 0.762 | 0.130 / 0.762 | 0.130 / 0.762 | 0.130 / 0.787 |
| texture ROIs combined: 079 grass / sunlit / 083 interior | 0.218 / 0.383 / 0.002 | 0.218 / 0.383 / 0.002 | **0.236** / 0.385 / **0.023** | 0.237 / 0.405 / 0.023 |
| texture ROIs core: 079 grass / sunlit / 083 interior | 0.003 / 0.006 / 0 | 0.003 / 0.006 / 0 | 0.070 / 0.010 / **0.021** | 0.079 / 0.037 / 0.021 |
| texture-labelled units relaxed (core) | 0 | 0.008 | 0.011 | 0.013 |
| geom_fp relaxed (core) | 0.052 | 0.044 | 0.044 | 0.044 |
| units 15 / 97 / 124 (core share) | 0 / 0 / 0 | 0 / 0 / 0.02 | 0 / 0 / 0.06 | 0 / 0 / 0.19 |

7 758 candidates are rescued through an undecidable veto (batch1 3 632, batch2 4 126), all of them new.
**Narrow-face split** (another accepted edge within 10 px): narrow 6 535 new candidates, closing unit 41 alone, with
079 grass core 0.053, sunlit 0.008, 083 interior 0, texture units relaxed 0.039; isolated 1 223, closing none, with 079
grass 0.031, sunlit 0.010, **083 interior 0.021**, texture units 0.020 (each group added alone to the base core w).
Overlays: 083 the collar ring (the labelled crease, welcome) but also the dome top and one stretch of the shaft (smooth
curvature: the isolated group); 075 the kiosk roof rim (real) and background trees and façades; the cylinder body mostly
untouched.

**Outcome: not met.** The closure target fails (1 of 21 against ≥ 7) and two fixed constraints fail: 079 grass (combined)
+0.017 > +0.01, the 083 interior changes (combined 0.002 → 0.023, core 0 → 0.021). Nothing is loosened; the option stays
`reject` and is not proposed. The vetoed Laplacian does not reject smooth curvature as the concentration veto does (the
dome and shaft pass it), and the narrow-face cases bring texture through their neighbours. The recall is real but
diffuse: the blocked units' relaxed share doubles (0.063 → 0.129) without crossing 50 %, and the core missed geometry falls
by 0.055 of the tri-only area. In geom_v2 the Laplacian fallback covers these units.

### 8.12 V5 results: fallback Laplacian (no setting passes)

Option `geometry.laplacian.role: channel | fallback` with `d_excl` (default channel; recorded only with the fallback;
arrays `laplacian_w_fallback`, `laplacian_exclusion`; scalars `role`, `d_excl`, `exclusion_share`; `laplacian_w_net` keeps
the channel's own output). `illum_baseline/geom/fallback.py` builds the exclusion zone (shared by Stage B and the sweep;
tested to give the same zone): a Euclidean distance transform at network resolution from the centreline pixels of the
accepted fold candidates (rescued included, at the V2b site) and the accepted depth candidates except thin doublets (sky
edges kept); w_lap′ = 1 within d_excl, the combined w stays min(w_depth, w_fold, w_lap′). Sweep
(`scripts/geom_v5_fallback.py` → `v2/fallback.json`, `fallback_settings.csv`): d_excl ∈ {2, 3, 4, 6} × pool_radius
∈ {2, 3} × band_min ∈ {2, 3} on geom_v1_hessian's tables (fold.site hessian_peak, short_thin off, rescue_undecidable
reject); depth and fold re-rasterized (band_min 2 reproduces the saved weights, 20 / 20), the Laplacian from the saved
geom_v1_hessian (r 2) and geom_v1_r3 (r 3) runs.

| setting d / r / band_min | missed / px | band excess | tri_fp | texture class | 079 sunlit | cov step / fold | RGB depth / fold targets / **fold cand.** | w < 0.5 b1 / b2 | zone share |
|---|---|---|---|---|---|---|---|---|---|
| geom_v1 (reference) | 0.079 | 0.214 | 0.069 | 0.197 | 0.383 | 0.900 / 0.982 | 0.977 / 0.869 / **0.800** | 0.216 / 0.277 | – |
| 2 / 2 / 2 | 0.085 | 0.217 | 0.067 | 0.192 | 0.367 | 0.900 / 0.981 | 0.976 / 0.869 / 0.762 | 0.213 / 0.271 | 0.057 |
| 2 / 2 / 3 | 0.085 | 0.217 | 0.067 | 0.192 | 0.367 | 0.900 / 0.981 | 0.976 / 0.869 / 0.766 | 0.214 / 0.272 | 0.057 |
| 2 / 3 / 2 | 0.007 | 0.256 | 0.082 | 0.248 | 0.488 | 0.909 / 0.999 | 0.991 / 0.907 / 0.777 | 0.237 / 0.298 | 0.057 |
| 2 / 3 / 3 | 0.006 | 0.256 | 0.082 | 0.248 | 0.488 | 0.909 / 0.999 | 0.991 / 0.907 / 0.778 | 0.238 / 0.298 | 0.057 |
| 3 / 2 / 2 | 0.116 | 0.189 | 0.067 | 0.187 | 0.356 | 0.900 / 0.978 | 0.948 / 0.865 / 0.746 | 0.203 / 0.257 | 0.083 |
| 3 / 2 / 3 | 0.088 | 0.217 | 0.067 | 0.189 | 0.359 | 0.900 / 0.981 | 0.976 / 0.868 / 0.759 | 0.212 / 0.270 | 0.083 |
| 3 / 3 / 2 | 0.037 | 0.229 | 0.081 | 0.243 | 0.476 | 0.909 / 0.999 | 0.963 / 0.902 / 0.758 | 0.226 / 0.284 | 0.083 |
| 3 / 3 / 3 | 0.010 | 0.256 | 0.081 | 0.244 | 0.479 | 0.909 / 0.999 | 0.991 / 0.907 / 0.771 | 0.236 / 0.296 | 0.083 |
| 4 / 2 / 2 | 0.148 | 0.160 | 0.066 | 0.183 | 0.346 | 0.900 / 0.978 | 0.919 / 0.851 / 0.724 | 0.192 / 0.243 | 0.107 |
| 4 / 2 / 3 | 0.119 | 0.188 | 0.066 | 0.185 | 0.349 | 0.900 / 0.981 | 0.948 / 0.853 / 0.736 | 0.201 / 0.256 | 0.107 |
| 4 / 3 / 2 | 0.076 | 0.199 | 0.080 | 0.237 | 0.464 | 0.909 / 0.999 | 0.935 / 0.882 / 0.734 | 0.215 / 0.268 | 0.107 |
| 4 / 3 / 3 | 0.047 | 0.228 | 0.080 | 0.239 | 0.467 | 0.909 / 0.999 | 0.963 / 0.884 / 0.746 | 0.224 / 0.281 | 0.107 |
| 6 / 2 / 2 | 0.213 | 0.100 | 0.063 | 0.174 | 0.321 | 0.882 / 0.975 | 0.860 / 0.822 / 0.703 | 0.169 / 0.218 | 0.159 |
| 6 / 2 / 3 | 0.182 | 0.131 | 0.064 | 0.176 | 0.324 | 0.892 / 0.979 | 0.889 / 0.824 / 0.715 | 0.178 / 0.232 | 0.159 |
| 6 / 3 / 2 | 0.153 | 0.127 | 0.077 | 0.226 | 0.433 | 0.890 / 0.992 | 0.872 / 0.836 / 0.707 | 0.188 / 0.238 | 0.159 |
| 6 / 3 / 3 | 0.122 | 0.158 | 0.078 | 0.227 | 0.436 | 0.901 / 0.996 | 0.900 / 0.838 / 0.718 | 0.198 / 0.252 | 0.159 |

(limb coverage 1.00 everywhere; geom_gain 0.589 and geom_fp 0.044 per pixel at every setting.)

**Selection: no setting passes.** Fails per constraint (of 16): fold-candidate RGB coverage ≥ 0.7795 **16** (the binding
one; best 0.7783 at 2 / 3 / 3), missed ≤ 0.0992 7, depth-edge RGB coverage ≥ 0.9566 8, fold-target RGB coverage ≥ 0.8492 4,
step / limb / fold ROI coverage 0. The batch1-only selection fails the same way (no choice; the full-sample and
held-out results agree), so the batch2 check and the all-edges check (both defined at a chosen setting) are not run.
Nothing is widened.

**Why the binding constraint fails** (`scripts/geom_v5_why.py`, at 2 / 2 / 2): 848 of the 17 950 reliable
fold-candidate RGB samples lose coverage, and 846 of them (99.8 %) lie within 3 px of an accepted fold candidate whose own
w_c is ≥ 0.5 (a weak fold, J ≤ J_min + J0, Δθ ≲ 20°): the exclusion removes the Laplacian around edges that relax
nothing themselves. The trade-off across the grid: d_excl ≥ 4 cuts the band excess (0.214 → 0.100 at 6 / 2 / 2) but
loses missed geometry (0.213) and RGB coverage; pool radius 3 keeps the geometry (missed 0.006) but brings tri_veto4's band
excess back (0.256) and texture (079 sunlit 0.488).

**Retirement metric** (core w, V2–V4 options as settled): missed true geometry per pixel 0.563 [0.495, 0.630] at band_min
2 (0.510 [0.447, 0.573] at 3), gap_unresolved alone 0.127 [0.082, 0.176] (0.114 at 3): the Laplacian cannot be retired.

---

# Appendix A. Integration record (TASK_GEOM_INTEGRATION G0–G5)

## A.1 Target architecture and data flow (G0 draft, superseded by section 1)

`geom_v1` is the geometry path of Stage B with channels `[depth, fold]` combined by the elementwise **min** (settled):
a pixel is relaxed if either channel relaxes it, and a place where both fire (silhouettes with a normal change, DA3
contact steps) counts once. Whether the Laplacian channel stays (as a third channel or in a restricted role) is decided
in G4.

```
Stage A cache (per image): D (canonical DA3 depth), sky_prob, meta (focal, frames)       [unchanged; no new DA3 pass]
        │
        ▼
GeometryContext (network res, upright, float64; lazily cached)
  S, Ω, ρ = 1/D, ρ_floor  ──►  ρ_f (trilateral)  ──►  dense fit, n̂, Σ_n (fold only)
  L_over_rho_f (record field; G3 crease rescue reads it, with a new record-only ŝ_L)
        │
        ├──► DepthChannel  (additive null, k_min 11, ΔZ₀ 0.10 m, w_min 0.05)  → w_depth, r (∞ at sky), ΔZ
        │      option G2: depth.sky_band = depth | depth_and_mask
        ├──► FoldChannel   (fold_v2: capped veto, multiplicative gate)          → w_fold, J
        │      option G3: fold.crease_rescue = off | laplacian_corroborated
        └──► [LaplacianChannel]  only if G4 decides (c) (or as corroboration, (b))
        ▼
combine: w_net = min over channels (the only mode)
        ▼
deliverable: upsample w_net to original resolution (bilinear, half-pixel centres), clip [w_min, 1],
             option G2: geometry.deliverable_shift_px = [dx, dy] (network px, applied here only), to the stored frame
        ▼
<stem>.npz: w (original res, stored frame), w_net and per-channel arrays (network res); candidate / null tables;
            JSON; manifest.   Consumers: pair_weights(w) -> (w_x, w_y) at load time (elementwise min of neighbours).
```

Invariants: geometry is pure numpy/scipy, float64, network resolution, upright; no channel reads RGB (only the
evaluation and diagnostic scripts do); no evaluation result feeds back into a channel; D is never shifted or
resampled (the alignment option moves only where the deliverable's relaxation lands).

## A.2 Evaluation harness and disagreement audit (G1)

### A.2.1 `scripts/geom_eval.py` (batch1, pinned ROIs)

- **ROIs:** `scripts/depth_rois.py` (POLYLINES, BOXES, RELABELLED; the 070 sky band from the tri_veto4 sky mask) and
  `scripts/fold_null_rois.py` (TARGETS: narrow-face edges → class fold, `crease` targets → class crease; the texture
  boxes `rois()`: 070 floor, 079 grass, 079 grass sunlit; the 083 bollard interior from `depth_rois.BOXES`, trimmed;
  the legacy box reported separately).
- **Classes:** step (079 silhouettes, 070 sky band, 080 side face edge R), limb (083 limbs), fold (079 lit/shaded fold,
  083 base rim, narrow-face targets), crease (076 façade ledge, 088 rock ridge), texture (the boxes).
- **Metrics per ROI and per class:** share of w < 0.5, median w, and for polylines the coverage = fraction of polyline
  samples (every 0.5 px) within 1 px of a pixel with w < 0.5. Evaluated on the run's combined `w_net` (network
  resolution, upright), which is where the ROIs live; per-channel w is reported alongside for geometry runs.
- **Runs:** any list; G1 uses tri_veto4, tri_veto4_fold_v2, geom_v0. Output `baseline_output/diag/geom/g1/eval.json`
  plus one table.

### A.2.2 Disagreement audit (batch1 + batch2)

- **Maps** (network res, upright): tri-only = tri_veto4 w < 0.5 ∧ geom_v0 w ≥ 0.9; geom-only = the reverse.
- **Components:** 8-connected; components closer than 2 px merged (label the 1-px dilation, assign the original
  pixels its labels); components under 10 px dropped. Full totals per direction and per image (count, area) are
  reported, not only the sample.
- **Sample:** 200 components, about 140 tri-only and 60 geom-only, stratified and area-weighted: within each direction,
  draw without replacement with probability ∝ area, spread over all 20 images by allocating each image's quota in
  proportion to its share of that direction's total area (at least 1 where the image has any component); fixed seed.
- **Contact sheets** `baseline_output/diag/geom/audit/sheet_XX.png`: per component a 4-panel crop (RGB upright at
  network resolution, log Z, tri_veto4 w, geom_v0 w) with the component outlined and its id; about 12 components per
  sheet.
- **Labels file** `baseline_output/diag/geom/audit/audit_labels.csv`: `id, stem, direction, area_px, bbox, label, note`,
  `label` empty; allowed `step, limb, fold, crease, texture, artifact, clutter, unsure` (`artifact`: a DA3 depth
  artefact, no real geometry). Each component's pixel mask is saved alongside (`audit_masks.npz`) so later runs are
  re-scored on exactly the labelled pixels (G4, G5: closed if the new run has w < 0.5 on ≥ 50 % of it).

### A.2.3 As built (G1)

- `configs/candidate_geom_v0.yaml` (`depth_fold_v1` with the additive depth default); runs `geom_v0` in
  `baseline_output/` and `baseline_output_batch2/`.
- `scripts/geom_eval.py` → `baseline_output/diag/geom/g1/eval.json` (per ROI and per class; combined w and each
  channel's w).
- `scripts/geom_audit.py` → `baseline_output/diag/geom/audit/` (as specified: components after merging within 2 px,
  pps draws with replacement per image × direction stratum; `audit_strata.csv` records A_h, N_h, n_h, n_unique per
  stratum; `audit_draws.csv` every draw). G4 estimator: share_c = Σ_h A_h p̂_hc / Σ_h A_h, p̂_hc the unweighted fraction
  of stratum h's draws labelled c.
- Variant `--split-px 1000 --tile 32` → `baseline_output/diag/geom/audit_split/`: components over 1000 px split into
  their pieces per 32-px tile as sampling units (G1 flag: tri-only area is concentrated in a few very large merged
  components).

## A.3 How the open items will be resolved or closed

| # | item | milestone | how |
|---|---|---|---|
| 1 | DA3 alignment shift | G2a, G2b | the D2 diagnostic on 20 images; **stable** if the pooled 95 % CI excludes 0 in an axis and ≥ 80 % of images lie within 0.3 px of the pooled shift; if stable, `geometry.deliverable_shift_px` (default [0, 0], bit-for-bit) |
| 2 | band coverage vs RGB | G2c | coverage of strong depth edges, fold targets and the 070 sky band over band_min ∈ {2, 3, 4} × shift {off, on}; `geometry.depth.sky_band: depth \| depth_and_mask`; texture constraint; selection rule proposed, not switched |
| 3 | ΔZ₀ knee | G3.2 | report deliverable shares and per-class metrics for ΔZ₀ ∈ {0.10, 0.30, 1.00} m; keep 0.10 unless decided (the weight stays a function of ΔZ: ruling) |
| 4 | fusion on strengths vs weights | G5 / G6 | min of weights stays (settled), documented as the decision; strengths documented with units (fold J ∈ [0, 1], depth r ∈ (0, ∞], ∞ at sky saturated) |
| 5 | retiring the Laplacian channel | G4.3 | (a) retire / (b) restricted role / (c) third channel, from the labelled audit (missed true geometry < 10 % of the labelled tri-only area after the G3 rescue → (a)) |
| 6 | local σ_geo (fold) | G4.4 | closed unless the labels show missed folds on textured surfaces; cite fold_channel.md §7 |
| 7 | narrow-face veto (fold) | G4.4 | closed unless the labels show missed plinth-type narrow faces; cite fold_channel.md §6.4, §7 |
| 8 | fence clutter policy | G3.3 | count thin-ring clusters (> 50 % thin accepted candidates per 32-px tile) on 20 images and say where they occur; lean: keep the relaxation, no code; revisit if G4 labels them `artifact` |
| 9 | moderate creases | G3.1 | `geometry.fold.crease_rescue: off \| laplacian_corroborated` (Δθ ≥ θ_rescue, J ≥ J_min, rescued component ≥ 10 px, Laplacian one-signed response > k_L ŝ_L on ≥ 70 % of it); targets: 076 ledge and 088 ridge coverage ≥ 0.5, texture ≤ geom_v0 + 0.01, 083 interior unchanged; else "documented gap" |
| 10 | recall vs tri_veto4 on true classes | G4.1–2 | the labelled audit: tri-only geometry (step/limb/fold/crease) = recall gaps; tri-only texture/artifact/clutter = tri_veto4 false positives; geom-only texture/artifact = geom_v0 false positives; re-scored after G2/G3 settings |

### A.3.1 G2 results (items 1, 2)

- **Shift (item 1): not stable.** 20 images, strong accepted depth edges (accepted, m8, r > 0.2, additive significance
  > 2z, not sky; 4083 samples with |s| ≤ 2; 069 and 074 excluded for < 20 samples). Image-level pooled shift
  (−0.07, −0.47) px (RGB = DA3 + shift), 95 % CI dx [−0.19, 0.06], dy [−0.64, −0.30] (excludes 0); but only 39 % of the
  images lie within 0.3 px of it (median distance 0.34 px, max 0.93; per-image dx −0.54 … +0.54). Rule: stable needs
  both → **not stable; `geometry.deliverable_shift_px` not added**. (`scripts/geom_g2_align.py` →
  `diag/geom/g2/align.json`.)
- **Band coverage (item 2)** (`scripts/geom_g2_coverage.py` → `diag/geom/g2/coverage.json`; w re-rasterized from the
  geom_v0 tables, bit-identical to the saved w_net at band_min 2): depth-edge coverage at the RGB edge 0.819 / 0.846 /
  0.879 for band_min 2 / 3 / 4 (1.000 at the DA3 position); fold targets 0.43 (the narrow plinth edges and creases are not
  detected, not a band question); 070 sky band 0.741 / 0.919 / 0.987 with `sky_band: depth`, 0.985 / 0.986 / 0.987 with
  `depth_and_mask`; texture ≤ geom_v0 + 0.01 everywhere (079 sunlit grass 0.006 → 0.012 at band_min 4). **No setting
  meets the rule** (depth-edge coverage ≥ 0.90): 74 % of the RGB edges lie within 2 px of the depth edge, 10 % beyond 6 px
  and 4.4 % at the ±8 px search limit (no luminance edge at the depth edge); 0.90 would need ~6-px bands.
- New option (default off): `geometry.depth.sky_band: depth | depth_and_mask` (ring radius 2 px within 4 px of an
  accepted sky edge; recorded only when on).

### A.3.2 G3 results (items 9, 3, 8)

- **Crease rescue (item 9).** `geometry.fold.crease_rescue: off | laplacian_corroborated` (recorded only when on; off is
  bit-identical on all 20 images). `ctx.s_hat_L` (B5 on the un-vetoed |L/ρ_f|) equals tri_veto4's recorded ŝ exactly on
  all 10 batch1 images. Sweep θ_rescue ∈ {6, 8, 10}° × k_L ∈ {2, 3, 4}: 088 rock ridge coverage 0.125 → 0.76 at every
  setting, 076 façade ledge 0 → 0.13 at most (its candidates are eligible, 49/53, but only 16 pass the capped veto as
  decided: the ledge is blocked by the veto, not by θ or k_L); texture within + 0.01 and the 083 interior unchanged at
  every setting; the fold class coverage rises 0.47 → 0.82 (narrow plinth edges). **No setting meets the targets**
  (both creases ≥ 0.5) → item 9 closed as a documented gap by the rule. Side effects on batch2: 55–2039 rescued
  candidates per image (8°/3), mostly real small geometry (068 siding laps, 074 downpipes, window frames, ledges;
  w < 0.5 0.046 → 0.103 on 074).
- **ΔZ₀ (item 3)**, mean deliverable share of w < 0.5 over 20 images: 0.090 / 0.084 / 0.072 for ΔZ₀ 0.10 / 0.30 / 1.00 m
  (w < 0.9: 0.106 / 0.105 / 0.102); per class at 1.00 m: step 0.775 → 0.751, limb 1.00 → 0.81, texture 0.003 → 0.000.
  ΔZ₀ stays 0.10 m.
- **Thin-ring clusters (item 8)**: 32-px tiles with ≥ 5 accepted candidates, > 50 % thin, sky edges excluded (a sky edge's
  far side is sky, so it falls back to the doublet and carries `thin`): 1–56 cluster tiles per image; fences, railings
  and balusters (090 chain-link, 066 / 067 picket fences, 071–073 iron railings, batch2 070 / 075 porch balusters, 084
  pipe rail), foliage (080, 075, 082, 067) and distant façades (076, 082). Lean: keep the relaxation (no code); revisit if
  G4 labels such clusters `artifact`.

### A.3.3 G4 results (items 5, 6, 7, 10)

Labels: `audit_split/audit_labels.csv` (190 units; `bucket` used for every count). Re-evaluation run `geom_g4_reeval`
(`configs/candidate_geom_g4_reeval.yaml`: geom_v0 + `sky_band: depth_and_mask` + `crease_rescue: laplacian_corroborated`
8° / k_L 3, both batches). A unit is closed if the run's w < 0.5 on ≥ 50 % of it. Script `scripts/geom_g4.py`, outputs
`baseline_output/diag/geom/g4/` (`g4.json`, `estimates.csv`, `strata.csv`, `units.csv`, `offsets*.csv`,
`open_step_candidates.parquet`).

**Estimator.** Area share per direction = Σ_h A_h p̂_hc / Σ_h A_h (p̂_hc the fraction of stratum h's draws, with
multiplicity); 95 % CI from a stratified bootstrap over draws (B = 10000); area = share × the G1 total. The pooled draw
fraction agrees within 0.01 on tri-only; on geom-only it differs where the minimum-one allocation over-samples small
strata (geom_gain 0.66 vs 0.60, geom_band_excess 0.03 vs 0.07, unsure 0.002 vs 0.017). Six geom-only strata have a single draw (3.3 % of the
geom-only area, marked in `strata.csv`): they add no bootstrap variance, so the geom-only CIs are understated.

Tri-only (665 801 px, 17.5 % of all image px; 140 draws, 136 units):

| bucket | share [95 % CI] | area px | closed by the re-evaluation |
|---|---|---|---|
| gap_fixable | 0.485 [0.414, 0.556] | 322 700 | 0.035 (5 of 65 units: 4 fold, 1 limb, all by the rescue) |
| gap_unresolved | 0.134 [0.089, 0.186] | 89 300 | 0 |
| gap_subscale | 0.034 [0.012, 0.062] | 22 600 | 0 |
| tri_band_excess | 0.256 [0.191, 0.324] | 170 700 | 0 |
| tri_fp | 0.083 [0.047, 0.125] | 55 500 | 0 (no new coverage inside a tri_fp unit) |
| unsure | 0.008 [0, 0.023] | 5 000 | 0 |

Geom-only (39 123 px, 1.0 % of image px; 60 draws, 54 units): geom_gain 0.664 [0.553, 0.767], geom_offset 0.250
[0.154, 0.354], geom_band_excess 0.033 [0.010, 0.060], geom_fp 0.052 [0, 0.104] (3 units, all foliage / surface marks
on DA3 bumps), unsure 0.002. The options only lower w (checked on all 20 images), so every geom-only unit stays covered.
The run adds 51 332 px of w < 0.5: 33 827 inside tri_veto4's w < 0.5, 7 638 where tri_veto4 ≥ 0.9 (geom-only area
39 694 → 47 332 px), 9 867 in between. The sky band closed no audited unit.

**Laplacian decision (item 5).** Missed true geometry = gap_fixable + gap_unresolved not closed: **0.584 [0.512, 0.656]**
of the tri-only area after the re-evaluation (0.619 before; 0.588 excluding unsure; 10.2 % of all image px). By label:
step 0.341, fold 0.172, limb 0.038, crease 0.033 (crease = 5.7 % of the missed area). gap_subscale (0.034) excluded
(below the depth channel's design resolution). Missed ≫ 10 % and not crease-like → **(c) keep the Laplacian as a third
channel**.

**Open step gaps (45 units, 25 089 px).** Stage that drops the edge (most advanced stage any depth candidate within 2 px
reached): component 25 units (15 733 px), significance 6 (2 202), τ_low 6 (3 284), D-3 tests 5 (2 176), sign 2 (1 407),
no candidate above the gradient floor 1 (287). In the component stage 84 % of the best candidates sit in contour
components < 5 px (median 4 px; their r P50 0.10, component max r P50 0.11 > τ_high), 82 % are `thin` (doublet fits):
fragmented thin contours on fences and railings (090, 073, batch2 075, 067). The significance gate does not dominate, so
a Laplacian-corroborated depth rescue (option (b) for steps) is not proposed.

**geom_offset (15 units, all fold candidates; signed s* of the RGB edge along the normal).** Vertical corners (5 units;
+ = toward the more frontal face): median −2.0 px (P25 −2.8, P75 −1.1), every unit negative (−0.8 to −3.2), raw dx
+0.75 to +3.2, all convex. Nosings (5; + = image up): median +2.9 px, every unit positive (+1.7 to +6.9), all convex.
Contact folds (5 incl. 173 by orientation; + = image up): median +1.2 px, mixed (−2.4 to +5.6), concave except 173.
12 % of contact-fold samples sit at the ±8 px search limit. Open fold-placement item for the G6 limitations (not the DA3
shift, which G2 found unstable and ≤ 0.5 px). The labels of 161 / 173 (offset near 0) and 165 / 180 (> 3 px) are kept
(review decision); the RGB edge can be a shadow edge, and the search limit caps the contact-fold samples.

**Items 6 / 7.** Narrow-face units after the re-evaluation: 92 (076 plinth contact fold) closed by the rescue (0.81);
15 (067 picket front/side folds, 0.00) and 37 (070 porch nosing, step, 0.32) open; 171 (075 curb top, gain) stays.
**Item 7 reopens** by the rule (15, 37). Textured-surface fold gaps 97 (fold), 124, 125 (creases) are all open (0.00,
0.00, 0.14): **item 6 reopens** by the rule.

### A.3.4 G5 results: geom_v1

`configs/candidate_geom_v1.yaml`, run `geom_v1` on both batches: channels **[depth, fold, laplacian]**, min;
`depth.sky_band: depth_and_mask`; band_min 2; `fold.crease_rescue: laplacian_corroborated` (8°, k_L 3); no deliverable
shift; **`geometry.laplacian.pool_radius: 2`** (chosen by the rule below). The Laplacian channel is tri_veto4's detector
with the pooling disk reduced from 3 to 2 px. `geom_v1_r3` (`configs/candidate_geom_v1_r3.yaml`, pool_radius null = the
detector's r = 3) is the first geom_v1 run, kept as a reference. Bit-for-bit references: `geom_v1` and `geom_v1_r3`,
batch1 and batch2. Scripts `scripts/geom_g5.py`, `scripts/geom_g5_radius.py` (outputs `baseline_output/diag/geom/g5/`:
`g5.json`, `eval.json`, `radius.json`, `units_v1.csv`, `estimates_v1.csv`); sheets `stage_b/geom_v1/viz_geom/<stem>.png`.

**Pooling radius rule** (`scripts/geom_g5_radius.py`; w = min(depth, fold, Laplacian_r); r = 3 reproduces geom_v1_r3 bit
for bit): r = 2 if its step, limb and fold coverage are each within 0.02 of r = 3, else 3.

| class | share w < 0.5, r = 1 / 2 / 3 | coverage, r = 1 / 2 / 3 |
|---|---|---|
| step | 0.952 / 0.958 / 0.962 | 0.887 / 0.900 / 0.909 |
| limb | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| fold | 0.923 / 0.940 / 0.971 | 0.966 / 0.982 / 0.999 |
| crease | 0.521 / 0.543 / 0.564 | 0.564 / 0.574 / 0.585 |
| texture | 0.147 / 0.198 / 0.254 | – |

Network px with w < 0.5: batch1 0.195 / 0.217 / 0.241, batch2 0.252 / 0.279 / 0.305 (r = 1 / 2 / 3). r = 2: step
−0.009, limb 0, fold −0.017 → **r = 2**.

**G1 table (combined w, batch1 pinned ROIs):**

| class | share w < 0.5: tri_veto4 / geom_v0 / geom_v1 (r = 2) / geom_v1_r3 | polyline coverage: same order |
|---|---|---|
| step | 0.962 / 0.775 / 0.958 / 0.962 | 0.909 / 0.850 / 0.900 / 0.909 |
| limb | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 |
| fold | 0.909 / 0.437 / 0.940 / 0.971 | 0.962 / 0.468 / 0.982 / 0.999 |
| crease | 0.436 / 0.037 / 0.543 / 0.564 | 0.495 / 0.053 / 0.574 / 0.585 |
| texture | 0.254 / 0.003 / 0.197 / 0.254 | – |

**Audit re-scored (G4 labels), geom_v1.** 135 of 136 tri-only units closed (w < 0.5 on ≥ 50 %); open: 72 (075 fold,
gap_fixable, 0.47). Missed true geometry 0.584 (geom_v0) → 0.007 [0, 0.021]; gap_fixable closed 0.478, gap_unresolved
0.134, gap_subscale 0.034 of the tri-only area. tri_band_excess (0.256 of the tri-only area) and tri_fp (0.083) units
are all closed again; per pixel 0.86 of the band-excess area is relaxed (1.00 at r = 3). Geom-only units stay covered.
Named units (w < 0.5 share, median w; all by the Laplacian, fold alone 0.00 / 0.32 / 0.00 / 0.00 / 0.14): 15 0.70 /
0.170, 37 0.78 / 0.275, 97 0.80 / 0.216, 124 0.89 / 0.203, 125 0.86 / 0.214 (r = 3: 1.00 each).

**Marginal contribution** (geom_v1 w < 0.5 area that exists only because of the channel or option; px, share of image px,
share of the w < 0.5 area):

| | batch1 (w < 0.5: 414 222 px) | batch2 (531 222 px) |
|---|---|---|
| depth | 2 (0.0001 %) | 3 (0.0002 %) |
| fold | 34 579 (1.8 %, 8.3 %) | 64 832 (3.4 %, 12.2 %) |
| laplacian | 253 475 (13.3 %, 61.2 %) | 286 907 (15.1 %, 54.0 %) |
| sky_band | 0 | 0 |
| crease_rescue | 7 222 (0.38 %, 1.7 %) | 13 894 (0.73 %, 2.6 %) |

(geom_v1_r3: depth 0 / 0, fold 26 705 / 57 444, laplacian 299 154 / 337 605, sky_band 0 / 0, crease_rescue 5 452 / 12 053.)
Depth's w < 0.5 pixels (63 590 / 132 888) lie inside the Laplacian's w < 0.5 except 2 / 3 px; depth is the binding
minimum on 0.011 % / 0.029 % of the pixels (median lowering 0.03 / 0.02). The sky ring lies inside the Laplacian's sky
band. The fold channel is binding on 3.2 % / 5.1 % of the pixels (median lowering 0.37 / 0.53). The re-rasterized
channel w (with / without the sky ring, with / without the rescued candidates) reproduces the saved w bit for bit.

**Audit closures per r** (w = min(depth, fold, Laplacian_r); report):

| r | gap_fixable closed | gap_unresolved | gap_subscale | missed true geometry | tri_band_excess px relaxed (area share) | tri_fp (area share) | texture: 070 floor / 079 grass / 079 sunlit / 083 interior |
|---|---|---|---|---|---|---|---|
| 1 | 61/65 | 17/18 | 2/5 | 0.036 | 0.71 (0.228) | 10/11 (0.070) | 0.004 / 0.169 / 0.279 / 0.000 |
| 2 | 64/65 | 18/18 | 5/5 | 0.007 | 0.86 (0.256) | 11/11 (0.083) | 0.008 / 0.219 / 0.383 / 0.002 |
| 3 | 65/65 | 18/18 | 5/5 | 0 | 1.00 (0.256) | 11/11 (0.083) | 0.015 / 0.267 / 0.504 / 0.004 |

**Fold-placement diagnostic** (`scripts/geom_g5_fold_site.py`, `g5/fold_site.json`; signed offset of the RGB edge from
the fold site along e1, G4 conventions; plane site = intersection of the two fitted side planes, ρ̂₊ = ρ̂₋):

| set, sub-type | n | planes cross | NMS site median [P25, P75] | plane site |
|---|---|---|---|---|
| geom_offset vertical-corner (5) | 585 | 0.99 | −2.02 [−2.80, −1.09] | −1.05 [−2.22, −0.46] |
| geom_offset nosing (5) | 213 | 0.80 | +2.59 [+2.07, +3.16] | +1.21 [+0.25, +1.78] |
| geom_offset contact (5) | 281 | 1.00 | +0.81 [−1.14, +5.44] | +0.32 [−1.96, +4.20] |
| fold targets vertical-corner (10) | 459 | 0.78 | −1.04 [−2.47, +0.46] | −0.06 [−2.64, +0.95] |
| fold targets contact (6) | 364 | 0.34 | +1.11 [−0.45, +4.11] | −0.77 [−3.78, +5.27] |
| fold targets nosing (2) | 20 | 0.75 | +8.00 (65 % at the search limit) | +6.28 |

All convex candidates in the frontal-face convention: NMS −1.75 px, plane −0.82 px; the plane site lies 0.57 px toward
the oblique face from the NMS site. The hypothesis holds in direction (the NMS peak sits toward the more frontal face)
but the plane site removes only about half of the offset; contact folds (concave) show no consistent effect, and on the
narrow plinth targets the planes often do not cross within ±k₁. Open fold-placement item (G6 limitations).

## A.4 Config and run names

| config | channels | run(s) | milestone |
|---|---|---|---|
| `configs/candidate_geom_v0.yaml` | `[depth, fold]` (depth additive default, fold_v2) | `geom_v0` in `baseline_output/` and `baseline_output_batch2/` | G1 |
| G2/G3 variants (planned as runs) | computed inside the diagnostic scripts instead: re-rasterization of the geom_v0 tables (band_min, sky ring, ΔZ₀) and fold-channel reruns (rescue sweep) in `scripts/geom_g2_coverage.py`, `scripts/geom_g3.py`; no run directories | – | G2, G3 |
| `configs/candidate_geom_g4_reeval.yaml` | geom_v0 + `sky_band: depth_and_mask` + `crease_rescue` 8° / 3 | `geom_g4_reeval` (both batches; re-evaluation, not a reference) | G4 |
| `configs/candidate_geom_v1.yaml` | `[depth, fold, laplacian]` + sky_band depth_and_mask + crease_rescue 8° / 3 + laplacian pool_radius 2 (G4 (c), the approved G2/G3 settings, the G5 rule) | `geom_v1` (both batches; bit-for-bit reference) | G5 |
| `configs/candidate_geom_v1_r3.yaml` | as geom_v1 with pool_radius null (r = 3) | `geom_v1_r3` (both batches; bit-for-bit reference) | G5 |
| `configs/default.yaml` | geom_v1 (= `candidate_geom_v1.yaml`; this commit, branch `proposal/geom_v1_default`) | `geom_v1` | G6 |

New options (each defaults to the old behaviour): `geometry.deliverable_shift_px` (planned; not added, G2: shift not
stable), `geometry.depth.sky_band: depth`, `geometry.fold.crease_rescue: off` (with `rescue_theta_deg` 8, `rescue_k_L` 3,
`rescue_min_component_px` 10, `rescue_support` 0.7), `geometry.laplacian.pool_radius: null` (G5; the detector's r = 3;
recorded only when set). Batch2 runs use the same config with `--input data/batch2
--output baseline_output_batch2`, as batch2 `tri_veto4` was made.

## A.5 What stays bit-for-bit

`baseline_v0`, both `tri_veto4` (batch1, batch2), `[laplacian]`, `fold_only`, `tri_veto4_fold`, `fold_v2`,
`tri_veto4_fold_v2` (`tests/test_baseline_repro.py`, `tests/test_geom_stage_b.py`), with every new option at its
default; plus tests that each option at its default leaves geom_v0 outputs unchanged. G5 added `geom_v1` (pooling
radius 2) and `geom_v1_r3` (radius 3, the first geom_v1 run) on both batches as references. JSON parameter records list a new option only when it is on (as `fold.params_record` and the
`params.geometry.depth` rule already do), so existing records stay identical.

## A.6 Missing pieces this task needs (to be built in the milestones named)

| needed | status | where |
|---|---|---|
| `configs/candidate_geom_v0.yaml` and the `geom_v0` runs (batch1, batch2) | missing | G1 |
| `scripts/geom_eval.py` (per-class harness; class `crease` split from fold) | missing | G1 |
| audit tool (components, sampling, contact sheets, `audit_labels.csv`, `audit_masks.npz`) | missing | G1 |
| RGB edge locator for the alignment / coverage checks | exists in `scripts/depth_d2_report.py` (`luminance`, `rgb_offset`); to be shared | G2 |
| batch2 sRGB for the audit and alignment | exists (`baseline_output_batch2/stage_a/srgb/`) | – |
| `geometry.deliverable_shift_px` (in `outputs.deliverable_w`; remap with an offset) | missing | G2 |
| `geometry.depth.sky_band` | missing | G2 |
| `geometry.fold.crease_rescue` | missing | G3 |
| ŝ for `L_over_rho_f` in the context | **missing**: the context has `L_over_rho_f` but no ŝ; the detector computes ŝ inside `compute_weights` (B5) | G3 |
| combined geometry sheet `viz_geom/<stem>.png` | missing | G5 |
| ROIs / fold targets on batch2 | none (not needed: batch2 is used only where no ROI is needed) | – |
| hand labels for the audit | missing by design (you label the CSV after G1) | G1 → G4 |
| annotations (`annotations/folds/`) | none exist; the pinned ROIs and the audit labels stand in | – |

## A.7 Flags and leans (G0)

1. **Batch2 runs** use the same config with `--input/--output` overrides (as batch2 `tri_veto4`), run name `geom_v0` in
   `baseline_output_batch2/`. *Lean:* yes (no separate config file).
2. **ŝ for the crease rescue.** The task says "use the context's existing L_over_rho_f and ŝ", but the context has no ŝ.
   *Lean:* add a record-only cached field `ctx.s_hat_L` computed exactly as the detector's B5 on |L_over_rho_f| (valid
   = Ω eroded by the support radius, border excluded, median/0.6745, clamped to s_hat_min/max); test that it equals
   tri_veto4's recorded ŝ on all 10 images, so "ŝ" means the same number in both places.
3. **Where metrics are measured.** G1 metrics on the network-resolution combined `w_net` (where the ROIs and the audit
   live). G2 band coverage on the **deliverable** at original resolution (that is where a shift acts): RGB-edge points
   mapped to original upright pixels (× 1/s_x, 1/s_y) and the upright deliverable sampled there. *Lean:* as stated.
4. **Shift convention.** The D2 fit gives RGB edge = DA3 edge + (dx, dy). The option moves the relaxation onto the RGB
   edge, so the deliverable is sampled at x − (dx, dy) (network px, scaled to original px). *Lean:* store and document
   the shift in that sign.
5. **Audit spread.** Per-image quotas proportional to each image's share of the direction's area (at least 1 where
   present), area-weighted draws within an image, fixed seed (0). *Lean:* as stated; the sample is reproducible and the
   totals are reported in full.
6. **"crease" as its own class** in the harness (the two crease targets), separate from fold. *Lean:* yes (the task
   lists it).
7. **Polyline coverage** uses samples every 0.5 px and a 1-px neighbourhood (Chebyshev, i.e. the 3×3 window around the
   sample's nearest pixel). *Lean:* as stated.
