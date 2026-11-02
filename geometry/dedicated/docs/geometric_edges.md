# Geometric edges (geom_v1): the combined depth + fold smoothness-loss weight

Status 2026-11-01: **design draft (TASK_GEOM_INTEGRATION G0), no code yet.** This document becomes the consumer-facing
description of `geom_v1` in G6. Until then it records the target architecture, the evaluation and audit design, how
each integration open item (depth_channel.md §7, items 1–10) will be resolved or closed, the config and run names, and
what stays bit-for-bit. Channel math lives in `docs/depth_channel.md` and `docs/fold_channel.md`, the shared
architecture in `docs/geometry_channels.md`.

## 1. Target architecture and data flow

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

## 2. Evaluation harness and disagreement audit (G1)

### 2.1 `scripts/geom_eval.py` (batch1, pinned ROIs)

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

### 2.2 Disagreement audit (batch1 + batch2)

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

### 2.3 As built (G1)

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

## 3. How the open items will be resolved or closed

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

## 4. Config and run names

| config | channels | run(s) | milestone |
|---|---|---|---|
| `configs/candidate_geom_v0.yaml` | `[depth, fold]` (depth additive default, fold_v2) | `geom_v0` in `baseline_output/` and `baseline_output_batch2/` | G1 |
| G2/G3 variants | via `--set` on `candidate_geom_v0.yaml` | `geom_v0_shift`, `geom_v0_band{3,4}`, `geom_v0_skymask`, `geom_v0_rescue_*`, `geom_v0_dz0_{030,100}` (diagnostic runs, both batches where no ROI is needed) | G2, G3 |
| `configs/candidate_geom_v1.yaml` | from G4 + the approved G2/G3 settings | `geom_v1` (both batches; new bit-for-bit reference) | G5 |
| `configs/default.yaml` | unchanged; a separate, unapplied commit proposes geom_v1 as the default | – | G6 |

New options (each defaults to the old behaviour): `geometry.deliverable_shift_px: [0, 0]`,
`geometry.depth.sky_band: depth`, `geometry.fold.crease_rescue: off` (with `rescue_theta_deg` 8, `rescue_k_L` 3,
`rescue_min_component_px` 10, `rescue_support` 0.7). Batch2 runs use the same config with `--input data/batch2
--output baseline_output_batch2`, as batch2 `tri_veto4` was made.

## 5. What stays bit-for-bit

`baseline_v0`, both `tri_veto4` (batch1, batch2), `[laplacian]`, `fold_only`, `tri_veto4_fold`, `fold_v2`,
`tri_veto4_fold_v2` (`tests/test_baseline_repro.py`, `tests/test_geom_stage_b.py`), with every new option at its
default; plus tests that each option at its default leaves geom_v0 outputs unchanged. `geom_v1` becomes a new
reference in G5. JSON parameter records list a new option only when it is on (as `fold.params_record` and the
`params.geometry.depth` rule already do), so existing records stay identical.

## 6. Missing pieces this task needs (to be built in the milestones named)

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

## 7. Flags and leans (G0)

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
