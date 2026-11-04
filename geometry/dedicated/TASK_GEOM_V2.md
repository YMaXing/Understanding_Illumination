# Task: geom_v2, with the Laplacian restricted to a fallback role

## How to work

- Same project (`geometry/dedicated/`), same session conventions: run everything as `env -u PYTHONPATH uv run ...`.
- Work milestone by milestone (V0–V6). **Stop after each milestone**, report tersely (numbers, tables, figure paths), and wait for my go-ahead.
- **Commit at the end of each milestone** once its tests pass: one commit per milestone on `yx_dev`, no push. Leave untracked task files alone.
- Read first:
  - `docs/geometric_edges.md` (consumer document, limitations, Appendix A);
  - `docs/geometry_channels.md` (§6.1 item status, §7 decisions log, §7.1 geom_v2 candidate list);
  - `docs/depth_channel.md` §7 and `docs/fold_channel.md` §7;
  - `scripts/geom_g4.py`, `scripts/geom_g2_coverage.py`, `scripts/geom_eval.py`, and the G5 fold-site diagnostic.
- Decisions in this file are settled; apply them without asking. Flag any other borderline choice **with your lean**. Do not resolve it silently.
- Every new behaviour sits behind a config option whose built-in default is the old behaviour.
- **Bit-for-bit references stay intact:** every existing reference, including `geom_v1_r3` and `geom_v1` on both batches.
- `configs/default.yaml` is not changed in this task. Switching the default is a separate decision (V6).
- Geometry code stays pure numpy/scipy, float64, at network resolution, in the upright frame. The RGB image may be read **only by evaluation and diagnostic scripts**, never by a channel.
- If a test or target fails for a principled reason, report it. Do not loosen anything silently.

## Goal

Produce `geom_v2`: geom_v1 with the Laplacian channel demoted to a **fallback**. It relaxes only away from accepted geometric edges, so near those edges the depth and fold channels alone set the band. Two prerequisites make this safe, and a fold-recall step reduces what the fallback has to cover:

1. **Fold re-siting.** geom_v1's convex fold bands sit 1–3 px toward the more camera-facing face. Today the Laplacian's wider band hides this; under the fallback it would not.
2. **Short thin depth components.** 63% of the open step-gap pixels are lost at the component minimum (fences, railings), although their r clears τ_high.
3. **Fold recall.** Diagnose where the fold gaps are dropped, then fix the dominant stage.
4. **The fallback Laplacian**, selected by a stated rule against the existing audit labels.

**Why fallback, not retirement.** Without the Laplacian, the missed true geometry is .584 [.512, .656] of the tri-only area (G4). Of this, `gap_unresolved` alone is .134 [.089, .186]: geometry that DA3 depth shows only as a smooth gradient. That share exceeds G4's 10% retirement threshold by itself, so no fix to the depth or fold channel can reach it. V5 still reports the retirement metric.

---

## Settled starting point

- **geom_v1:**
  - channels `[depth, fold, laplacian]`, combined by elementwise min;
  - `depth.sky_band: depth_and_mask`, band_min 2;
  - `fold.crease_rescue: laplacian_corroborated` (8°, k_L 3);
  - `laplacian.pool_radius: 2`;
  - no deliverable shift.
- **Depth defaults:** additive null, k_min 11, ΔZ₀ 0.10 m, w_min 0.05.
- **Fold defaults:** fold_v2 settings (capped veto, multiplicative gate).
- **Labels:** `audit_split/audit_labels.csv` with its `tags` and `bucket` columns. It is final: do not relabel or extend it.
- **Image sets:** batch1 (10 images with pinned ROIs) and batch2 (10 images). Use both wherever no ROI is needed.
- **geom_v1 numbers to compare against** (shares of the tri-only area unless stated):

| metric | geom_v1 |
|---|---|
| missed true geometry (gap_fixable + gap_unresolved), per pixel / per unit | .079 / .007 |
| band excess still relaxed | .214 [.159, .271] |
| tri_fp still relaxed | .069 [.038, .104] |
| ROI coverage step / limb / fold / crease | .900 / 1.00 / .982 / .574 |
| texture ROIs: class share / 079 sunlit / 079 grass | .198 / .383 / .219 |
| w < 0.5 per batch (deliverable) | .216 / .277 |
| missed true geometry without the Laplacian (core w) | .584 |

## Definitions used throughout

- **Combined w:** the run's deliverable weight.
- **Core w:** min(w_depth, w_fold), computed from the saved per-channel arrays. It measures what the geometric channels cover without the Laplacian.
- **Per-unit closure:** w < 0.5 on ≥ 50% of the unit (the G5 rule).
- **Per-pixel share:** the fraction of a unit's pixels with w < 0.5.
- Audit estimates use the stratified estimator with stratified-bootstrap CIs, scaled to the G1 totals, exactly as in `geom_g4.py`.

---

## Milestones

### V0 — Design check (no code)

Add a "geom_v2 design" section to `docs/geometric_edges.md`. It should cover:
- the four parts above, and how each is scored;
- the new options and their defaults;
- run and config names;
- what stays bit-for-bit.

List any tool or annotation this needs that does not exist. **STOP.**

### V1 — Scorer and gap map

**1. `scripts/geom_score.py`:** one scorer for any list of runs, refactored from `geom_g4.py`, `geom_g2_coverage.py` and `geom_eval.py` rather than duplicated.
- **For combined w and for core w:**
  - the G1 ROI table;
  - audit metrics per bucket, both per unit and per pixel;
  - RGB-edge coverage for strong depth edges and fold targets (the G2(c) measure);
  - texture ROI shares;
  - w < 0.5 per batch.
- **Also:** Stage B runtime per image.
- **Baselines:** tri_veto4, geom_v1_r3, geom_v1.
- **Test:** on geom_v1 it reproduces the G5/G6 numbers.

**2. Fold-gap stage diagnostic.** Use the units labelled fold or crease in gap_fixable or gap_unresolved that are still open under core w.
- **Assign each unit** the furthest stage reached by any fold candidate within 2 px:
  - no candidate (below the NMS floor);
  - window validity or contamination;
  - planarity;
  - χ² gate;
  - capped concentration veto;
  - J_min;
  - component minimum;
  - rescue ineligible.
- **Report** units and px per stage, the textured-surface units (97, 124, 125), and the narrow-face units (15, 37).
- **Propose V4's option design** from this table: one option, aimed at the dominant addressable stage, with its targets. Do not code it yet.

**STOP.** I approve the V4 design here.

### V2 — Fold re-siting

**Option `geometry.fold.site: nms | plane_intersection`, default `nms`.**
- **Site.** For each accepted fold candidate, solve ρ̂₊(x₀ + t·n) = ρ̂₋(x₀ + t·n) for t, using its two fitted side planes.
  - **Valid** if a solution exists with |t| ≤ 3 px.
  - **Otherwise** keep x₀. Narrow plinths fall back here (their planes cross for only 34% of contact targets).
- **Smoothing.** Median-filter t along each component over 5 candidates before rasterizing.
  - *Flag:* the filter. *Lean:* as stated, since per-candidate t jitters.
- **What changes and what doesn't.** Δθ, J and the gate are unchanged; only the band's centreline moves. Record t and its validity in the candidate table.

**Report:**
- signed RGB-edge offsets, NMS vs re-sited, per sub-type: geom_offset units, fold targets, all convex candidates;
- the validity fraction;
- fold-target RGB-edge coverage;
- closure of geom_gain and geom_offset units;
- texture ROI shares.

**Rule (propose, don't switch):** propose `plane_intersection` if all of these hold:
- the convex-candidate median |offset| drops by ≥ 0.5 px;
- fold-target coverage drops by at most .01;
- no geom_gain unit loses closure.

Report the residual offset. Do not add any further correction in this task. **STOP.**

### V3 — Short thin depth components

**Option `geometry.depth.short_thin: off | strong | linked`, default `off`.**
- **`strong`:** waive the component minimum for thin-doublet pieces of ≥ 2 px whose every candidate clears τ_high and the significance gate.
- **`linked`:** link thin pieces end to end, then apply the existing minimum to the linked chain. Two pieces link if:
  - their endpoint gap is ≤ 2 px;
  - their tangents differ by ≤ 20°;
  - they have the same polarity.
- *Flag:* the thresholds. *Lean:* as stated, plus a sweep of the gap ∈ {2, 3}.

**Report, under core w:**
- closures (per unit and per pixel) of the 25 component-stage step units;
- all step-gap closures;
- per-pixel relaxed shares of the tri_fp, geom_fp and texture units;
- texture ROI shares, under both core and combined w;
- overlays of newly accepted pieces on 080, 075, 082 and 067 (foliage) and on 090, 073 (fences).

**Rule (propose, don't switch):** the variant that closes the most step-gap px, provided that:
- texture ROI shares (combined w) are ≤ geom_v1 + .01;
- the per-pixel relaxed share of texture-labelled units (core w) rises by at most .02.

If no variant passes, document the gap. **STOP.**

### V4 — Fold recall

Implement the design approved at V1: one option, default old behaviour, with its targets. Whatever the design, these also hold:
- texture ROI shares ≤ geom_v1 + .01;
- the 083 bollard interior is unchanged.

If the approved design reopens item 6 (local σ_geo) or item 7 (narrow-face veto), record the outcome in fold_channel.md §7. **STOP.**

### V5 — Fallback Laplacian

**Option `geometry.laplacian.role: channel | fallback`, default `channel`.**
- **`fallback`:** w_lap′ = 1 within distance d_excl of any accepted geometric edge, and w_lap elsewhere.
  - The combined w stays min(w_depth, w_fold, w_lap′). **The fusion is unchanged; the fallback only masks the Laplacian's input to the min.**
  - **Distance:** a Euclidean distance transform from the rasterized edge centrelines, at network resolution. The centrelines include the V2 site where it is on.
- **Which edges define the exclusion:** accepted depth candidates (including sky edges) and accepted fold candidates (including rescued ones).
  - *Flag:* including the sky edges. *Lean:* include them, since sky coverage is .985 with `depth_and_mask`.

**Sweep:**
- d_excl ∈ {2, 3, 4, 6} × pool_radius ∈ {2, 3} × band_min ∈ {2, 3};
- with the V2–V4 options as proposed;
- use re-rasterization from the saved tables wherever possible.

**Score every setting with `geom_score.py`:**
- combined per-pixel missed true geometry;
- band excess and tri_fp still relaxed;
- texture ROIs;
- ROI coverage;
- RGB-edge coverage (depth edges, fold targets);
- geom-only buckets;
- w < 0.5 per batch.

**Selection rule (propose, don't switch).** Among the settings that meet all of these:
- per-pixel missed true geometry ≤ geom_v1 + .02;
- ROI step, limb and fold coverage each ≥ geom_v1 − .02;
- RGB-edge coverage (depth edges, fold targets) each ≥ geom_v1 − .02;

choose the lowest band-excess share, breaking ties by lower texture.

**Held-out check:**
- repeat the selection using only batch1 audit units, and report the chosen setting's numbers on batch2 units;
- report whether the full-sample choice is the same.
- *Why:* the audit is also the test set, so selecting on it makes the estimates optimistic.

**Retirement metric.** Report the core per-pixel missed true geometry with the V2–V4 options on. Give gap_unresolved separately.

If no setting passes, report which constraint binds and stop. Do not widen the rule. **STOP.**

### V6 — geom_v2 delivery

- **Configuration:** `configs/candidate_geom_v2.yaml`, run `geom_v2`: geom_v1 plus the options I approve.
- **Runs:** both batches, scored with `geom_score.py`.
- **Combined sheet** `viz_geom/<stem>.png`, as in G5 except:
  - panel 7: the Laplacian channel after the fallback, with the exclusion zone outlined;
  - panel 8: geom_v2 vs geom_v1, relaxation removed in red and added in blue.
- **Supplementary audit (conditional).**
  - **Trigger:** the w < 0.5 status flips between geom_v1 and geom_v2, and more than 20% of the flipped area lies outside labelled units.
  - **Sample:** up to 60 components of that unlabelled area (stratified, area-weighted, fixed seed, as in G1).
  - **Output:** contact sheets and a separate `audit_v2_labels.csv`. **STOP** for my labels, then finish.
- **Repro:** add `geom_v2` as a bit-for-bit reference with a test. All existing references stay bit-for-bit.
- **Docs:**
  - `docs/geometric_edges.md`: v2 changes for consumers, limitations with the V5 numbers;
  - `docs/geometry_channels.md` §6–§7;
  - `docs/depth_channel.md` §7, `docs/fold_channel.md` §7;
  - `docs/viz_guide.md`.
- **Default proposal:** prepare (do not apply) a commit on `proposal/geom_v2_default`, with the same table as G6: the w < 0.5 change, marginal area per channel, runtime, new outputs and config keys. I decide whether to apply it.

**STOP.**

---

## Do not

- Change any channel's math or defaults beyond the options listed here. Each new option defaults to the old behaviour.
- Read RGB inside a channel, or let any evaluation result feed back into a channel.
- Implement a fusion other than min. The fallback is a mask on the Laplacian channel's weight before the min.
- Relabel or extend `audit_labels.csv`. Any new labels go in `audit_v2_labels.csv`.
- Tune on the audit beyond the stated grids and rules.
- Replace the ΔZ-based depth weight with a binary one.
- Re-run DA3, change Stage A outputs, or shift D.
- Optimize the fold channel's dense fit (4.5 s of the 7.0 s). That is a separate task.
- Change `configs/default.yaml` before I approve V6's proposal.
