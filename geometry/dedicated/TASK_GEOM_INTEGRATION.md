# Task: integrate the depth and fold channels into the geometric edge detector (geom_v1)

## How to work

- Same project (`geometry/dedicated/`), same session conventions: run everything as `env -u PYTHONPATH uv run ...`.
- Work milestone by milestone (G0–G6). **Stop after each milestone**, report tersely (numbers, tables, figure paths), and wait for my go-ahead.
- **Commit at the end of each milestone** once its tests pass: one commit per phase on `yx_dev`, no push. Leave untracked task files alone.
- Read first:
  - `docs/geometry_channels.md` (architecture, §6 integration notes, §7 decisions log);
  - `docs/depth_channel.md` (§7 open items 1–10);
  - `docs/fold_channel.md` (§6.4 v2, §7);
  - `docs/viz_guide.md`;
  - `scripts/depth_rois.py` and `scripts/fold_null_rois.py`.
- Decisions in this file are settled; apply them without asking. Flag any other borderline choice **with your lean**. Do not resolve it silently.
- Every new behaviour sits behind config options whose built-in defaults are the old behaviour.
- **Bit-for-bit references stay intact:** `baseline_v0`, both `tri_veto4`, `[laplacian]`, `fold_only`, `tri_veto4_fold`, `fold_v2`, `tri_veto4_fold_v2`.
- `configs/default.yaml` is not changed in this task. Switching the pipeline default is a separate decision (G6).
- Geometry code stays pure numpy/scipy, float64, at network resolution, in the upright frame. The RGB image may be read **only by evaluation and diagnostic scripts**, never by a channel.
- If a test or target fails for a principled reason, report it. Do not loosen anything silently.

## Goal

Produce the **geometric edge detector** `geom_v1`: the depth and fold channels combined into one per-pixel smoothness-loss weight, with every integration open item (depth_channel.md §7, items 1–10) either resolved by evidence or closed with a documented reason. The work has six parts:

1. **Evaluation:** a single evaluation harness for any run, plus a *disagreement audit* against tri_veto4 that I label by hand.
2. **Alignment:** the DA3 alignment shift and band coverage (items 1, 2).
3. **Gaps and policies:** the moderate-crease gap (item 9), the ΔZ₀ knee (item 3) and the fence clutter policy (item 8).
4. **Laplacian decision:** whether to retire the Laplacian channel (items 5, 10), based on the labelled audit.
5. **Fusion:** keep the min combination, documented as a decision (item 4); close items 6 and 7 unless the audit reopens them.
6. **Delivery:** the final `geom_v1` config, runs on 20 images, a combined visualization sheet, and consumer-facing docs.

---

## Settled starting point

- **Channels:** `[depth, fold]`, combined by elementwise **min**.
  - The min *is* the union rule: a pixel is relaxed if either channel relaxes it, and a place where both fire (silhouettes with a normal change, DA3 contact steps) counts once.
- **Depth channel defaults:** additive null, k_min 11, ΔZ₀ 0.10 m, w_min 0.05.
- **Fold channel:** fold_v2 settings (capped veto, multiplicative gate).
- **Starting run:** `configs/candidate_geom_v0.yaml`, channels `[depth, fold]`, run `geom_v0`. This is `depth_fold_v1` re-made under the additive depth default.
- **Image sets:**
  - batch1: the 10 images with pinned ROIs;
  - batch2: the 10 images of batch2 `tri_veto4`, which already have Stage A caches.
  - Use both (20 images) wherever no ROI is needed.

## Ruling that applies

**The depth-edge magnitude reflects metric ΔZ.** So the depth weight stays a function of ΔZ: no binary replacement. Only the knee ΔZ₀ is open (G3).

---

## Milestones

### G0 — Design check (no code)

Write `docs/geometric_edges.md` (draft). It will become the consumer-facing document. It should contain:
- target architecture and data flow;
- the evaluation harness and audit design (G1);
- how each open item 1–10 will be resolved or closed;
- config and run names;
- what stays bit-for-bit.

List any ROI, annotation or tool this needs that does not exist. **STOP.**

### G1 — Evaluation harness and disagreement audit

**1. `scripts/geom_eval.py`: per-class metrics for any list of runs, on batch1.**
- **Inputs:** all pinned ROIs (`depth_rois.py` POLYLINES/BOXES/RELABELLED, the 070 sky band; `fold_null_rois.py` TARGETS and texture boxes).
- **Metrics per ROI and per class** (step / limb / fold / crease / texture): share of w < 0.5, median w, polyline coverage (the fraction of the polyline within 1 px of a pixel with w < 0.5).
- **Runs:** tri_veto4, tri_veto4_fold_v2, geom_v0.
- **Output:** `baseline_output/diag/geom/g1/eval.json` plus one table.

**2. Disagreement audit (both batches).**
- **Components:** 8-connected regions where
  - **"tri-only":** tri_veto4 has w < 0.5 and geom_v0 has w ≥ 0.9;
  - **"geom-only":** the reverse.
  - Merge components closer than 2 px and drop those under 10 px.
- **Sample:** a stratified, area-weighted sample of **200 components**, about 140 tri-only and 60 geom-only, spread over all 20 images. Use a fixed seed.
- **Contact sheets** `baseline_output/diag/geom/audit/sheet_XX.png`. For each component, a 4-panel crop with the component outlined:
  - the RGB image (upright, network resolution);
  - log Z;
  - tri_veto4 w;
  - geom_v0 w.
- **Labels file** `audit_labels.csv`: `id, stem, direction, area_px, bbox, label, note`, with `label` left empty.
  - Allowed labels: `step, limb, fold, crease, texture, artifact, clutter, unsure`.
  - `artifact` means a DA3 depth artefact: no real geometry.
- **Also report** the full (unsampled) totals per direction and per image (count and area). The sample is for labelling; the totals give the scale.

**STOP.** I label the CSV, and G4 depends on it. G2 and G3 do not, and can proceed while I label.

### G2 — Alignment shift and band coverage (items 1, 2)

**(a) Is the shift stable?** Repeat the D2 alignment diagnostic on all 20 images:
- strong accepted depth edges;
- the RGB edge taken as the maximum luminance gradient along n within ±8 px;
- the 2D fit restricted to |s| ≤ 2.

Report the per-image shift with its SE, the pooled shift, and the spread. Call the shift **stable** if both hold:
- the pooled 95% CI excludes 0 in at least one axis;
- at least 80% of images lie within 0.3 px of the pooled shift.

**(b) If it is stable, add an option to shift the deliverable.** `geometry.deliverable_shift_px: [dx, dy]`, in network px, default `[0, 0]`.
- Apply it **only to the deliverable**, when w is upsampled and mapped to original resolution. The geometry, the saved network-resolution arrays and the tables stay in DA3's frame.
- The default must keep every reference bit-for-bit. Test this.
- *Why the deliverable and not D:* shifting D would resample disparity (blurring steps) and change every channel's inputs. Shifting w only corrects where the relaxation lands.

**(c) Band coverage.**
- **Depth and fold edges:** for strong depth edges and the fold targets, coverage = the share whose RGB edge (as in (a)) lies within the w < 0.5 band of the deliverable. Measure it for every combination of band_min ∈ {2, 3, 4} × shift ∈ {off, on (if stable)}.
- **Sky:** the 070 sky band coverage (currently 1309/1766). Add an option `geometry.depth.sky_band: depth | depth_and_mask`, default `depth`.
  - `depth_and_mask` also paints w_min on the sky-mask boundary ring (radius 2), restricted to within 4 px of an accepted sky-edge candidate. A misclassified sky region with no accepted sky edge therefore stays unpainted.
- **Constraint:** for each candidate setting, report the texture ROI shares (070 floor, both 079 grass ROIs, 083 bollard interior).
- **Selection rule (propose, don't switch):** the smallest setting that reaches **≥ 90% coverage** of the depth edges and **≥ 95%** of the sky band, with every texture ROI share ≤ geom_v0 + 0.01.

**STOP.**

### G3 — Moderate creases, ΔZ₀, fence (items 9, 3, 8)

**1. Crease rescue (item 9).** Implement it as a fold-channel option, `geometry.fold.crease_rescue: off | laplacian_corroborated`, default off.
- **Which candidates are eligible:** a fold candidate is rescued if it fails **only** the gate (valid windows, veto decided as pass, component size met when rescued) and all of these hold:
  - Δθ(k₁) ≥ θ_rescue (default 8°);
  - J ≥ J_min;
  - its rescued component is ≥ 10 px long;
  - along the component, the Laplacian's un-vetoed one-signed response (`ctx.L_over_rho_f`, sign consistent with the recorded convexity) exceeds k_L·ŝ on ≥ 70% of its pixels.
- **What a rescued candidate gets:** w_c from its own J, as usual. Use the context's existing `L_over_rho_f` and ŝ; do not call the Laplacian channel.
- **Targets:**
  - 076 façade ledge and 088 rock ridge: coverage ≥ 0.5;
  - texture ROIs: share ≤ geom_v0 + 0.01;
  - 083 bollard interior: unchanged.
- Report rescued counts per image.
- *Flag:* θ_rescue and k_L. *Lean:* 8° and 3; report a small sweep.
- If no setting meets the targets, report it and close item 9 as "documented gap".

**2. ΔZ₀ (item 3).** Report the deliverable shares and per-class metrics for ΔZ₀ ∈ {0.10, 0.30, 1.00} m. Keep 0.10 unless I decide otherwise; this is a report, not a switch.

**3. Fence clutter (item 8).** Count thin doublet rings in dense clusters on all 20 images. A cluster is one where more than 50% of the accepted candidates within a 32-px tile are `thin`.
- Report where clusters occur (fence, foliage, other).
- *Lean:* keep relaxation there (the illumination is genuinely discontinuous in clutter), so no code.
- Revisit only if G4's labels mark such clusters `artifact`.

**STOP.**

### G4 — Labelled audit: recall vs tri_veto4 and the Laplacian decision (items 5, 10)

Requires my `audit_labels.csv`.

**1. Report, per direction and per label:** count, area share, and examples.
- **Tri-only components labelled step, limb, fold or crease** are geometry that geom_v0 misses (recall gaps).
- **Tri-only components labelled texture, artifact or clutter** are tri_veto4 false positives.
- **Geom-only components labelled texture or artifact** are geom_v0 false positives.

**2. Re-evaluate.** Run the audit on geom_v0 plus whatever G2 and G3 settings were proposed, to show how many gaps they close.

**3. Laplacian decision.** Propose one of the following, with numbers:
- **(a) Retire the Laplacian channel.** Applies if missed true geometry is under 10% of the labelled tri-only area, after the G3 rescue.
- **(b) Keep it in a restricted role,** e.g. only as corroboration, as in G3. Applies if missed geometry falls in a class the rescue covers.
- **(c) Keep it as a third channel.** Applies if missed geometry is substantial and not crease-like.

*Flag:* the 10% threshold. *Lean:* as written.

**4. Items 6 (local σ_geo) and 7 (narrow-face veto).**
- Reopen only if the labels show missed folds attributable to them: textured-surface folds for item 6, or plinth-type narrow faces for item 7.
- Otherwise close both, citing fold_channel.md §7 and these numbers.

**STOP.**

### G5 — geom_v1: final configuration and runs

- **Configuration:** `configs/candidate_geom_v1.yaml`, run `geom_v1`, containing the channels from G4 plus the G2 and G3 settings I approve.
- **Runs:** batch1 and batch2.
- **Results:** the G1 evaluation table and the audit metrics (re-scored using the existing labels: a component counts as closed if geom_v1 now has w < 0.5 on ≥ 50% of it).
- **Combined geometry sheet** `viz_geom/<stem>.png`, upright:
  1. RGB with accepted depth edges (coloured by log ΔZ), fold edges (coloured by Δθ) and rescued creases (dashed);
  2. log Z;
  3. normal map;
  4. w_depth;
  5. w_fold;
  6. combined w;
  7. tri_veto4 w;
  8. the disagreement map (tri-only red, geom-only blue);
  9. a text box: channel settings, counts, shares, alignment shift used.
- **Repro:** add `geom_v1` as a new bit-for-bit reference, with a test. All existing references stay bit-for-bit.

**STOP.**

### G6 — Docs and default proposal

- **`docs/geometric_edges.md`** (final, consumer-facing):
  - what w means, its frame and resolution, and how to load it;
  - the `pair_weights` semantics;
  - the strengths and their units (fold J ∈ [0,1]; depth r ∈ (0,∞], with ∞ at the sky saturated);
  - the alignment option;
  - limitations;
  - how to re-run.
- **`docs/geometry_channels.md` §6–§7:** the integration decisions and the closed or open status of items 1–10.
- **`docs/depth_channel.md` §7 and `docs/fold_channel.md` §7:** mark resolved items.
- **`docs/viz_guide.md`:** the combined sheet.
- **Default proposal.** Prepare (do not apply) a separate commit that makes `geom_v1` the pipeline default in `configs/default.yaml`, and list exactly what changes for a default run. I decide whether to apply it.

**STOP.**

---

## Do not

- Change any channel's math or defaults beyond the options listed here (each new option defaults to old behaviour).
- Read RGB inside a channel, or let any evaluation result feed back into a channel.
- Replace the ΔZ-based depth weight with a binary one (ruling), or implement a fusion other than min.
- Re-run DA3, change Stage A outputs, or shift D itself.
- Change `configs/default.yaml` before I approve G6's proposal.
- Tune on the 090 fence or on the audit sample beyond the rules stated here.
