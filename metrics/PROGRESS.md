# Metrics v1: progress log

Spec: `metrics/SPEC_v1.md` (Part A §1–§8 normative; Part B phases 0–5). `metrics/DESIGN.md` = Part A verbatim +
"Deltas from Checkpoint 0" (the owner's rulings; they override Part A). This file is the resume point: phase status,
approved decisions, deviations with reasons, open questions.

## Phase status

| phase | status | date |
|---|---|---|
| 0 Recon | done; Checkpoint 0 approved with rulings (DESIGN.md deltas) | 2026-11-18 |
| 1 Schema, I/O, migration | done; Checkpoint 1 approved with rulings (DESIGN.md deltas) | 2026-11-18 |
| 2 Pair metrics and tests | done; Checkpoint 2 approved with rulings (DESIGN.md deltas) | 2026-11-18 |
| 3 Aggregation, diagnostics, CLI | done; Checkpoint 3 approved with rulings (DESIGN.md deltas) | 2026-11-18 |
| 4 Annotation tool | done; Checkpoint 4 trial bug fixed, rulings in DESIGN.md | 2026-11-18 |
| 5 Review mode and docs | done; Checkpoint 5 rulings applied (DESIGN.md deltas) | 2026-11-18 |
| 5b Fresh-clone setup + 100-image split | done; Checkpoint 6 rulings applied (DESIGN.md deltas) | 2026-11-18 |
| 5c Teammate onboarding (ids, assignments, pixel identity, check_pr) | **done, at Checkpoint 7** | 2026-11-18 |
| 6 (queued) Detector-based Retinex baseline | queued after Phase 5 (owner, Checkpoint 3) | |

Git: metrics/ is committed on branch **`metrics-v1`** (owner, Checkpoint 1). The shared working tree stays on `study1`
because a parallel session (`material_edge_detector_upgrade`, Study 1) works in it; switching branches under it would
move its commits. So metrics commits are made with plumbing that never touches HEAD, the index or the working tree:
`metrics/tools/commit_metrics.sh "message"` (temporary index from `metrics-v1`, `git add metrics/`, commit-tree,
update-ref). On `study1`, metrics/ shows as untracked; that is expected.

## How to run

- `metrics/run.sh SCRIPT.py ...` / `metrics/run.sh -m pytest -q -p no:cacheprovider` (drops PYTHONPATH, no bytecode).
- Env: `metrics/pyproject.toml`, python 3.12, pinned to the log_chromaticity env's exact versions (numpy 2.5.4,
  scipy 1.18.1, opencv-headless 5.0.0.93, tifffile 2026.10.15, imagecodecs 2026.10.10) + pyyaml 6.0.3, pytest 9.2.0.
  `uv sync` in metrics/ (UV_LINK_MODE=copy on this WSL mount).
- `migrate_v0.py run [--force] | verify | counts`; `tools/ck1_report.py` (read-only reports → `runs/ck1/`).

## Approved decisions (Checkpoint 0, 2026-11-18)

Full text in DESIGN.md "Deltas from Checkpoint 0". Short form:
B1 interim ℓI hook = illumination_edge/common.py (imported); parity vs frozen_v2_pairs_fullres.csv at 5e-6; prediction
`<pred>/<stem>.npz` key `log_R` float32 (H, W, 3) natural log, optional `log_L` (identity checked); Phase 3 baselines
only; report black-level pedestal read-only. B2 pluggable region source; detector_split/D4/A1/A2 = "n/a: no region
source"; never derive regions from open contours. B3 eval list in config, all annotated images eval, train empty,
per-image seen_in_training default unknown. C1 sat 65000 (config); report [64224, 65000) counts. C2 near-black floor
not in the mask. C3 square patches, annotated half-size per end; disc a config option; keep the auto-sizer in the
tool; report low-SNR for both shapes on dev. C4 type decides near/distant; lengths = tool warnings + flag; 80–150 px
pairs scored as near, flagged, in the headline. C5 migrate all four sets with source + hash, all eval; headline = dev v2
+ held-out; glossy dev / confirm separate strata; deleted pairs in legacy, never scored. C6 per-pair annotator (CSV
column everywhere; migrated = "Yueren"); tool name-tag selector, annotators.json, no anonymous pairs,
last_edited_by, filter/count by annotator, guide section; merge command (union by id, report conflicts); annotator
stratification. C7 own uv project. C8 new tool. C9 old keys stay (u = undo); confidence toggle on another key.

## Approved decisions (Checkpoint 1, 2026-11-18)

Full text in DESIGN.md "Deltas from Checkpoint 1". Short form: C8 new tool in metrics/annotate/. Deviations 1–4
approved. Q1 noise and texture separate: low_snr from the imported sensor-noise model SE (still excludes);
se_tex = MAD SE on ℓI, reported + textured/smooth stratum, never excludes; R-flat types (i, s, same-material d):
E ≤ τ_eff = max(τ, 2·se_tex) and H_R ≤ max(τ, H_I); m keeps plain τ; synthetic textured material (GT passes;
report its i failures under plain τ); Checkpoint 2 shows plain vs widened side by side. Q2 lit_end as inferred;
list 13 corrected + 2 marginal in the Phase 3 report. Q3 commit metrics/ incl. annotations/ on metrics-v1; runs/
and caches ignored. No further writes outside metrics/ (the 9 venv .pyc stay). Empty types report "n/a: 0 pairs".

## Phase 1: what was built

- `DESIGN.md`: Part A verbatim (diff-checked against SPEC_v1.md lines 11–200) + deltas.
- `config.yaml`: all Part A constants and the rulings; sets with source SHA-256; eval lists (46 images).
- `schema/pairs.schema.json` + `core/schema.py` (stdlib validator for exactly the keywords the schema uses; a test
  fails if the schema uses any other keyword). The Phase 4 server will reuse it.
- `core/config.py` (config, paths, `import_hook` by file path, never copying), `core/io.py` (load_image via the hook,
  validity mask, load_prediction with the contract checks, derive_log_L), `core/patches.py` (square/disc, median,
  MAD SE, 70% rule, low-SNR guard).
- `migrate_v0.py`: 46 files in `annotations/` (+ `annotators.json` = ["Yueren"]); `migration_v0/manifest.json`
  (source headers, code/config SHA-256, annotation file SHA-256, relabels, lit_end summary) and `counts.csv`
  (with the annotator column). `verify` rebuilds every source's pairs and deleted lists exactly.
- `tests/`: 44 tests (schema cases, lossless migration + tamper detection, 070 ruling, defaults, parity, validity mask
  = detectors' valid_mask, hook imported not copied, identity incl. degenerate R = I / R = 0, log_L key check,
  prediction contract rejections, patch shapes, out-of-bounds counting).
- `tools/ck1_report.py`: read-only Checkpoint 1 reports.

## Phase 1 results

- Migration: 5011 → 5011 pairs (dev v2 871, held-out 1098, glossy dev 1428, glossy confirm 1614); 070
  t=1794165897383 m → b (dev m 417 → 416, b 28 → 29); 387 deleted v0 records kept in legacy. verify OK.
- lit_end (§3 brighter b(P(ℓI))): 2312 i pairs; 15 have B brighter, i.e. disagree with the guides' "A lit" rule
  (13 held-out, 2 glossy confirm; dev 0). 13 differ by 1.0–2.1 in b (clear A/B swaps, mostly "major" tagged); 2 are
  marginal: heldout baskar_akilan_068 t=1794268381568 (0.208), glossy_confirm visaria_keval_042 t=1794796442046 (0.104).
  No fallbacks (every i patch has valid pixels).
- Parity: 1737 non-empty dev patch medians × 3 channels: max |dev| 5e-6 (= the CSV's rounding); **all 1737 identical
  after the exporter's round(x, 5)**; valid-pixel counts identical for all 1742 patches (5 have 0 valid px).
- Identity: exact in float64 (0 deviation); log_L key check rejects a 1e-3 perturbation.
- Black level: no TIFF metadata beyond baseline tags (12 structural tags; no BlackLevel/DNG/EXIF). Per-channel
  minima 0–1696 DN, 0.1th percentiles 26–2045 DN, varying widely by image and channel; 7 images have pixels with a 0
  channel (075: 421, amir_adnan_044: 85, 076: 27, baskar_akilan_013: 11, 079 / 084 / baskar_akilan_030: 1). Exact
  zeros and near-zero minima suggest any pedestal was already subtracted (clipped at 0) before the TIFFs were
  written; nothing indicates a residual constant pedestal. Convention unchanged.
- [64224, 65000): 1–1029 px per image (dev max 083: 1029 = 0.13%); full table in `runs/ck1/ck1.md`.
- Low-SNR (§2 guard, MAD SE, tau 0.10): see "Open questions" Q1.

## Phase 2: what was built

- `core/patches.py`: + sensor-noise SE of the median (`se_noise`, from the imported `estimate_noise_robust` model:
  per pixel var_log = (g I + r2) / I², SE = 1.2533 sqrt(mean var_log / n)); `combined_se` (RMS_c sqrt(SE_A² + SE_B²)).
- `core/profiles.py`: band sampling (bilinear, validity-aware), per-t band median, normalized Gaussian smoothing,
  excursion H; optional raised sigma_t for textured pairs (config `profile.sigma_t_textured_px`, **null = ruled**).
- `core/pair_metrics.py`: `measure` (tau-free: Δ vectors, norms, alpha, H_I/H_R/H_L, se_noise, se_tex, flags) and
  `judge(rec, tau)` (low_snr, alpha_defined, structured_path, plain and widened pass, bins, b both-step, exclusions,
  headline group); kinds i, m, b, s_random, s_targeted, d_R, d_L, d_unscored; split=train raises.
- `core/baselines.py` (all_R, all_L, blur_L σ 8), `core/io.py` + `noise_params`, `geom_edge`, `context`;
  `core/summary.py` (per-kind error rates; empty -> "n/a: 0 pairs").
- `tests/synth.py` + `tests/test_synth.py` (16 tests): scene with 4 smooth materials, a textured one, a low-contrast
  chromatic patch, a soft coloured penumbra (linear-space mixing), shading, Poisson-Gaussian noise; 64 pairs of every
  kind. GT passes, exact all-R / all-L bins, blur-L profile, gap leak (E decays), halo, scale invariance (global and
  per channel, GT and blur-L), identities, noise guard, train refusal, empty types, excursion basics.
- `tools/texture_mc.py` (texture-tolerance Monte Carlo), `tools/ck2_report.py` (baselines on the real annotations;
  `runs/ck2/ck2.md`, per_pair_<baseline>.csv with the annotator column).

## Phase 2 results

- Tests: 60/60 pass (44 Phase 1 + 16 synthetic).
- Synthetic seed 0: GT passes every scored pair (59); textured i pairs failing under plain tau: **0/12**. Gap leak
  E 0.825 / 0.443 / 0.228 / 0.060 / 0.016 / 0.002 / 0.000 at d 0 / 20 / 40 / 80 / 120 / 180 / 240 px (first 3 fail).
- Texture Monte Carlo (30 seeds, 25 textured i pairs per r, `runs/texture_mc.txt`), GT false-fail rates:
  - grey SD 0.10 (moderate, se_tex ≈ dev's upper range): plain 2.3–13.5%; ruled widened 1.6–8.0% (E 0–4.0%, H 1.6–4.0%).
  - grey SD 0.20 (strong): plain 76–88%; ruled widened 39–45%, mostly H (32–42%).
  - Cause of the H failures: H_I is the distance to a segment along the shadow direction, so texture along that
    (mostly grey) direction is projected out of H_I but not out of H_R; H_R > H_I on ~40–57% of GT pairs.
  - sigma_t 3 px for textured pairs (§2 "raise sigma_t for textured materials"): strong-texture widened 16–24%, halo
    detection 91–100%; 4 px: 10–19%, halo 78–100%; 6 px kills halo detection (~50%).
  - E rule k_tex 2 / 2.5 / 3: GT E false fails ≤ 7.5 / 2.4 / 0.7%.
- Real annotations, headline (dev_v2 + heldout; 926 i, 917 m, 126 b; no s, d): all_R Split 0.500 (plain) / 0.499
  (wide), i 924/925 fail (922 full leak), m 0/903; all_L Split 0.496, i 0/925, m 895/903 (817 full leak, 78 n/a);
  blur_L i H_R > tau 709/925, E_i median 0.076 (patches ≥ 16 px from the midpoint, 127 pairs) vs 0.588 nearer.
  Glossy sets behave the same (table in `runs/ck2/ck2.md`).
- Headline exclusions: b 126, low_snr 15 (noise model; was ~40% under MAD SE), invalid 8.
- Flags on headline pairs: geom_cross 812/1969 (geom_v1_1 w < 0.5 covers 10–31% of each image), textured 674,
  length_out_of_range 30, structured_path 828 (i 20%, m 62%, b 60%; see D2).

## Phase 3: what was built

- `core/aggregate.py`: headline numbers (Split = mean(ER_i, ER_m), Leak = ER_s_random, Long-range = ER_d_R; micro over
  eligible pairs; "n/a: 0 pairs" propagates; composite only when all three exist, never alone), plain alongside
  widened, counted exclusions per kind and tau, tau sweep, vectorised cluster bootstrap over images (2000, seed
  20261118; undefined resamples dropped and counted), companions, bins, b diagnostics, strata (§5 list + textured,
  annotator, length_out_of_range), m with/without input_excursion, input_excursion × textured × geom_cross, per image.
- `core/dense.py`: D1 (w_union ≥ 0.9), D2, D3 (material px off geom_v1_1 w < 0.5 and > 2 px from illumination px),
  central differences valid where the pixel and its 4 neighbours are valid; D4 / A1 / A2 = "n/a: no region source";
  missing maps = "n/a: map not available" (never re-derived). Maps in config `maps.smoothness_weight / illum_edges /
  material_edges` per set: dev and held-out all three; glossy_confirm D2 + D3 only; glossy_dev none.
- `run_metrics.py`: CLI (`--pred`, `--name`, `--images` file / list / set names, `--out`); always runs all_R, all_L,
  blur_L; writes per_pair.csv (annotator column, pass/excluded per sweep tau), profiles.npz (concatenated t, I, R, L +
  index), per_image.csv, summary.json (config, git state read-only incl. whether metrics/ matches metrics-v1, map
  meta + per-image map SHA-256, geom SHA-256, annotation and prediction SHA-256, texture floor), summary.md.
- `thresholds.bin_eps` = 1e-4: numerical slack at the 0 and 1 bin edges. Found by the CLI test: a float32 ℓR = ℓI
  gives alpha ≈ −1e-7 and binned "reversed" instead of "full_leak" (alpha error for float32 inputs is <~ 1e-5 because
  alpha needs ||ΔℓI|| >= 0.2).
- `tests/test_aggregate.py` (5): n/a propagation, low_contrast per tau, bootstrap determinism / coverage / one
  cluster, dense degenerate values (all_R D1 = 0, D2 = 1, D3 = 0; all_L D2 = 0, D3 = 1; GT D2 < 0.05), CLI --pred
  path (float32 ℓI as a prediction scores exactly like all_R).

## Phase 3 results (runs/baselines_v1, 46 images, baselines only)

- Tests 65/65. Run time 81 s.
- Headline (dev_v2 + heldout), widened (plain): all_R Split 0.499 (0.500) [0.498, 0.500]; all_L 0.500 (0.500);
  blur_L 0.949 (0.951) [0.932, 0.966]. Leak, Long-range, composite: n/a (0 s / d pairs).
- Exclusions (headline): b 126; m low_contrast 29, low_snr 11, invalid 7; i low_snr 1. low_contrast by tau 0.05 /
  0.10 / 0.20: 5 / 29 / 89.
- Dense (headline, 25 images): all_R D1 0, D2 1, D3 0; all_L D1 0.090, D2 0, D3 1; blur_L D1 0.011, D2 0.92, D3 0.13.
- lit_end list: 15 pairs (13 clear swaps, 2 marginal), in summary.md.

## Phase 4: what was built

- `annotate/prepare.py` (repo deps): cache per image in `annotate/cache/<stem>/` (git-ignored, ~11 MB per image,
  498 MB for 46): raw.u16.gz (loader.read), lI.f32.gz (loader.log_i), valid.u8.gz (metrics validity mask),
  illum / material / geom overlay PNGs where the maps exist, meta.json (noise model, tone reference, ℓI range);
  `cache/manifest.json` carries every constant, so the server never parses YAML.
- `annotate/server.py` (stdlib only): static files + JSON API; ETag / If-Match (409 on stale writes); every PUT is
  schema-validated (core/schema.py) plus integrity rules (image / set / size fixed; s_seed_log append-only with
  immutable index / seed / x / y / proposed_*; an existing pair keeps annotator and created; new pairs and
  last_edited_by must be known name tags; ids unique across pairs and tombstones); atomic write after a timestamped
  backup in `annotations/.backups/<stem>/` (git-ignored); seeds from random.Random(f"{seed_base}:{stem}:{index}"),
  uniform over valid pixels >= default_r from the border, logged on proposal. No route to predictions or runs.
- `annotate/index.html`, `style.css`, `lib.js` (pure helpers: patch stats on the cached ℓI, noise-model SE, se_tex,
  lit-end inference, the old auto-sizer, warnings, tone map), `app.js` (UI). `annotate/run.sh` launches
  (prepares missing images, prints the URL). `annotate/merge.py`: union by pair id, tombstone-aware, conflicts
  reported and never overwritten, dry run unless --apply. `annotate/selftest.py`: headless Windows Chrome against a
  throwaway copy (22 in-page checks + 3 on disk) and a screenshot (`runs/annotate_selftest/`).
- `tests/test_annotate.py` (11): round trip + backup + 409, six rejection cases, seed log append-only /
  reason-required / cross-machine determinism, annotators trim + case-insensitive de-dup, 404 routes (traversal,
  runs, config, source), merge.
- Results: pytest 76/76; browser self-test 22/22 + 3/3; the real annotation files are byte-identical to metrics-v1.

## Phase 4 decisions (approved at Checkpoint 4)

- **Keys:** old keys kept (u undo; e x4 exposure; x invalid pixels; [ ] half-size, Alt+[ ] active end; g auto-size;
  w swap; Delete). z also undoes (spec). **q toggles confidence** (C9: moved off u). t toggles lit end; N note;
  1 2 3 overlays (4 regions disabled: n/a); r random s seed; n / p images; f fit; ? help.
- **s pairs:** a plain click-click s pair is a targeted s and needs an overlay on (gap_ref asked); random s pairs come
  only from the seed button (pool random). overlay_seen is set at creation (an overlay visible then).
- **i pairs:** lit_end is asked with the inferred (brighter) end pre-selected; Enter accepts.
- **Auto-sizer:** the old rule (largest r <= 4, both patches fully valid, plateau shift 3 px, tol 0.10) with the metrics
  validity mask; the old tool also refused near-black pixels (C2 keeps them out of the mask).
- **Schema extension:** optional top-level `deleted_pairs` (tombstones: id, deleted_at, deleted_by, pair) so merges
  do not resurrect deleted pairs; `s_seed_log` items now have a schema (index, seed, x, y, proposed_at, proposed_by,
  outcome pending / accepted / nudged / rejected / abandoned / undone; rejected needs a reason).
- **Not added:** a live "segment crosses a geometric edge" warning for m / s pairs (the placement rule): it would
  expose detector output while overlays are off. Lean: leave it to the geom overlay and the geom_cross flag.

## Checkpoint 4 trial and fix (2026-11-18)

- **Bug:** while any dialog was open, the global key handler forwarded every keydown to the dialog and called
  preventDefault(), which cancels the browser's text insertion: typed characters never reached the name / note /
  gap_ref / reason fields. The old self-test missed it because it set `.value` and called the handler directly.
- **Fixed:** hotkeys off while a dialog is open and for editable targets; dialogs handle only their own keys; focus is
  released when a dialog closes (a hidden focused field would swallow the next hotkey); the key that opens a dialog
  is prevented from typing itself into it (Shift+N used to start every note with "N"; found by the guide
  screenshots); name-tag errors from the server are shown in the dialog (new server rules: <= 64 characters, no
  control characters); password-manager opt-outs on every input.
- **Self-test** now drives Windows Chrome over the DevTools Protocol from PowerShell (`annotate/cdp_driver.ps1`;
  WSL's NAT networking cannot reach the debugging port): real key events (the name tag typed key by key), real
  insertText, real mouse clicks and drags. 35/35 checks.
- **Trial data:** the owner's trial wrote two random-s seed proposals (index 0 abandoned, index 1 pending, as
  "Yueren") and an empty `deleted_pairs` to `annotations/acharya_mrudula_048.json`; it was committed with the fix
  (f0c0759). The pre-trial file is in `annotations/.backups/acharya_mrudula_048/`. Open question Q5.

## Phase 5: what was built

- **Groups:** `aggregate.groups` (dev = dev_v2 + heldout, headline; confirmation, held-out headline; glossy strata).
  per_pair.csv has `group` and `set` (provenance). Reports are per group.
- **Confirmation candidates:** `tools/confirmation_candidates.py` -> `confirmation/candidates.json` (seed 20261118,
  347 eligible of 503) and `runs/confirmation/candidates.png`. Excluded: 46 annotated, 40 glossy, acharya_mrudula_068
  (held-out protocol), visaria_keval_087 (dropped by the held-out annotator), and **positions 17–109 of the held-out
  draw sequence** (FRESH_PROTOCOL.md reserves 17–25 and counts 26–109 for the fresh bundle test): Q6.
  `tools/init_images.py` creates empty files for the pick; then `annotate/prepare.py --images LIST`.
- **Targets:** dev means per image i 37.0, m 36.7, b 5.0 (medians 33 / 33 / 3) -> config i 37, m 37, b 5; random s
  12–15; d 12–15, >= 50% same material / different light. Progress bars in the pair list panel.
- **Blind relabel mode** (Shift+R): queue from `GET /api/relabel/queue/<stem>` is geometry only (pair_id, A, B), in a
  fixed per-relabeler shuffle, other annotators' pairs by default (own pairs optional, for test-retest; flagged
  `own_pair`); type, then lit end (i) / lighting (s) / material + illumination (d), then confidence; `k` skips;
  Esc leaves. Pair list, editor, progress and overlays hidden. Results: `annotations/relabels/<stem>.json`
  (schema/relabels.schema.json; append-only, one per (pair, relabeler); backups). merge.py unions relabels.
  `tools/relabel_agreement.py`: per-type agreement, confusion matrix, Cohen's κ for type and for d material /
  illumination, lit-end agreement; cross-annotator and own-pair reported separately.
- **Review mode** (`annotate/review.html`, `review.js`): only when the server runs with `--review` (otherwise 404), so
  annotation mode has no route to run data. Run / method / image pickers; pairs coloured pass / fail / excluded / b;
  filters by type, status, bin, flag; selected pair: numbers, A / B patch crops, p_I / p_R / p_L plots with the
  endpoint segment (dashed) and H marked. profiles.npz is read with a stdlib .npy reader (`annotate/npyread.py`).
  run_metrics now writes PA_I, PB_I, PA_R, PB_R.
- **Docs:** `docs/ANNOTATION_GUIDE.md` (24 screenshots in docs/img/, 2.6 MB, from `annotate/guide_shots.py`; the
  pair-type examples are demonstration pairs placed by eye on taiwei_cui_080 and drawn for the picture only);
  `README.md`.
- **Tests:** pytest 80/80; browser self-test 35/35 (adds progress, relabel, review).

## Open questions (Checkpoint 5, resolved)

- **Q5. Trial data in acharya_mrudula_048.json** (two seed entries, one still "pending"). Lean: keep (the seed log
  is append-only by design and the entries are honest). Note: the tool does not resume a pending seed when an image is
  reopened, so that entry stays "pending" (harmless: only resolved entries are used). Alternative: restore the
  pre-trial backup (outside the tool).
- **Q6. Confirmation draw excludes the fresh-protocol span 17–109** (not in the ruling). Lean: keep the exclusion so
  the fresh bundle test stays untouched; `--keep-fresh-span` regenerates without it.
- **Q7. Relabel scope:** own pairs are opt-in and reported separately (test-retest). Lean: keep.

## Checkpoint 5 rulings applied (2026-11-18)

- Q5: acharya_mrudula_048.json restored from its oldest backup (byte-identical to the migrated file), committed alone
  (ab8accd). The owner's re-test added the tag "Yueren_test" to annotators.json (12:20 local; no pairs placed): removed
  as trial data in line with Q5 (flagged in the Checkpoint 6 report). `tools/commit_metrics.sh` now commits code
  (everything but annotations/) or `--annotations` (only annotations/), never both.
- Pending seeds: offered on open (resume / abandon); every abandon asks for a reason; no new seed while one is pending;
  undo re-offers. Found while testing: a dialog's Enter also reached the global hotkeys after the dialog closed
  (resuming a seed accepted it at once); dialog keys now stop propagation.
- Targets i 33, m 33, b 3; exhausted marks (`exhausted.{i,m,b,s_random,d}` = {by, at, note}; server checks `by`).
- Confirmation set dropped (tools/confirmation_candidates.py and confirmation/ removed).
- Setup: `config.local.yaml` (+ example); dataset = one folder (`val_set`; all 100 images are byte-identical there);
  map folders, the parity CSV and browser paths moved out of the tracked config; `reference/glossy_pool.json` (tracked
  list of the 40 glossy-pool stems, so draws do not need the image_data/glossy folder); prepare.py SHA-256 check;
  `uv.lock` tracked (metrics/.gitignore re-includes it); `doctor.py`; `tools/overlays.py` export / import; tests skip
  without local files; `docs/SETUP.md`.
- **Dependency audit:** `illumination_edge/common.py` (tracked) imports `log_chromaticity/material_edges.py` (git-ignored)
  at import time; the hooks do not use it, so `core/config.import_hook` registers an inert placeholder when the file
  is absent (config `loader.optional_imports`; any use raises; doctor reports it). Q8.
- 100-image split: `tools/draw_images.py` (54 new; first 25 = the former confirmation candidates, same seed and
  rules), `tools/assign_splits.py` (`splits/split100.json`), `tools/upgrade_schema_1_1.py` (46 files), `tools/init_images.py`
  (54 new files). Table: A glossy 21 -> 13 / 4 / 4; A non-glossy 25 -> 15 / 5 / 5; B 54 -> 32 / 11 / 11; total 60 / 20 / 20.
  By provenance: dev 15 / 5 / 5, glossy_dev 8 / 1 / 3, glossy_confirm 5 / 3 / 1, draw54 32 / 11 / 11.
- run_metrics: blocks per data split (train, val; test only with --test, not even loaded otherwise), supervised-fit
  blocks for used_in_loss pairs, train - val gap table, strata by provenance group / annotator group / glossy pool.
- Tests 92/92; browser self-test 40/40 (adds pending-seed resume / abandon across a reload, exhausted marks, split
  label). Guide screenshots retaken (27).

## Fresh-clone verification (2026-11-18)

`git clone --branch metrics-v1` into a scratch folder (no log_chromaticity/, no outputs/, no cache), then SETUP.md
steps 2–6: `uv sync --locked` OK; doctor before prepare: only the cache missing (+ the material_edges placeholder
note); pytest 75 passed / 17 skipped (clear messages); `tools/overlays.py import` of an archive exported here (46
images, 114 overlays, 2.7 MB): 46 imported, 0 refused; `annotate/prepare.py`: 100 images in 1 m 45 s; doctor READY;
pytest 89 passed / 3 skipped (parity CSV, local-map export). Found and fixed on the way: scripts were committed
without the executable bit (NTFS mount, core.fileMode false): the commit helper now sets +x on metrics/*.sh. Not run
in the clone: the browser self-test (needs the Windows side; it passes here, 40/40).

## Open questions (Checkpoint 6, resolved)

Q8 keep the placeholder, fail loudly; Q9 keep the paths, verify is maintainer-only; Q10 the tag stays removed
(DESIGN.md, Deltas from Checkpoint 6).

## Checkpoint 6 rulings applied (2026-11-18)

- **Schema 1.2** (`tools/upgrade_schema_1_2.py`, run once on the 100 files; idempotent): annotator names → ids in
  pairs, tombstones, seed logs and exhausted marks; `assigned_to`; `image_pixel_sha256`; `reflective_advisory` on the
  11 draw54 ranks. `annotators.json` 1.2: `yueren` ("Yueren") and `annotator_b` ("Annotator B", the teammate's
  placeholder until she renames it). New `core/annotators.py` (load, resolve, add, rename) and
  `schema/annotators.schema.json`.
- **The rated image** is **acharya_mrudula_075** (draw54 rank 48, assigned to annotator_b). Its val_set file carries
  an XMP / EXIF rating (Rating 4, RatingPercent 75, padding), mtime 2026-11-02, file SHA-256 `4b0e82b5…`, which is
  also the hash in `illumination_edge/heldout_frozen/sample.json`: it was rated before that file was written. The
  unrated copies in `geometry/baseline/data/batch2/` and `geometry/dedicated/data/batch2/` (2025-07-08, file hash
  `116078e8…`, about 1 MB smaller) decode to the **same pixels**: pixel SHA-256 `4c65e623883f34b7…` (1023 × 768 × 3,
  uint16) for both, equal to the stored `image_pixel_sha256`. Only metadata differs; prepare and doctor accept it with
  a warning.
- **Tool:** rename in the name-tag menu (usage count, confirmation, refusals); names resolved everywhere; "mine only"
  image list (default on); warning with open / blind-relabel / stay when opening another annotator's image; targeted s
  disabled without overlays (progress shows "maint. only"). Selftest additions: colliding rename refused, rename
  leaves every annotation file byte-identical and every view shows the new name, assigned-image warning, blind relabel
  exempt.
- **Exports:** `per_pair.csv`, `counts.csv` carry `annotator` (current name) and `annotator_id`; `reflective_advisory`
  stratum in the summary.
- **Overlays / maps maintainer-only:** doctor READY without maps (notes only); `tools/build_maps.sh` checks
  nvidia-smi, the geometry env, DA3, log_chromaticity and loss/ before any work (exit 2 with the reason; checked with
  nvidia-smi hidden from PATH). The pipelines themselves live outside metrics/ and were not modified: the wrapper is
  the early-failure guard.
- **CPU-only:** no torch / DA3 import anywhere in metrics/; `uv.lock` has neither. `uv pip install --only-binary
  :all:` dry-runs of the locked set resolve for `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu` (11 packages,
  wheels only).
- **Assignments:** `docs/ASSIGNMENTS.md` (Annotator B: 54 images, val 11, test 11, train 32; Yueren: 46) and
  `docs/img/draw54_contact_sheet.jpg`, generated by `tools/assignments.py`. `tools/init_images.py` now writes 1.2
  files (`--assign ID` required).
- **check_pr:** `tools/check_pr.py` + `tests/test_check_pr.py` (a throwaway repo: a good branch passes; a bad branch
  trips every rule). `annotations/.backups/` is git-ignored (`metrics/.gitignore`).
- **Docs:** SETUP.md rewritten (Part A annotators: no archive or map steps, branch workflow, image-safety rule,
  platforms; Part B maintainers). ANNOTATION_GUIDE.md: assignments, rename, mine-only list and warning, targeted s
  maintainer-only (removed from the workflow and the targets), troubleshooting. Guide screenshots retaken (30).
- **Tests:** pytest 104 passed; selftest 46/46.
- **Fresh clone, annotator path, CPU only** (`git clone --branch metrics-v1` at e91d8e3, branch
  `annotations/test-annotator`, `config.local.yaml` with only `dataset.val_set`, no log_chromaticity/, no maps, no
  overlays, `CUDA_VISIBLE_DEVICES=` empty): `uv sync --locked` OK (torch not in the environment); doctor before
  prepare: only the cache missing; prepare 100 images in 1 m 50 s; doctor **READY** (7 notes: placeholder, 4 map
  sets, parity CSV); pytest 101 passed / 3 skipped (overlay export from local maps; parity CSV ×2). check_pr on a
  test commit (one new annotator_b pair on adke_neel_026) against the clone's metrics-v1: OK, exit 0, 54-image
  progress table.
- **Commits:** d901b4a (code), e91d8e3 (annotations: the 100 files at 1.2 + annotators.json; verified that only
  id fields and the three new per-image fields changed, 0 content differences in pairs, tombstones, seed logs and
  exhausted marks).

## Publishing to GitHub (owner, after Checkpoint 6, 2026-11-18)

- Owner: push only what the pair-annotation tool needs, on its own branch **`metric-pair-annotation`**; minimal scope
  (`metrics/` + `illumination_edge/common.py`). metrics-v1 is not pushed: it sits on study1, 78 commits ahead of
  origin/main (Study 1 and unpushed yx_dev work), and none of illumination_edge/ is on GitHub.
- Minimal set tested alone (scratch repo with only those files): uv sync, doctor READY, pytest 99 passed / 3 skipped
  / 2 failed (the two migrate-verify tests, which need the old pair files): those now skip with a message.
- `tools/publish.py` (pull / build; plumbing on a temporary index, never pushes) + `tests/test_publish.py` (bare
  remote, maintainer repo, annotator clone: snapshot excludes local history; refuses to overwrite a merged annotator
  pair until pulled; refuses a code change made on GitHub). check_pr's default base is now
  `origin/metric-pair-annotation`; SETUP.md and README updated.

## Teammate report: greyed s button (2026-11-19)

- Cause: by design since Checkpoint 6, the s type button (targeted s, maintainer-only) is off on images without
  overlay maps, i.e. every image on an annotator's machine (and the 54 draw54 images here, which have no maps yet).
  Random s pairs come from the **random s seed** button / r. The explanation was only a tooltip on a *disabled*
  button: Chrome shows neither the tooltip nor any reaction to a click, so it never reached her.
- Fix: the button is greyed but clickable (class `off`, aria-disabled); a click or the s key keeps the current type,
  shows the reason in the banner and highlights the seed button for 3 s. Guide: tour row 3 and troubleshooting.
  Selftest step 12b on adke_neel_026 (no overlays): real mouse click on s, then the seed button proposes a seed with
  no overlays (49/49).
- Found on the way (the owner's 349 new pairs on 12 images, 2026-11-19, uncommitted): `migrate_v0.py verify` assumed
  every pair on a migrated image was migrated (KeyError on tool-made pairs). Now it checks only pairs with
  `legacy.v0`, including ones deleted later (tombstones), and skips the field check for pairs edited in the tool
  (`last_edited_by`). `test_cli_prediction_path` no longer hardcodes 39 pairs.
- **Open question (Q11): float32 tie on the widened H rule.** Textured s pairs pass when `H_R <= max(tau, H_I)`;
  all_R has `H_R == H_I` exactly and passes on the tie. The same prediction stored as float32 (the required
  format) gives `H_R = H_I + 1e-7` and fails: pair t=1795055865898 on taiwei_cui_070 (H_I 0.1126111, tolH_wide =
  H_I). `test_cli_prediction_path` fails on it (deliberately left failing). Lean: compare with a numerical slack
  of 1e-6 on input-referenced tolerances (`H_R <= max(tau, H_I) + 1e-6`), far below any meaningful excursion; the
  alternative is to keep the rule and document that ties are precision-dependent.

## Deviations from Part A (implementation; approved at Checkpoint 1)

1. **Patch size per end inside A/B:** `A = {x, y, r}`, `B = {x, y, r}` instead of one pair-level `r` (old pairs have
   per-end half sizes; C3 keeps them). r = square half-size (or disc radius under `patch.shape: disc`).
2. **Per-image file fields** beyond §3's pair fields: `set`, `seen_in_training` (unknown/yes/no), `width`, `height`,
   `image_sha256`, `s_seed_log` (Phase 4 random-pool log), `legacy` (source set/file/hash, deleted v0 records).
   `set` is per image because no image or pair id occurs in two sets.
3. **Optional pair fields:** `last_edited_by`, `edited`, `history` (Phase 4 audit), `legacy` ({v0 record verbatim,
   v0_index, lit_end inference record, relabel}).
4. `schema_version` = "1.0". `created` for migrated pairs = ISO time of t (the old tool's Date.now() id).
5. Held-out images are read from `image_data/val_set` (byte-identical to `illumination_edge/outputs/heldout/images`,
   checked on two; the per-image SHA-256 is stored). Glossy sets read from the tracked `glossy_*/frozen/` copies
   (same SHA-256 as the `annotation/` copies).
6. `predictions.log_L_tol` = 1e-5 for the optional float32 `log_L` key (two float32 roundings at |log| ≈ 11 already
   reach ~1e-6); our own ℓL is float64 and meets 1e-6 exactly.

## D1 alternative test (pre-declared, 2026-11-18, before running)

Owner rule: adopt H_ref = max_t ||p_I(t) - iso_I(t)|| (R-flat profile pass H_R <= max(tau, H_ref) at the default
sigma_t) only if it beats sigma_t,tex = 3 px on strong-texture GT false fails with halo detection >= 90%; otherwise
keep sigma_t,tex = 3 px. My reading, fixed before the run:
- iso_I: per channel, pool-adjacent-violators isotonic fit of p_I over its finite samples, increasing or decreasing,
  whichever has the smaller squared residual.
- "strong-texture false fails": widened GT false-fail rate at grey SD 0.20, pooled over r = 1, 2, 3 (750 pairs x 30
  seeds); H_ref must be strictly lower than sigma_t 3.
- "halo detection >= 90%": in every (texture level, r) cell, under H_ref.
- Both options at k_tex = 2.5 (the approved fallback); same seeds, pairs and scene (tools/texture_mc.py).

## Approved decisions (Checkpoint 2, 2026-11-18)

Full text in DESIGN.md "Deltas from Checkpoint 2". Short form: linear-space synthetic penumbra approved. D1 k_tex 2.5
and sigma_t 3 px for textured pairs; H_ref (isotonic) alternative tested under the pre-declared rule and NOT adopted
(strong texture 57.5% vs 16.7% GT false fails; moderate 4.1% vs 1.5%; halo detection 99.8% vs 95.8%); textured pairs
stay in the headline; textured stratum reported separately with the synthetic GT false-fail floor
(`reference/texture_mc_d1.json`: sigma3 pooled r 1–3: moderate 1.5%, strong 16.7%; smooth pairs 0%). D2
structured_path renamed input_excursion; stratum only; m error with/without; cross-tab input_excursion × textured ×
geom_cross per type (headline sets). D3 geom_cross w < 0.5 kept. D4 per §4: every same-material d pair in
Long-range; only same-illumination different-material d diagnostic (§5/§8 wording fixed in the deltas). D5–D7 kept.
New: low_contrast (i, m: ||ΔℓI|| <= tau at each tau) excluded from Split and counted.

## Approved decisions (Checkpoint 3, 2026-11-18)

Full text in DESIGN.md "Deltas from Checkpoint 3". E1 bin_eps 1e-4 kept (config). E2 glossy dense diagnostics stay
n/a; the exporter is not run. E3 D3 settings kept. summary.md additions (done before Phase 4): tau sweep on per-tau and
fixed (tau 0.10) populations; approximate perfect-decomposition Split floor next to every Split number. Shareable
summary page: not now.
- Floor implementation: texture_mc.py now also tallies GT false fails per kind x textured status and includes m
  pairs with a textured side; reference/texture_mc_d1.json regenerated (same i numbers). sigma3: i textured moderate
  1.3%, strong 16.7%, i not-textured on textured material 1.8%, m 0% everywhere. Strong boundary se_tex 0.065
  (config texture.strong_se_tex). Headline floor ≈ 0.006–0.013 (i strata 678 / 190 / 57).

## Queue (after Phase 5)

- **Detector-based Retinex baseline** (owner, Checkpoint 3): log-I gradients on default-detector illumination-edge
  pixels go to L, all other gradients to R; reintegrate by Poisson solve; run it through the metrics like the other
  baselines. Purpose: the first mid-range reference, testing whether the metric resolves differences below Split 0.5.

## Housekeeping

- Phase 0 recon ran one version check in the log_chromaticity env without PYTHONDONTWRITEBYTECODE; it wrote 9 scipy
  `.pyc` caches inside `log_chromaticity/.venv/` (git-ignored). Nothing else outside metrics/ was written.

## Phase 0 recon notes (reference)

- Old tool: `illumination_edge/annotate_pairs.py` (frozen, SCORER_v4) / `annotate_pairs_v5.py`; static HTML,
  localStorage autosave, `s` downloads JSON v2; coords row/col 0-based full res; square patches (2h+1)², default h 4;
  display = 99.5th pct → 1, gamma 2.2, `e` ×4, `x` bad pixels, `c` candidate overlay; old `u` = undo.
- Maps (full res, stored frame): geom_v1_1 `geometry/dedicated/baseline_output/stage_b/geom_v1_1/<stem>.npz` key `w`
  (edge = w < 0.5, a band); default illumination detector h29_h35sb: `loss/outputs/dev/weights/default/<stem>.npz`
  (`illum`, `w_illum`, `w_union`); its material edges: `illumination_edge/outputs/failure_review/dev_default/det/h29_h35sb/<stem>.npz`
  key `material` (dev only so far); smoothness weight w_union = clip(min(geom w, w_illum), 0.05, 1): 1 away from edges.
  Held-out geom/ISD/det under `illumination_edge/outputs/heldout/`; glossy maps to be located in Phase 2/3.
- No training code, loader, network, predictions or train/eval lists anywhere (all branches).
