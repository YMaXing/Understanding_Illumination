# metrics/: evaluation metrics v1 for Understanding Illumination

Sparse annotated-pair metrics (headline) and dense diagnostics for a network that predicts log reflectance ℓR from
log intensity ℓI (ℓL = ℓI − ℓR). The design is `DESIGN.md` (Part A of `SPEC_v1.md`, verbatim, followed by the
owner's rulings, "Deltas from Checkpoint 0…6", which take precedence). Status and decisions: `PROGRESS.md`.
Annotators: `docs/ANNOTATION_GUIDE.md`, their images: `docs/ASSIGNMENTS.md`. A new machine (annotator and maintainer
parts): `docs/SETUP.md`.

## Setup

```
cd metrics && env -u PYTHONPATH UV_LINK_MODE=copy uv sync      # Python 3.12; versions pinned in pyproject.toml + uv.lock
cp config.local.example.yaml config.local.yaml                  # machine-specific: dataset.val_set, maps, parity CSV, tools
cd .. && metrics/run.sh doctor.py                               # what is missing on this machine, and how to fix it
metrics/run.sh -m pytest -q -p no:cacheprovider                 # 105 tests; those needing local files skip with a message
metrics/run.sh annotate/selftest.py                             # annotation tool in headless Chrome with real input events
```

Tracked files carry no machine paths. `metrics/config.local.yaml` (untracked) is merged over `config.yaml`; see
`config.local.example.yaml`. Caches, maps and runs never go into git.

`metrics/run.sh` drops PYTHONPATH (ROS leaks into it) and writes no bytecode. ℓI is never computed here: it comes
from the loader hook in `config.yaml` (`loader.*`, today `illumination_edge/common.py`).

## Running the metrics

```
metrics/run.sh run_metrics.py --out runs/NAME [--pred DIR [--name model]] [--images LIST]
```

- `--pred DIR`: `DIR/<stem>.npz` with `log_R` float32 (H, W, 3), natural log, full resolution, stored TIFF frame;
  optional `log_L` (the identity ℓL = ℓI − ℓR is checked). Without `--pred` only the baselines run.
- `--images`: a file of stems, a comma list, or image-set names (`dev_v2`, `heldout`, `glossy_dev`, `glossy_confirm`,
  `draw54`). Default: all 100 images.
- `--test`: also score and report the **test** split. Without it, test images are not even loaded.
- The three degenerate baselines (`all_R`: ℓR = ℓI; `all_L`: ℓR = 0; `blur_L`: ℓR = ℓI − G₈ ∗ ℓI) always run alongside.
- Runtime: about 80 s for 46 images and the three baselines.

## Reading the outputs (`metrics/runs/NAME/`, git-ignored)

| file | content |
|---|---|
| `summary.md` | one block per **data split** (train, val; test only with `--test`), never pooled, plus a "supervised fit" block per split for pairs with `used_in_loss` = true (never pooled with unused pairs), and the **train − val gap** per headline number. Per block: **Split error** = mean(ER_i, ER_m), **Leak** = ER of random s, **Long-range** = ER of same-material d (a rate over 0 pairs reads "n/a: 0 pairs"), each with a 95% image-bootstrap CI and the approximate perfect-decomposition floor; strata by provenance group (dev, glossy_dev, glossy_confirm, draw54), annotator group (A / B), glossy pool and reflective_advisory; plain-τ values in brackets; the τ sweep on per-τ and fixed (τ = 0.10) populations; exclusions; split bins; companions; b diagnostics; strata; the input_excursion × textured × geom_cross table; dense diagnostics D1–D3 (D4, A1, A2: n/a, no region source); the textured stratum and its synthetic error floor; migrated i pairs whose inferred lit end is B |
| `summary.json` | all of the above, plus the config, git state (HEAD, metrics-v1, whether metrics/ matches it), map provenance and SHA-256s, annotation and prediction SHA-256s |
| `per_pair.csv` | one row per (method, pair): data split, used_in_loss, provenance group and set, annotator group, glossy pool, reflective_advisory, `annotator` (the tag's current name) and `annotator_id` (the stable id stored in the files); Δ vectors and endpoint values, norms, α, H_I / H_R / H_L, E, H, tolerances, pass (plain / widened), bins, every flag, exclusions, pass / excluded at each sweep τ |
| `per_image.csv` | per (method, image): eligible counts and error rates per kind, dense diagnostics |
| `profiles.npz` | p_I, p_R, p_L per (method, pair): arrays `t`, `I`, `R`, `L` + `index` (method, image, id, start, stop) |

Error rate = failing / eligible pairs, micro over pairs. A pair passes when its endpoint error E and its profile
excursion H are within tolerance: plain τ = 0.10; widened for R-flat kinds (i, s, same-material d): E ≤ max(τ, 2.5 ·
se_tex), H_R ≤ max(τ, H_I), with σ_t = 3 px on textured pairs. Excluded from headlines (and counted): low SNR
(sensor-noise model), invalid patches, unsure, low_contrast (i, m with ‖ΔℓI‖ ≤ τ), b pairs, targeted s,
same-illumination different-material d, unscored d labels.

To look at a run pair by pair: `metrics/annotate/run.sh --review`, then open `http://localhost:8765/review.html`
(read-only; the annotation page never sees run data).

## Editing the config (`config.yaml`)

Every threshold, path and switch is in `config.yaml`, with defaults from DESIGN.md and comments. The ones most
likely to change:

| key | meaning |
|---|---|
| `loader.read / log_i / mask` | the ℓI hook (switch to the training loader once it exists; then re-run the parity test at 1e-6) |
| `validity.sat_dn`, `eps_dn` | pixel validity (65000 / 0 DN) |
| `patch.shape` | `square` (default) or `disc` |
| `thresholds.tau`, `tau_sweep`, `snr_divisor`, `alpha_min_factor`, `bin_eps` | §2 / §4 thresholds |
| `texture.k_tex`, `textured_se_tex`, `strong_se_tex`, `profile_ref`; `profile.sigma_t_textured_px` | the texture tolerances (Delta Q1 / D1) |
| `dataset.val_set` | where the images are (set it in `config.local.yaml`) |
| `image_sets` | the 100 images per provenance set; each image's data split is its annotation file's `split` (record: `splits/split100.json`) |
| `aggregate.report_splits`, `provenance_groups`, `annotator_groups` | report blocks (train, val) and strata |
| `maps.*` | geometric edges, smoothness weight, illumination and material edges per set (machine-specific: `config.local.yaml`) |
| `annotate.*` | tool settings, per-image targets (i 33, m 33, b 3 = dev medians; random s and d 12–15), annotation order, relabel folder |

After changing a config value, re-run the tests (`metrics/run.sh -m pytest -q -p no:cacheprovider`).

## Other tools

| command | what it does |
|---|---|
| `metrics/run.sh migrate_v0.py run / verify / counts` | lossless migration of the old pair files (done; verify re-checks it) |
| `metrics/run.sh doctor.py` | checks env, dataset, image identity (file hash, else pixel hash with a warning), cache; overlays and maps as notes; READY needs no maps; prints fixes |
| `metrics/run.sh tools/draw_images.py` | seeded draw of new images from val_set (excludes annotated, glossy pool, FRESH_PROTOCOL span) |
| `metrics/run.sh tools/assign_splits.py` | the stratified train / val / test assignment (`splits/split100.json`) |
| `metrics/run.sh tools/init_images.py --assign ID --images LIST` | empty annotation files for new images (split, assigned_to, pixel hash); then `annotate/prepare.py --images LIST` |
| `metrics/run.sh tools/upgrade_schema_1_1.py`, `upgrade_schema_1_2.py` | annotation files 1.0 -> 1.1 -> 1.2 (done; idempotent). 1.2: annotator ids, `assigned_to`, `image_pixel_sha256`, `reflective_advisory` |
| `metrics/run.sh tools/assignments.py` | regenerate `docs/ASSIGNMENTS.md` and the draw54 contact sheet copy |
| `metrics/run.sh tools/check_pr.py BRANCH` | read-only pre-merge check of an annotator's branch against `origin/metric-pair-annotation`; per-image progress; exit 1 on any error |
| `metrics/run.sh tools/publish.py pull / build` | publish the tool to GitHub as `metric-pair-annotation` (origin/main + a snapshot of metrics/ and `illumination_edge/common.py`); pull merged annotations back first; never pushes |
| `metrics/tools/build_maps.sh [OUT]` | maintainer, GPU: detector maps for the 100 images (fails early without a GPU or an environment) |
| `metrics/run.sh tools/overlays.py export / import` | maintainer: overlay-map archive with a hash manifest |
| `metrics/run.sh tools/relabel_agreement.py` | blind-relabel agreement: per-type agreement, Cohen's κ for type and for the d labels |
| `metrics/run.sh tools/texture_mc.py` | texture-tolerance Monte Carlo on the synthetic scene (the error-floor reference) |
| `python3 metrics/annotate/merge.py --from DIR [--apply]` | merge another annotator's folder (dry run by default) |
| `metrics/tools/commit_metrics.sh "msg"` | commit metrics/ to branch metrics-v1 without touching the shared working tree |
