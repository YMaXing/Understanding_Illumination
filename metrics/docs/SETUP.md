# Setting up the annotation tool on a new machine

Part A is for an **annotator** on their own laptop: clone, install, point the tool at your images, annotate your
assigned images, send them back. No GPU, no detector maps, no archives. About 15 minutes.

Part B is for **maintainers** (overlay maps, targeted s pairs, merging annotators' branches).

Your images are listed in [ASSIGNMENTS.md](ASSIGNMENTS.md). How to annotate: [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md).

---

# Part A: annotators

## Prerequisites

- **uv** (the Python package manager): <https://docs.astral.sh/uv/getting-started/installation/>. uv installs the
  right Python (3.12) by itself.
- **git**.
- **Supported systems:** macOS on Apple silicon (arm64), Linux x86_64, and Windows through **WSL2** (Ubuntu; run every
  command below inside WSL and open the tool in your normal Windows browser). `metrics/uv.lock` resolves to ready-made
  packages (wheels) for macOS arm64 and Linux x86_64 with Python 3.12 (checked 2026-11-18); nothing needs a compiler.
- **No GPU needed.** Nothing in this part imports torch or Depth Anything 3 (they are not even in the environment).
- **The image set**: the project's `val_set` folder (503 `.tif` files). It can be anywhere on your disk.
- **Disk space**: about 1 GB for the tool's cache.

> **Never rate, tag or edit the dataset images in Windows Explorer, Photos or any other app.** Rating a photo
> rewrites the file. The tool identifies images by their pixels, so a rated file is still accepted (with a warning),
> but an edit that changes pixels makes it a different image and the tool refuses it.

## Steps

1. **Clone** the tool's branch, `metric-pair-annotation`, and make your own branch from it (use your name):

   ```
   git clone --branch metric-pair-annotation <repository URL> codebase
   cd codebase
   git checkout -b annotations/<your-name>
   ```

2. **Install** the environment (Python 3.12 and the pinned packages from `metrics/uv.lock`):

   ```
   cd metrics
   env -u PYTHONPATH uv sync
   cd ..
   ```

   (`env -u PYTHONPATH` keeps other Python installations out of the way. It is harmless otherwise.)

3. **Tell the tool where your images are.** Copy the example local config and edit one line:

   ```
   cp metrics/config.local.example.yaml metrics/config.local.yaml
   ```

   In `metrics/config.local.yaml`, set:

   ```
   dataset:
     val_set: /path/to/your/val_set
   ```

   On WSL, a Windows folder `D:\data\val_set` is `/mnt/d/data/val_set`. On macOS, for example
   `/Users/you/data/val_set`. This file is yours: git ignores it.

4. **Build the cache** (the display data for each image; about 2 minutes for 100 images):

   ```
   metrics/run.sh annotate/prepare.py
   ```

5. **Check the setup:**

   ```
   metrics/run.sh doctor.py
   ```

   It must print **READY**. It also lists notes that are fine to ignore: no overlay maps, no detector maps, no parity
   file, no browser-test tools. Every problem has a `fix:` line. If it reports an image whose **pixels** differ,
   your copy is a different image: get the original file from the project lead.

6. **Start the tool:**

   ```
   metrics/annotate/run.sh
   ```

   Open `http://localhost:8765/`. Choose your name tag (**Annotator B** was made for you; rename it to your name with
   **Rename this tag…** in the name menu). Then follow [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) from section 1,
   step 9.

To stop the tool, press **Ctrl+C** in the terminal. Next time, only step 6 is needed (after the session routine
below).

## Your session routine

- **Annotate only your assigned images** ([ASSIGNMENTS.md](ASSIGNMENTS.md); the tool lists only them by default),
  **val and test first, then train**.
- **Before each session**, bring in the project's latest changes:

  ```
  git fetch origin
  git merge origin/metric-pair-annotation
  ```

- **At the end of each session**, commit only your annotation files and push:

  ```
  git add metrics/annotations/
  git commit -m "annotations: <images>"
  git push -u origin annotations/<your-name>
  ```

  Never commit anything outside `metrics/annotations/` (caches, `config.local.yaml` and backups are ignored anyway).
- **When a batch is finished** (for example all your val images), open a pull request from your branch into
  `metric-pair-annotation`. The project lead checks it with `tools/check_pr.py` and merges it.

Each image has its own annotation file and each of us annotates different images, so changes almost never touch the
same file. Only `metrics/annotations/annotators.json` (the name tags) can conflict: keep both sides' entries.

---

# Part B: maintainers

## Checking an annotator's branch before merging

```
git fetch origin
metrics/run.sh tools/check_pr.py origin/annotations/<name>           # read-only; exit code 1 on any error
```

It compares the branch with its merge base on `origin/metric-pair-annotation` (`--base` to change it) and reports
errors for:

- **files:** anything other than annotation files of images assigned to the branch's annotator, `annotators.json`
  and relabel files;
- **image files:** a schema failure, or a changed image identity (pixel hash, split, assignment);
- **authorship:** new pairs not by the branch's annotator, edits without their `last_edited_by`, or authorship
  changes on existing pairs;
- **deletions:** a pair removed without a tombstone, or a tombstoned pair resurrected;
- **logs:** a non-append-only seed log or relabel file;
- **name tags:** in `annotators.json`, anything but additions and recorded renames.

It then prints per-image progress against the targets (exhausted marks count as met).

After merging a pull request on GitHub, bring the annotations back into the local `metrics-v1` (next section, step 1).

## Publishing the tool to GitHub

The local `metrics-v1` branch sits on top of unpublished study work, so it is never pushed. GitHub gets the branch
**`metric-pair-annotation`**: `origin/main` plus one snapshot commit per publish that contains only `metrics/` and
`illumination_edge/common.py` (the loader and noise hooks the tool needs).

```
metrics/run.sh tools/publish.py pull                    # 1. annotations merged on GitHub: dry run, then --apply
metrics/tools/commit_metrics.sh --annotations "annotations: merged from metric-pair-annotation"   # if 1 wrote files
metrics/run.sh tools/publish.py build                   # 2. snapshot metrics-v1 onto the local metric-pair-annotation
git push origin metric-pair-annotation                  # 3. after reading what build printed
```

`build` refuses, and writes nothing, if GitHub has annotations that `metrics-v1` does not contain yet (step 1 first,
so no one's pairs are overwritten), if anything else under `metrics/` was changed on GitHub since the last snapshot,
or if the local and GitHub branches have diverged. It never pushes.

## Overlay maps (optional; maintainer only)

Overlays and targeted s pairs are not part of the annotator workflow. The lead adds targeted s pairs to an
annotator's images after their batch is merged. To build the maps for all 100 images (needs a CUDA GPU, Depth
Anything 3, the `geometry/dedicated` and `log_chromaticity` environments):

```
metrics/tools/build_maps.sh                  # fails at once, with the reason, if the GPU or an environment is missing
```

It runs `loss/run.sh unified_weight.py run` (geometry, ISD maps, raw step, default-arm weights) and the frozen
`illumination_edge/run.sh glossy_h35sb.py run` (the default arm's material edges) into `loss/outputs/split100`. Then
point `metrics/config.local.yaml` at them for every image set (`dev_v2`, `heldout`, `glossy_dev`, `glossy_confirm`,
`draw54`):

```
maps:
  geom_v1_1: {dirs: {dev_v2: loss/outputs/split100/geometry/stage_b/geom_v1_1, ...}}
  smoothness_weight: {dev_v2: {dir: loss/outputs/split100/weights/default, key: w_union}, ...}
  illum_edges:       {dev_v2: {dir: loss/outputs/split100/weights/default, key: illum}, ...}
  material_edges:    {dev_v2: {dir: loss/outputs/split100/det/h29_h35sb, key: material}, ...}
```

and rebuild the cache: `metrics/run.sh annotate/prepare.py`.

For a second maintainer without a GPU, export the overlays of all 100 images into one small archive with a hash
manifest, and import it on their machine:

```
metrics/run.sh tools/overlays.py export --out overlays_$(date +%F).tar.gz      # on the machine with the maps
metrics/run.sh tools/overlays.py import overlays_<date>.tar.gz                  # on the other machine
metrics/run.sh annotate/prepare.py
```

## Other maintainer commands

| command | purpose |
|---|---|
| `metrics/run.sh tools/assignments.py` | regenerate `docs/ASSIGNMENTS.md` (after a rename or a new assignment) |
| `metrics/run.sh migrate_v0.py verify` | **maintainer only**: re-checks the one-time migration of the old pair files; needs the original source files under `illumination_edge/` |
| `python3 metrics/annotate/merge.py --from DIR [--apply]` | merge an annotations folder that was edited on two machines without git (unions by pair id; conflicts reported) |
| `metrics/run.sh annotate/selftest.py` | browser self-test (Windows Chrome + PowerShell from WSL; `tools.*` in `config.local.yaml`) |

## What the tool needs from outside metrics/

| needed | why | in git? |
|---|---|---|
| `illumination_edge/common.py` | the ℓI loader hook (`load_rgb`, `log_rgb`) and the sensor-noise model (`estimate_noise_robust`) | yes |
| `log_chromaticity/material_edges.py` | imported by `common.py` at import time, unused by the hooks | **no** (git-ignored): an inert placeholder is used; any use raises an error naming the missing file; doctor reports it |
| `illumination_edge/heldout_frozen/sample.json` | the FRESH_PROTOCOL exclusion span for image draws (`tools/draw_images.py`, maintainer) | yes, but not on `metric-pair-annotation` |
| the four migrated source pair files under `illumination_edge/` | only `migrate_v0.py verify` and its two tests (they skip without them) | yes, but not on `metric-pair-annotation` |
| the image set (`val_set`) | images | no: `dataset.val_set` in `config.local.yaml` |
| detector maps (`loss/outputs/`, `illumination_edge/outputs/`, `geometry/.../baseline_output/`) | overlays, `geom_cross`, dense diagnostics | no: maintainer only (`maps.*` in `config.local.yaml`) |
| `illumination_edge/outputs/annotate/frozen_v2_pairs_fullres.csv` | the parity test only (it skips without it) | no |

The display transform (99.5th-percentile scaling, gamma 2.2) is reimplemented in `metrics/annotate/lib.js` for
display only; ℓI itself always comes from the loader hook. `migration_v0/manifest.json` keeps four absolute
`image_dir` paths copied verbatim from the old files (provenance).
