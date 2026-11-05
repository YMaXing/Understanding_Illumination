# geometry/dedicated: geometric-edge weights for the smoothness loss

This project computes a per-pixel weight **w** for the smoothness term of a shading / illumination loss. It is
self-contained: code, configs, tests, docs and outputs all live here. `geometry/baseline/` is frozen.

- **w = 1** where the surface is smooth.
- **w → 0.05** at geometric edges: depth steps and silhouettes, folds between faces, and creases.

The current default pipeline is **geom_v1_1** (`configs/default.yaml` = `configs/candidate_geom_v1_1.yaml`). It
combines three channels by the elementwise minimum:

| channel | fires on |
|---|---|
| depth | occluding depth steps in DA3's depth |
| fold | surface-normal discontinuities, with the band centred on the oriented-Hessian peak |
| laplacian | the `tri_veto4` detector on the filtered disparity |

## Where to read

Start with **`docs/geometric_edges.md`**, the consumer-facing description of the default weight.

- **§1 "What w is"** is the high-level overview. It covers:
  - the three channels and what each fires on;
  - how each channel's weight is formed;
  - the default settings;
  - a data-flow diagram from the Stage A cache, through the elementwise min, to `w`.
- §2–§6 cover loading and using `w` (`pair_weights`), the strengths, alignment, limitations and how to re-run.
- §7.1 gives geom_v1_1's numbers against the previous default, geom_v1.

For more depth:

| doc | covers |
|---|---|
| `docs/geometry_channels.md` | the architecture: module map, data flow, interfaces, Stage B integration; §7 holds the decisions log and the default history |
| `docs/depth_channel.md` | the depth-step channel |
| `docs/fold_channel.md` | the fold (normal-discontinuity) channel |
| `docs/pipeline_tri_veto4.md` | the Laplacian detector (`tri_veto4`, the default before geom_v1) |
| `docs/viz_guide.md` | how to read every visualization panel |
| `docs/api_notes.md` | the DA3 and library APIs this code relies on |

## Visualizations of the default pipeline

Each batch has four sheet folders with one PNG per image:

- batch1: `baseline_output/stage_b/geom_v1_1/`
- batch2: `baseline_output_batch2/stage_b/geom_v1_1/`

| folder | shows | panel guide in `docs/viz_guide.md` |
|---|---|---|
| `viz_geom/` | **the combined sheet**: all three channels, the final min and where the channels disagree; start here | "The combined geometry sheet" |
| `viz_fold/` | the fold channel: normals, candidates, gate, veto, `w_fold` | "The fold sheet and the fold profile tool" |
| `viz_depth/` | the depth channel: accepted edges, ΔZ, significance, `w_depth` | "The depth sheet and the depth profile tool" |
| `viz/` | the Laplacian detector and the final `w` on the photo | "Reading the Stage B visualization panels" |

## Running inference

Run every command from this directory (`geometry/dedicated/`). Prefix each one with `env -u PYTHONPATH`, so that other
Python installs (ROS) stay off the path, and run it through `uv run`. Without `--config`, every command uses
`configs/default.yaml`, which is geom_v1_1 with run name `geom_v1_1`.

### Inputs

- **Images.** Linear RGB TIFFs (`.tif` / `.tiff`). Mosaic (raw Bayer) input is not supported. The EXIF orientation
  tag is respected.
- **Where they are read from.** The input folder is searched **recursively** (`paths.glob: "**/*.tif*"`). The default
  input is `data/`, so a bare `infer` also picks up `data/batch2/`: it sees 20 TIFFs, not 10. Name the folder with `-i`.
- **The output folder** must not be inside the input folder.

### One command: `infer` (Stage A, then Stage B)

```bash
# a new folder of TIFFs
env -u PYTHONPATH uv run python -m illum_baseline infer -i /path/to/tiffs -o /path/to/outputs
# the two batches here, each into its own output folder (how the reference runs were made)
env -u PYTHONPATH uv run python -m illum_baseline infer -i data/batch2 -o baseline_output_batch2
# dataset scale: skip the diagnostic sheets; only some images
env -u PYTHONPATH uv run python -m illum_baseline infer -i /path/to/tiffs -o /path/to/outputs --no-viz
env -u PYTHONPATH uv run python -m illum_baseline infer -i /path/to/tiffs -o /path/to/outputs --stems img_001 img_002
```

To run batch1 alone, don't point `-i` at `data/`: that also picks up `data/batch2/`. Give `--stems` with its ten
`taiwei_cui_*` names, or put the batch1 TIFFs in their own folder.

`infer` runs two stages and prints a JSON summary:
1. **Stage A** (`device: cuda`; falls back to the CPU, with a warning, if no GPU is found):
   - converts each TIFF to sRGB;
   - estimates the focal length with GeoCalib, per image;
   - runs DA3METRIC-LARGE once, at network resolution (long side 504: 504 × 378 or 378 × 504), about 0.1 s per image
     on the GPU.

   It caches the results as `<output>/stage_a/<stem>.npz` (D, sky_prob), `<stem>.json` and `srgb/<stem>.png`.
2. **Stage B** (CPU): the depth, fold and Laplacian channels, their minimum, and the deliverable. About 7.7–7.9 s per
   image, of which about 4.5 s is the dense fit. The four diagnostic sheets add about 24 s per image unless `--no-viz`.

Both stages skip images whose outputs already exist. `--force` recomputes them.

### Outputs: `<output>/stage_b/geom_v1_1/`

| file | contents |
|---|---|
| `<stem>.npz` | **`w`: the deliverable**, float16, at the TIFF's original size and in its **stored** frame (EXIF orientation not applied), 0.05 ≤ w ≤ 1. Also `w_net` (network resolution) and the per-channel weights, strengths and diagnostics (`docs/geometric_edges.md` §1) |
| `<stem>.json` | per-image scalars: channel counts, noise scales, frames (`orientation_applied`, `upright_WxH`, `stored_WxH`) |
| `<stem>_depth_candidates.parquet`, `<stem>_fold_candidates.parquet` (and `_null` tables) | candidate tables, upright network pixel coordinates |
| `manifest.csv` / `manifest.parquet` | one row per image |
| `config_resolved.yaml` | the exact settings of the run |
| `viz/`, `viz_depth/`, `viz_fold/`, `viz_geom/` | the sheets (unless `--no-viz`) |

Using `w` in a smoothness loss (`docs/geometric_edges.md` §1–§2):

```python
import numpy as np
from illum_baseline.outputs import pair_weights

w = np.load("<output>/stage_b/geom_v1_1/<stem>.npz")["w"].astype(np.float32)  # (H, W), stored frame
w_x, w_y = pair_weights(w)   # (H, W-1), (H-1, W): the min of each neighbour pair
# loss term: sum(w_x * |dI_x|^p) + sum(w_y * |dI_y|^p), on the image in the same (stored) frame
```

Multiply the neighbour differences by the weight, never divide by it. Don't re-filter or differentiate `w`.

### Running the stages separately

```bash
env -u PYTHONPATH uv run python -m illum_baseline stage-a -i /path/to/tiffs -o /path/to/outputs   # GPU, cached
env -u PYTHONPATH uv run python -m illum_baseline stage-b -i /path/to/tiffs -o /path/to/outputs   # reads the cache
```

- **What `stage-b` reads.** It processes the images in `<output>/stage_a/`. `-i` is used only to warn about TIFFs that
  have no cache yet.
- **Changing Stage B settings.** Stage A doesn't depend on any Stage B setting, so a change to a detector or
  geometry setting reuses the cache. Pass it with `--config` or `--set` (e.g. `--set geometry.fold.window_cap=end_only`),
  and give the run a new name with `--run-name`. A run folder refuses different detector / sky / preprocess / geometry
  settings unless `--force`, which overwrites it.

### Re-rendering sheets, and the previous defaults

```bash
# sheets from saved outputs only (no recomputation); stems optional, default all
env -u PYTHONPATH uv run python -m illum_baseline viz -o /path/to/outputs [stems ...]
# a previous default, as its own run folder
env -u PYTHONPATH uv run python -m illum_baseline stage-b --config configs/candidate_geom_v1.yaml -i /path/to/tiffs -o /path/to/outputs
```

The previous defaults are `configs/candidate_geom_v1.yaml` (run `geom_v1`) and `configs/candidate_tri_veto4.yaml`
(run `tri_veto4`, the Laplacian detector only).

### Tests

```bash
env -u PYTHONPATH uv run pytest   # about 27 min; includes the bit-for-bit check of every reference run
```
