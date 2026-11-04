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

## Running

Run from this directory. Unset `PYTHONPATH` so that other Python installs, such as ROS, stay off the path.

```bash
# full default pipeline (Stage A: DA3 on the GPU, cached; Stage B: the geometry channels on the CPU)
env -u PYTHONPATH uv run python -m illum_baseline infer
# Stage B only, on another batch
env -u PYTHONPATH uv run python -m illum_baseline stage-b -i data/batch2 -o baseline_output_batch2
# re-render the sheets from saved outputs, for chosen images
env -u PYTHONPATH uv run python -m illum_baseline viz --config configs/candidate_geom_v1_1.yaml [stems ...]
# tests, including the bit-for-bit check of every reference run
env -u PYTHONPATH uv run pytest
```

Pass `--no-viz` to `infer` or `stage-b` for dataset-scale runs. Rendering the four sheets adds about 24 s per image to
Stage B's 7–8 s (CPU).

Each run writes to `stage_b/<run.name>/`. That directory refuses different settings unless `--force`, so a changed
config needs a new `run.name`. The earlier defaults stay reproducible with `--config`:

- `configs/candidate_geom_v1.yaml`
- `configs/candidate_tri_veto4.yaml`
