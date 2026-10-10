# Point Annotator

A small OpenCV tool for clicking sparse ground-truth points on images and labeling each group of points as **material**, **texture** or **specular** (Understanding Illumination project). It saves one CSV per image, with coordinates in the original image's resolution.

## Setup

You need [uv](https://docs.astral.sh/uv/) and a desktop display (on WSL2, WSLg works). From this directory:

```bash
uv sync
```

This installs Python 3.12, `opencv-python`, `numpy` and the dev dependency `pytest` into the workspace's `.venv`. Add or change dependencies only with `uv add` / `uv add --dev`; don't use pip.

> **ROS users:** if ROS (e.g. Jazzy) is sourced in your shell, its `PYTHONPATH` gets mixed into Python's import path. Prefix every command with `env -u PYTHONPATH`, e.g. `env -u PYTHONPATH uv run pytest -q`. The examples below leave the prefix out.

## Quick start

Try it on the three bundled sample images:

```bash
uv run annotate.py samples out
```

- `samples`: the folder of images to annotate. Files ending in `.tif`, `.tiff`, `.png`, `.jpg` or `.jpeg` are loaded in name order. Subfolders are ignored.
- `out`: where the CSVs go. It's created if it doesn't exist.
- `--max-size WxH` (optional, default `1600x900`): the largest the main window may be, status strip included. Larger images are shrunk to fit and never enlarged. Use it on small screens, e.g. `--max-size 1200x800`.

On the full dataset:

```bash
uv run annotate.py ../material_clustering_data/original_data out
```

## The window

```
┌───────────────────────────────┐
│                               │
│            image              │  ← click here to place points
│   ●1  ●      ■2               │    (color = region, shape = label,
│                ▲3             │     number = region id)
├───────────────────────────────┤
│ ● material  region 1 (2 pts)  total 4  [1/3] adke_neel_050.tif
│ gamma on  zoom off  cursor (412,630)   added (412,630)
│ m/t/p label  space new region  u undo  z zoom  g gamma  s save  n/b next/prev  q quit
└───────────────────────────────┘
```

- **Markers:** the shape shows the label (● material, ■ texture, ▲ specular) and the color shows the region. Each region's number is drawn next to its first point.
- **Status strip** (below the image; clicks there do nothing):
  1. The current label and region, the region's point count, the image's total point count, and the image's position and filename.
  2. Gamma and zoom on/off, the cursor position in original-image pixels, and the result of the last action (e.g. `pixel (412,630) already used`).
  3. The key legend.

## Controls

Keys are lowercase letters. Keep **Caps Lock off**: `Z` means something else, and the status strip warns you about unmapped capital letters. The image window needs keyboard focus, so click it first if keys seem to do nothing.

| Key | Action |
|---|---|
| Left click | Add a point to the current region |
| `m` / `t` / `p` | Set the label to material / texture / specular |
| `space` | Start a new region (does nothing if the current region is empty) |
| `u` | Undo the most recent point |
| `z` | Open the zoom window at the cursor; press again to re-center it |
| `Z` or `Esc` | Close the zoom window |
| `g` | Turn display gamma on or off |
| `s` | Save |
| `n` / `b` | Next / previous image (saves first) |
| `q` | Save and quit (closing the window also saves) |

### Regions and labels

A **region** is a group of points on one surface that share one label.

1. Choose a label (`m`, `t` or `p`). New images start on `material`.
2. Click a few points on the surface.
3. Press `space` to start the next region, or press a different label key. Changing the label while the current region has points starts a new region automatically, so a region never mixes labels. If the region has no points yet, the label key just changes its label.

**Undo** (`u`) always removes the very last point you placed, even if it belongs to an earlier region. If that empties the current region, you go back to the previous region and its label, so pressing `u` repeatedly walks back through your work.

**Each pixel can be used once per image.** A click on a pixel that already has a point (in any region) is ignored and reported in the status strip.

### Zoom

When the image is shrunk to fit the screen, use zoom to place points precisely, e.g. away from material boundaries.

- `z` opens a second window showing a 128×128-pixel area around the cursor, enlarged 4× with no smoothing, so every image pixel is visible.
- The zoom view stays where you opened it while you move the mouse, so you can click inside it. Press `z` again to move it to the cursor's current position.
- Clicks in the zoom window add points exactly like clicks in the main window. The pixel under the cursor is outlined in white, and existing points are drawn there too.
- Moving to another image closes the zoom window.

### Display

The source TIFFs are 16-bit linear images, so they're adjusted for viewing. Brightness is stretched between the 0.5th and 99.5th percentiles, using the same range for all three channels so colors don't shift, and gamma 1/2.2 is applied. `g` turns the gamma off if dark areas look washed out. **This only affects the display, never the saved coordinates.**

## Output

One CSV per image: `<out_dir>/<image_stem>.csv`, e.g. `out/adke_neel_050.csv`.

```csv
image,x,y,region_id,label
adke_neel_050.tif,412,630,1,material
adke_neel_050.tif,415,641,1,material
adke_neel_050.tif,201,88,2,specular
```

| Column | Meaning |
|---|---|
| `image` | Image filename, including its extension |
| `x`, `y` | Column and row, as integer pixels in the **original** resolution (whatever the display scale) |
| `region_id` | Region number within the image: 1, 2, 3, … |
| `label` | `material`, `texture` or `specular` |

- An image you visited but didn't annotate gets a CSV with just the header line. That's how you tell "checked, nothing to mark" apart from "not visited yet".
- **Reopening an image loads its CSV**, so you can stop and continue later. New points go into a new region after the existing ones.
- Region numbers are renumbered 1, 2, 3, … when saving, so gaps left by undo are never saved.
- Saves first write to a temporary file and then replace the old CSV, so a crash mid-save can't leave a half-written file.

### Checking your output

`check_csv.py` checks every CSV in an output folder: points are inside the image, no pixel is used twice, each region has a single label, and region ids are 1..k.

```bash
uv run check_csv.py samples out
```

## Development

```bash
uv run pytest -q
```

Everything in `annotate.py` that doesn't need a window is a plain function with pytest tests in `tests/test_annotate.py`. That covers CSV reading and writing, click-to-pixel mapping, zoom mapping, display tone mapping and the region/undo logic. The pytest settings are in `pyproject.toml`. See `CLAUDE.md` for the design spec.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ImportError` or odd `cv2`/`numpy` versions | ROS `PYTHONPATH` is leaking in; run with `env -u PYTHONPATH uv run …` |
| `The function is not implemented … GTK/Qt` | You have `opencv-python-headless`; this tool needs `opencv-python` (run `uv sync`) |
| No window appears (WSL/SSH) | Make sure a display is available (WSLg on Windows 11, or X forwarding) |
| Keys do nothing | Click the image window to give it focus; check Caps Lock |
| `CSV name collision: a.tif and a.png` | Two images share a filename stem and would write the same CSV; rename one |
| `…csv:N: unknown label …` on load | That CSV was edited by hand and has a bad row; fix line N (the tool refuses to drop data silently) |
| `cannot read image` | The file is corrupt or not a real image; move it out of the input folder |
