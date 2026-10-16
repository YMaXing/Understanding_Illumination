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
- `--max-size WxH` (optional, default `1600x900`): the largest the main window may be, help panel included. Larger images are shrunk to fit and never enlarged. Use it on small screens, e.g. `--max-size 1200x800`.
- `--panel auto|right|bottom` (optional, default `auto`): where the status/help panel goes. `auto` puts it to the right of portrait images and below landscape ones.
- `--debug-mouse` (optional): prints the GUI backend, the window flags and every mouse click to the terminal. Use it if clicks don't seem to do anything.
- `uv run annotate.py --palette-preview colors.png` writes a picture of the region colors 1–40 and exits.

On the full dataset:

```bash
uv run annotate.py ../material_clustering_data/original_data out
```

## The window

```
portrait image (panel on the right)          landscape image (panel below)
┌──────────────┬─────────────────────────┐   ┌──────────────────────────────────┐
│              │ Status                  │   │                                  │
│    image     │ ● material              │   │              image               │
│  ●1  ●  ■2   │ region 1 (existing, 2)  │   │        ●1  ●      ■2   ▲3        │
│       ▲3     │ regions 3   points 4    │   │                                  │
│              │ image 1/3 (n = next,    │   ├───────────┬───────────┬──────────┤
│              │   b = previous) a.tif   │   │ Status    │ Labels    │ Edit     │
│              │ gamma 2.2 on  zoom off  │   │ …         │ Regions   │ View     │
│              │ mode ADD - ADDING TO    │   │           │           │ Files    │
│              │   existing region 1     │   └───────────┴───────────┴──────────┘
│              │ saved                   │
│              │ Labels / Regions / Edit │
│              │ / View / Files help     │
└──────────────┴─────────────────────────┘
```

- **Markers:** the shape shows the label (● material, ■ texture, ▲ specular) and the color shows the region. Each region's number is drawn next to its first point. The **current region's** dots have a thick white outline, in both the main and zoom windows.
- **Status block:**
  - the current label (with its marker in the region's color)
  - `region 3 (existing, 5 pts)` or `region 4 (new)`
  - the image's total regions and points
  - `image i/N (n = next, b = previous)` and the filename
  - gamma value and on/off, zoom on/off
  - the mode: `mode ADD - ADDING TO existing region N`, or `mode PICK` (orange) while you're choosing a region
- **Message line** (yellow): the result of your last action, e.g. `saved`, `region 9 started`, `selected region 4`, `deleted point`, `last image`, `no dot near click`. It disappears after about 3 seconds.
- **Help:** every key, grouped under Labels, Regions, Edit, View and Files.
- Clicks in the panel do nothing.

## Controls

Keys are lowercase letters (plus `space`, `[`, `]`, `+`, `-` and `Esc`). Keep **Caps Lock off**: the panel warns you about unmapped capital letters. The image window needs keyboard focus, so click it first if keys seem to do nothing. Only `q` quits; `Esc` never does.

| Key | Action |
|---|---|
| Left click | Add a point to the current region (in PICK mode: select the clicked dot's region) |
| `r` | PICK mode: your next left click on (or near) a dot selects its region; `r` again or `Esc` cancels |
| Right click / Ctrl+left click | Select the region of the nearest dot (within 10 px), in the main or zoom window |
| `[` / `]` | Cycle back / forward through regions 1…k and a new one, wrapping around |
| `m` / `t` / `p` | Set the label: **material** (one reflectance, no variation: paint, a single brick, plastic), **texture** (fine-scale variation: grass, asphalt, speckled concrete), **specular** (shiny/metal). Opens a new region if the current one has points |
| `space` | Start a new region (does nothing if the current region is empty) |
| `x` | Delete the dot with the cyan ring (the nearest one within 10 px of the cursor) |
| `u` | Undo the last add or delete |
| `z` | Open the zoom window at the cursor; press again to re-center it |
| `Esc` | Cancel PICK mode, otherwise close the zoom window (works from either window) |
| `g` | Turn display gamma on or off (keeps the value) |
| `+` / `-` | Gamma up / down by 0.2 (`=` and `_` also work); if gamma is off, the first press just turns it on |
| `s` | Save |
| `n` / `b` | Next / previous image (saves first); does nothing on the last / first image |
| `q` | Save and quit (closing the main window also saves and quits) |
| `c` | Create a super-region from the current region (see below) |
| `a` | Add the current region to the selected super-region (moves it if it's in another one) |
| `d` | Drop the current region from its super-region (the region and its points stay) |
| `,` / `.` | Select the previous / next super-region |
| `y` | Super-pick: click a dot to select its region's super-region |
| `v` | Show / hide the super-region rings |

### Regions and labels

A **region** is a group of points on one surface that share one label.

1. Choose a label (`m`, `t` or `p`). New images start on `material`.
2. Click a few points on the surface.
3. Press `space` to start the next region, or press a different label key. Changing the label while the current region has points starts a new region automatically, so a region never mixes labels. If the region has no points yet, the label key just changes its label. Pressing the label you're already on does nothing.

New regions always get the next unused number (the highest existing number + 1), wherever you are.

**Going back to a region.** To add more points to a region you've already left, press **`r`** and click one of its dots (the cyan ring shows which dot you'll get), or step to it with `[` / `]`. Right-click and Ctrl+click are shortcuts for the same thing. That region becomes current, with its own label; the panel says `ADDING TO existing region N`, and your next left clicks go into it. If your click misses every dot, you stay in PICK mode and nothing is added. The cycle runs 1, 2, …, k, then a new empty region, then back to 1. A region's label can't be changed once it has points: pressing a different label key while an existing region is selected opens a new region with that label instead, and the panel says `new region N started`.

**Each pixel can be used once per image.** A click on a pixel that already has a point (in any region) is ignored and reported in the status strip.

### Super-regions

A region is one surface. A **super-region** says "these separate regions are the same material or texture", even when other surfaces lie between them. Example: one region per red brick, then all the bricks in one super-region.

1. Select a region (click its dots with `r`, or use `[` / `]`) and press **`c`**. That creates a new super-region (S1, S2, …) with the region's label and selects it.
2. Select the next region that belongs with it and press **`a`**. Repeat for each one. A region with a different label is refused, and a region already in another super-region is moved ("moved from S3 to S5").
3. **`d`** takes the current region out of its super-region. A super-region that loses its last region disappears.
4. To switch to another super-region, press **`,`** / **`.`** or press **`y`** and click one of its dots. Neither changes your current region.

Each member dot gets a ring in its super-region's color, which is thicker for the selected one. The labels read `6>S2` (region 6 in S2), and the selected super-region's regions are joined by thin lines. **`v`** hides or shows all of this; it never changes the data. `u` undoes `c`, `a` and `d` like any other edit. Deleting a region's last point takes it out of its super-region, and undoing that delete puts it back.

### Deleting points

Move the mouse near a dot: a **cyan ring** appears around the nearest dot within 10 px. That's the dot `x` will delete. Press `x` to delete it, from any region. If no ring is showing, `x` does nothing. When two dots are equally close, the most recently placed one gets the ring. If dots are crowded, open zoom (`z`) and hover there instead; the ring shows in both windows.

Deleting doesn't change which region is current. If you delete the last point of the current region, it stays current (now empty, same label), so your next click refills it. The delete key is `x` rather than Delete because OpenCV reports special keys differently on each platform.

### Undo

`u` undoes your last **action**, either adding or deleting a point, whichever region it was in. Press it repeatedly to walk back through your work. There is no redo.

- **Undoing an add** removes that point. The current region and label become those of the last remaining point (region 1 / material if none are left).
- **Undoing a delete** puts the point back in its original region with its original label, and returns you to the region you were on when you deleted it.

### Zoom

When the image is shrunk to fit the screen, use zoom to place points precisely, e.g. away from material boundaries.

- `z` opens a second window showing a 128×128-pixel area around the cursor, enlarged 4× with no smoothing, so every image pixel is visible.
- The zoom view stays where you opened it while you move the mouse, so you can click inside it. Press `z` again to move it to the cursor's current position.
- Left-clicks in the zoom window add points, PICK / right-click / Ctrl+click select a region, and `x` deletes the ringed dot, exactly like in the main window. The pixel under the cursor is outlined in white, and existing points are drawn there too, with the current region highlighted. Right-click is often easier here, because dots that overlap in the main view are spread apart.
- `Esc` closes it, whichever window has focus. Closing it with its title-bar X is fine too.
- Moving to another image closes the zoom window.

### Display

The source TIFFs are 16-bit linear images, so they're adjusted for viewing:

```
display = clip((v - lo) / (hi - lo), 0, 1) ** (1 / gamma)
```

- `lo` / `hi` are the 0.5th and 99.5th percentiles, using the same range for all three channels so colors don't shift.
- `gamma` starts at 2.2. A **larger** gamma brightens the shadows; `+` / `-` change it by 0.2 within 1.0–5.0, and `g` turns it off (linear) and on again.
- The gamma setting carries over to the next image and applies to the zoom window too.

**This only affects the display, never the saved coordinates.**

### Region colors

Regions 1–20 use a fixed palette of bright, clearly different colors; regions above 20 get colors computed from the region number. A region always has the same color, in every session. `uv run annotate.py --palette-preview colors.png` shows them.

## Output

One CSV per image: `<out_dir>/<image_stem>.csv`, e.g. `out/adke_neel_050.csv`.

```csv
image,x,y,region_id,label,super_region_id
adke_neel_050.tif,412,630,1,texture,1
adke_neel_050.tif,415,641,1,texture,1
adke_neel_050.tif,201,88,2,specular,0
adke_neel_050.tif,330,512,3,texture,1
```

| Column | Meaning |
|---|---|
| `image` | Image filename, including its extension |
| `x`, `y` | Column and row, as integer pixels in the **original** resolution (whatever the display scale) |
| `region_id` | Region number within the image: 1, 2, 3, … |
| `label` | `material`, `texture` or `specular` |
| `super_region_id` | Super-region number within the image: 1, 2, 3, …; `0` = in no super-region |

- An image you visited but didn't annotate gets a CSV with just the header line. That's how you tell "checked, nothing to mark" apart from "not visited yet".
- **Reopening an image loads its CSV**, so you can stop and continue later. New points go into a new region after the existing ones; right-click or `[` / `]` to add to an old one. After reopening, undo removes points in file order (last row first); deletes made after reopening can be undone as usual.
- Rows are sorted by region, keeping the order you placed them within each region. Region numbers are renumbered 1, 2, 3, … in order when saving, so gaps left by undo or by deleting a whole region are never saved. On screen the old numbers stay until you reopen the image (e.g. `n` then `b`), so they can differ from the CSV in the meantime.
- Super-region numbers are also renumbered 1, 2, 3, … when saving, in order of their lowest region. Older CSVs without the `super_region_id` column still load (no super-regions).
- Saves first write to a temporary file and then replace the old CSV, so a crash mid-save can't leave a half-written file.

### Checking your output

`check_csv.py` checks every CSV in an output folder: points are inside the image, no pixel is used twice, each region has a single label, and region ids are 1..k. For super-regions it checks that each region has one `super_region_id`, each super-region has one label, and super-region ids are 1..K.

```bash
uv run check_csv.py samples out
```

## Development

```bash
uv run pytest -q
```

Everything in `annotate.py` that doesn't need a window is a plain function with pytest tests in `tests/test_annotate.py`. That covers CSV reading and writing, click-to-pixel mapping, zoom mapping, display tone mapping and the gamma lookup table, the region/label rules, the add/delete undo stack, dot picking (PICK mode, right-click, Ctrl+click) and deleting, `[`/`]` cycling, the palette, panel layout and key handling. The pytest settings are in `pyproject.toml`. See `CLAUDE.md` for the design spec.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ImportError` or odd `cv2`/`numpy` versions | ROS `PYTHONPATH` is leaking in; run with `env -u PYTHONPATH uv run …` |
| `The function is not implemented … GTK/Qt` | You have `opencv-python-headless`; this tool needs `opencv-python` (run `uv sync`) |
| No window appears (WSL/SSH) | Make sure a display is available (WSLg on Windows 11, or X forwarding) |
| Keys do nothing | Click the image window to give it focus; check Caps Lock |
| Right-click doesn't select anything | Use `r` and then a left click (or Ctrl+click, or `[` / `]`). To see what's happening, run with `--debug-mouse`: if no `event=RBUTTONDOWN` line appears when you right-click, your window system isn't passing right-clicks to OpenCV |
| `no dot near click` | The click wasn't within 10 px of a dot. Click closer (watch for the cyan ring), or open zoom (`z`) and pick there |
| `x` says `no dot near cursor` | Move until the cyan ring shows on the dot (in zoom if dots are crowded), then press `x` |
| Image too dark / washed out | Press `+` / `-` to change gamma, or `g` to turn it off |
| `CSV name collision: a.tif and a.png` | Two images share a filename stem and would write the same CSV; rename one |
| `…csv:N: region R has super_region_id …` or `super-region S mixes labels …` on load | The CSV was edited by hand so a region is in two super-regions, or a super-region mixes labels; fix line N |
| `…csv:N: unknown label …` on load | That CSV was edited by hand and has a bad row; fix line N (the tool refuses to drop data silently) |
| `cannot read image` | The file is corrupt or not a real image; move it out of the input folder |
