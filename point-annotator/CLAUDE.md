# Sparse point annotator (OpenCV)
Ground-truth tool for material/texture point annotations (Understanding Illumination project).

## Stack
Python 3.10+, single file `annotate.py`, `opencv-python` (NOT -headless), numpy. No other GUI libs.
Run: `env -u PYTHONPATH uv run annotate.py <image_dir> <out_dir>` (see Command line)

## Controls (letter keys only; arrow-key codes from cv2.waitKey are platform-dependent)
Modes: ADD (default) and PICK. Only q quits (main window X also saves + quits); Esc never quits.
- Left click: ADD mode adds a point to the current region; PICK mode selects the region of the nearest dot within ~10 display px (hit returns to ADD; miss shows "no dot near click" and stays in PICK). Clicks in the panel are ignored
- r: toggle PICK mode (Esc also cancels). Right-click and Ctrl+left-click pick in either mode (optional shortcuts; right-click may not arrive on every backend, check with --debug-mouse)
- Ties when picking/deleting go to the nearest dot, then the most recently added
- [ / ]: cycle current region down / up through 1..k, (new), wrapping; (new) = max(region_id)+1 and keeps the current label
- After a selection the panel shows "ADDING TO existing region N"; further clicks add to it. The current region's dots always get a thick white outline (both windows)
- m / t / p: set label = material / texture / specular. Same label: no-op. If the current region has points, start a new region max(region_id)+1 with that label and show "new region N started"; an empty (new) region is just relabeled. A region's label never changes once it has points
- space: start new region max(region_id)+1 (not current+1); no-op if current region is empty
- x: delete the dot under the ring. Hovering draws a ring around the nearest dot within ~10 display px (main or zoom window). No dot in range: no-op. Use x, not Delete (cv2.waitKey special-key codes are platform-dependent)
- Deleting never changes the current region: if it empties the current region, that stays current (empty, same label); emptying another region leaves the current region as is
- u: undo the last action from a stack of add / delete actions (no redo). Undo add: remove the point; current region/label become those of the new last point, or region 1 / material if none are left. Undo delete: restore the point at its original position, region and label, and restore the current region/label from just before the delete. Reopening an image seeds the stack with one add per loaded row (undo removes the last row first)
- z: open zoom (frozen center at the cursor when pressed; z again re-centers). Esc closes it from either window (Esc first cancels PICK if active). Closing zoom with its X just resets the zoom state
- g: gamma on/off (keeps the value). + (or =) / - (or _): gamma up / down by 0.2 in [1.0, 5.0], default 2.2; if gamma is off, either key only turns it on. Gamma persists across images and applies to zoom
- s: save. n / b: next / previous image (autosave); on the last / first image they show "last image" / "first image" and do nothing
- q: quit (autosave)

## Command line
`uv run annotate.py <image_dir> <out_dir> [--max-size WxH] [--panel auto|right|bottom] [--debug-mouse]`
- --max-size: window budget incl. the panel (default 1600x900)
- --panel: auto = right for portrait images (h > w), bottom otherwise
- --debug-mouse: print GUI backend, window flags (WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL) and every mouse button event to the terminal
- `uv run annotate.py --palette-preview out.png`: write a swatch grid of region colors 1-40 and exit

## Output
One CSV per image: `<out_dir>/<image_stem>.csv`, columns `image,x,y,region_id,label`
- One row per point, sorted by (region_id, insertion order); region ids renumbered contiguously 1..k in id order on save
- x = column, y = row, integer pixels in ORIGINAL image resolution
- region_id is per image, starting at 1
- label ∈ {material, texture, specular}
- Each region has exactly one label: switching label while the current region has points starts a new region
- Reopening an image loads its existing CSV
- Ids are not renumbered in memory: after deleting a whole region, on-screen region ids can differ from the CSV until the image is reloaded (region id text is then drawn from the renumbered ids)

## Display
- Image at canvas origin (0,0) so coordinate mapping is unchanged; downscale to fit --max-size minus the panel (compute_scale subtracts the panel width/height); map clicks back via the display size
- Status/help panel: right (560 px wide, full height) for portrait, below (~220 px, grows if the text needs it; its measured height is subtracted) for landscape. ~20 px lines, word-wrapped, grouped headings. State block: swatch + label, region N (new / existing, K pts), totals, image i/N (n/b) + filename, gamma value on/off, zoom on/off, mode ADD/PICK. Message line shows the last action for ~3 s
- Display value = clip((v-lo)/(hi-lo), 0, 1) ** (1/gamma), one joint lo/hi (0.5/99.5 pct) over all channels, computed once per image; a cached 65536-entry LUT per gamma (numpy indexing). Display only, CSV unaffected
- Region colors: PALETTE_20 (first 8 original; 9-20 bright, L* >= 45; min pairwise CIELAB dE >= 25). region_id > 20: golden-angle hue (137.50776 deg * id), S 0.85, V 1.0 - a function of region_id only
- Marker shape by label, color by region, region id text beside each region's first point
- Ring: cyan ring around the hovered dot that x would delete / a pick would select (both windows)

## Rules
- Keep CSV I/O and coordinate mapping in pure functions, tested with pytest
- Manage dependencies with uv only: `uv add` / `uv add --dev`. Never pip install; no requirements.txt
- Run everything via `uv run` (tests: `uv run pytest -q`)
- ROS Jazzy is sourced in the shell and leaks into sys.path. Always run `env -u PYTHONPATH uv run ...`