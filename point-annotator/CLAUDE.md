# Sparse point annotator (OpenCV)
Ground-truth tool for material/texture point annotations (Understanding Illumination project).

## Stack
Python 3.10+, single file `annotate.py`, `opencv-python` (NOT -headless), numpy. No other GUI libs.
Run: `python annotate.py <image_dir> <out_dir>`

## Controls (letter keys only; arrow-key codes from cv2.waitKey are platform-dependent)
- Left click: add point to current region
- Right click: select the region of the nearest dot within ~10 display px (main or zoom window); ties go to the most recently added dot. Current region and label switch to that region's
- [ / ]: cycle current region down / up through 1..k, (new), wrapping; (new) = max(region_id)+1 and keeps the current label
- m / t / p: set label = material / texture / specular. Same label as current: no-op. If the current region has points, start a new region max(region_id)+1 with that label; an empty (new) region is just relabeled. A region's label never changes once it has points
- space: start new region max(region_id)+1 (not current+1); no-op if current region is empty
- x: delete the dot previewed under the cursor. Hovering draws a ring around the nearest dot within ~10 display px (main or zoom window, via the display/zoom mapping); ties go to the nearest, then the most recently added dot. No dot in range: no-op. Use x, not Delete (cv2.waitKey special-key codes are platform-dependent)
- Deleting never changes the current region: if it empties the current region, that stays current (empty, same label); emptying another region leaves the current region as is
- u: undo the last action from a stack of add / delete actions (no redo). Undo add: remove the point; current region/label become those of the new last point, or region 1 / material if none are left. Undo delete: restore the point at its original position, region and label, and restore the current region/label from just before the delete. Reopening an image seeds the stack with one add per loaded row (undo removes the last row first)
- s: save; n / b: next / previous image (autosave); q: quit (autosave)
- z: toggle zoom view around cursor (for placing points away from boundaries)

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
- Downscale to fit screen; map clicks back to original coordinates via the scale factor
- One dot color per region; HUD showing image name, label mode, region_id, point count
- Current region's dots get a thicker white outline (main and zoom window); HUD shows `region N (existing, k pts)` or `region N (new)`
- Delete preview: cyan ring around the hovered dot that x would delete (both windows)

## Rules
- Keep CSV I/O and coordinate mapping in pure functions, tested with pytest
- Manage dependencies with uv only: `uv add` / `uv add --dev`. Never pip install; no requirements.txt
- Run everything via `uv run` (tests: `uv run pytest -q`)
- ROS Jazzy is sourced in the shell and leaks into sys.path. Always run `env -u PYTHONPATH uv run ...`