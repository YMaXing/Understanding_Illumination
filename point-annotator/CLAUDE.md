# Sparse point annotator (OpenCV)
Ground-truth tool for material/texture point annotations (Understanding Illumination project).

## Stack
Python 3.10+, single file `annotate.py`, `opencv-python` (NOT -headless), numpy. No other GUI libs.
Run: `python annotate.py <image_dir> <out_dir>`

## Controls (letter keys only; arrow-key codes from cv2.waitKey are platform-dependent)
- Left click: add point to current region
- m / t / p: set label = material / texture / specular
- space: start new region (region_id += 1); no-op if current region is empty
- u: undo last point
- s: save; n / b: next / previous image (autosave); q: quit (autosave)
- z: toggle zoom view around cursor (for placing points away from boundaries)

## Output
One CSV per image: `<out_dir>/<image_stem>.csv`, columns `image,x,y,region_id,label`
- One row per point
- x = column, y = row, integer pixels in ORIGINAL image resolution
- region_id is per image, starting at 1
- label ∈ {material, texture, specular}
- Each region has exactly one label: switching label while the current region has points starts a new region
- Reopening an image loads its existing CSV

## Display
- Downscale to fit screen; map clicks back to original coordinates via the scale factor
- One dot color per region; HUD showing image name, label mode, region_id, point count

## Rules
- Keep CSV I/O and coordinate mapping in pure functions, tested with pytest
- Manage dependencies with uv only: `uv add` / `uv add --dev`. Never pip install; no requirements.txt
- Run everything via `uv run` (tests: `uv run pytest -q`)
- ROS Jazzy is sourced in the shell and leaks into sys.path. Always run `env -u PYTHONPATH uv run ...`