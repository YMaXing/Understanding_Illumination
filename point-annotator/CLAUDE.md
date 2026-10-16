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

### Super-regions (same material/texture across separate regions)
A region stays single-surface (material segmentation). A super-region groups several regions of the same material or texture even when other surfaces lie between them (intrinsic imaging), e.g. one region per red brick, all bricks in one super-region.
- Membership is stored beside the points (`super_of: region_id -> super_id`); Point is unchanged. A region is in at most one super; only regions with >= 1 point can join. A super's label is its members' label; adding a region with a different label is rejected. Super ids are per image, one id space across labels; new id = max + 1. A single-member super is kept; a super that loses its last member ceases to exist (and is deselected if it was current_super)
- Super-region operations never change points, region ids, region labels or the current region
- c: create a new super from the current region and select it. Rejected if the region is empty or already in a super (use a to move it)
- a: add the current region to current_super; if it is in another super it moves ("moved from S3 to S5"). Rejected: no super selected ("no super-region selected"), label mismatch, empty region, already a member
- d: drop the current region from its super (region and points stay)
- , / .: select previous / next super by id, wrapping; with none selected, . starts at the lowest id and , at the highest. Not undoable
- y: SUPER-PICK mode: the next left-click near a dot (nearest within ~10 display px, ties to the most recent, hover ring shown) selects that dot's region's super; the current region does not change. A miss ("no dot near click") or a region in no super ("region N is in no super-region") stays in SUPER-PICK; y again or Esc cancels. r and y switch between the two pick modes. Not undoable
- v: show/hide the super-region display (carries over to the next image; never affects state or CSV). c/a/d/,/./y turn the display on and say so
- u undoes c/a/d on the same chronological stack as point add/delete (restores the previous memberships and current_super). Deleting a region's last point (x, or undo of an add) removes the region from its super; undoing that delete restores the membership

## Command line
`uv run annotate.py <image_dir> <out_dir> [--max-size WxH] [--panel auto|right|bottom] [--debug-mouse]`
- --max-size: window budget incl. the panel (default 1600x900)
- --panel: auto = right for portrait images (h > w), bottom otherwise
- --debug-mouse: print GUI backend, window flags (WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL) and every mouse button event to the terminal
- `uv run annotate.py --palette-preview out.png`: write a swatch grid of region colors 1-40 and exit

## Output
One CSV per image: `<out_dir>/<image_stem>.csv`, columns `image,x,y,region_id,label,super_region_id`
- One row per point, sorted by (region_id, insertion order); region ids renumbered contiguously 1..k in id order on save
- x = column, y = row, integer pixels in ORIGINAL image resolution
- region_id is per image, starting at 1
- label ∈ {material, texture, specular}
- Each region has exactly one label: switching label while the current region has points starts a new region
- Reopening an image loads its existing CSV
- Ids are not renumbered in memory: after deleting a whole region, on-screen region ids can differ from the CSV until the image is reloaded (region id text is then drawn from the renumbered ids)
- super_region_id is appended as the 6th column; 0 = in no super-region. Older 5-column files still load (all 0)
- On load, a clear error (file:line) if one region's rows carry different super_region_id values, or one super's regions have different labels. After load no super is selected; the next new super id is max + 1
- On save: region ids renumbered 1..k (as before), then super ids renumbered 1..K in order of their smallest member region id, membership remapped consistently. Saves stay atomic, rows sorted as before
- check_csv.py also checks: one super_region_id per region, one label per super, super ids contiguous 1..K (ignoring 0)

## Display
- Image at canvas origin (0,0) so coordinate mapping is unchanged; downscale to fit --max-size minus the panel (compute_scale subtracts the panel width/height); map clicks back via the display size
- Status/help panel: right (560 px wide, full height) for portrait, below (~220 px, grows if the text needs it; its measured height is subtracted) for landscape. ~20 px lines, word-wrapped, grouped headings. State block: swatch + label, region N (new / existing, K pts), totals, image i/N (n/b) + filename, gamma value on/off, zoom on/off, mode ADD/PICK. Message line shows the last action for ~3 s
- Display value = clip((v-lo)/(hi-lo), 0, 1) ** (1/gamma), one joint lo/hi (0.5/99.5 pct) over all channels, computed once per image; a cached 65536-entry LUT per gamma (numpy indexing). Display only, CSV unaffected
- Region colors: PALETTE_20 (first 8 original; 9-20 bright, L* >= 45; min pairwise CIELAB dE >= 25). region_id > 20: golden-angle hue (137.50776 deg * id), S 0.85, V 1.0 - a function of region_id only
- Marker shape by label, color by region, region id text beside each region's first point
- Ring: cyan ring around the hovered dot that x would delete / a pick would select (both windows), drawn outside any super ring
- Super-regions (when shown, both windows): an outer ring around every dot of every member region in its super's color (~3 px; ~5 px for current_super), under the current-region white outline; the id text beside each member region's first point becomes "6>S2"; for current_super only, thin lines from each member's first point to the first point of its lowest-id member. PALETTE_SUPER: 12 explicit bright saturated colors (L* >= 55), indexed (super_id-1) % 12, pairwise CIELAB dE >= 25, chosen away from PALETTE_20
- Panel: "region N: in S2 / in no super", "super: S2 (texture), regions 3,5,9" or "super: none selected", display ON/OFF, and a list of supers (one line each, "+N more" past 4 lines). Bottom layout has 5 columns (status | super-regions | 3 help columns, up to 1400 px wide, capped by --max-size); the right panel puts help in two sub-columns

## Rules
- Keep CSV I/O and coordinate mapping in pure functions, tested with pytest
- Manage dependencies with uv only: `uv add` / `uv add --dev`. Never pip install; no requirements.txt
- Run everything via `uv run` (tests: `uv run pytest -q`)
- ROS Jazzy is sourced in the shell and leaks into sys.path. Always run `env -u PYTHONPATH uv run ...`