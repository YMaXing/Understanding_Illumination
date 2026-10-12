"""Sparse point annotator for material / texture / specular ground truth.

Usage: uv run annotate.py <image_dir> <out_dir> [--max-size WxH] [--panel auto|right|bottom]
                          [--debug-mouse]
       uv run annotate.py --palette-preview out.png

Keys (also listed in the on-screen help panel):
  m/t/p label, click add point, space new region, [ ] previous/next region,
  r pick a region by clicking one of its dots (right-click / Ctrl+click also work),
  x delete the ringed dot, u undo, z zoom (Esc closes), g gamma on/off,
  + / - gamma up/down, s save, n/b next/previous image, q quit.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LABELS = ("material", "texture", "specular")
LABEL_KEYS = {ord("m"): "material", ord("t"): "texture", ord("p"): "specular"}
MARKERS = {"material": "circle", "texture": "square", "specular": "triangle"}
CSV_HEADER = ["image", "x", "y", "region_id", "label"]
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# no Qt toolbar / status bar / built-in wheel zoom
WINDOW_FLAGS = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL
ZOOM = 4
ZOOM_HALF = 64  # crop is 128x128 original pixels -> 512x512 window
SELECT_RADIUS = 10  # pick / hover-delete radius, canvas px (both windows)
LO_PCT, HI_PCT = 0.5, 99.5

# status/help panel: right of portrait images, below landscape ones
PANEL_RIGHT_W = 560
PANEL_BOTTOM_H = 220  # nominal; grows if the text needs more room
BOTTOM_MIN_W = 1200  # canvas width for the 3-column bottom layout
PANEL_PAD = 12
LINE_H = 20
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
MSG_SECONDS = 3.0

# display = clip((v-lo)/(hi-lo), 0, 1) ** (1/gamma); larger gamma brightens shadows
GAMMA_DEFAULT = 2.2
GAMMA_STEP = 0.2
GAMMA_MIN, GAMMA_MAX = 1.0, 5.0

# BGR. The first 8 are the original colors; 9-20 are bright (L* >= 45) and
# every pair is >= 25 apart in CIELAB (checked in tests).
PALETTE_20 = [
    (75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0),
    (48, 130, 245), (180, 30, 145), (240, 240, 70), (230, 50, 240),
    (0, 255, 0), (80, 159, 178), (169, 98, 217), (255, 121, 64),
    (80, 100, 178), (0, 25, 255), (0, 217, 152), (179, 255, 0),
    (153, 0, 255), (178, 159, 80), (139, 178, 80), (98, 217, 193),
]
GOLDEN_ANGLE = 137.50776  # degrees; hue step for region ids beyond the palette

# ---------------------------------------------------------------- state


@dataclass(frozen=True)
class Point:
    x: int
    y: int
    region_id: int
    label: str


@dataclass(frozen=True)
class Action:
    """One undoable edit. prev_* is the current region/label before it."""
    kind: str  # "add" | "delete"
    point: Point
    index: int  # position in AnnotationState.points
    prev_region: int
    prev_label: str


@dataclass
class AnnotationState:
    points: list[Point] = field(default_factory=list)
    region_id: int = 1
    label: str = "material"
    history: list[Action] = field(default_factory=list)

    def region_points(self, region_id: int | None = None) -> list[Point]:
        rid = self.region_id if region_id is None else region_id
        return [p for p in self.points if p.region_id == rid]

    def is_used(self, x: int, y: int) -> bool:
        return any(p.x == x and p.y == y for p in self.points)

    def add_point(self, x: int, y: int) -> bool:
        """Add a point to the current region; reject pixels used by any region."""
        if self.is_used(x, y):
            return False
        p = Point(x, y, self.region_id, self.label)
        self.history.append(Action("add", p, len(self.points), self.region_id, self.label))
        self.points.append(p)
        return True

    def delete_point(self, x: int, y: int) -> Point | None:
        """Delete the point at (x, y); the current region is left unchanged,
        even if this empties it."""
        for i, p in enumerate(self.points):
            if p.x == x and p.y == y:
                del self.points[i]
                self.history.append(Action("delete", p, i, self.region_id, self.label))
                return p
        return None

    # Invariant: region_id is an existing region, next_region_id(), or a region
    # just emptied by a delete; an empty current region acts as the "new" one.

    def region_ids(self) -> list[int]:
        return sorted({p.region_id for p in self.points})

    def next_region_id(self) -> int:
        return max(self.region_ids(), default=0) + 1

    def is_existing(self) -> bool:
        return bool(self.region_points())

    def new_region(self) -> bool:
        """Start region max+1; no-op if the current region is empty."""
        if not self.is_existing():
            return False
        self.region_id = self.next_region_id()
        return True

    def set_label(self, label: str) -> None:
        """Same label: no-op. Otherwise relabel the empty new region, or open
        max+1 with the label; a region with points is never relabeled."""
        if label not in LABELS:
            raise ValueError(f"unknown label: {label}")
        if label == self.label:
            return
        self.new_region()
        self.label = label

    def select_region(self, region_id: int) -> bool:
        """Make an existing region current, taking its label."""
        pts = self.region_points(region_id)
        if not pts:
            return False
        self.region_id, self.label = region_id, pts[0].label
        return True

    def cycle_region(self, step: int) -> int:
        """Move through 1..k, (new), wrapping; (new) keeps the current label."""
        slots = self.region_ids() + [self.next_region_id()]
        i = slots.index(self.region_id) if self.is_existing() else len(slots) - 1
        target = slots[(i + step) % len(slots)]
        if not self.select_region(target):
            self.region_id = target
        return self.region_id

    def undo(self) -> str | None:
        """Revert the last action; return its kind, or None if there is none.

        Undoing an add removes the point; region/label follow the new last
        point (region 1 / material if none). Undoing a delete restores the
        point at its old position and the region/label from before the delete.
        Actions are undone newest-first, so the index and pixel are always free.
        """
        if not self.history:
            return None
        a = self.history.pop()
        if a.kind == "add":
            del self.points[a.index]
            if self.points:
                last = self.points[-1]
                self.region_id, self.label = last.region_id, last.label
            else:
                self.region_id, self.label = 1, "material"
        else:
            self.points.insert(a.index, a.point)
            self.region_id, self.label = a.prev_region, a.prev_label
        return a.kind


def state_from_points(points: list[Point]) -> AnnotationState:
    """State for a reopened image: a fresh empty region after the existing ones.
    History is seeded with one add per loaded point, so undo removes them last-row-first."""
    if not points:
        return AnnotationState()
    points = list(points)
    return AnnotationState(
        points=points,
        region_id=max(p.region_id for p in points) + 1,
        label=points[-1].label,
        history=[Action("add", p, i, p.region_id, p.label) for i, p in enumerate(points)],
    )


def renumber_regions(points: list[Point]) -> list[Point]:
    """Map region ids to 1..k in order of first appearance."""
    mapping: dict[int, int] = {}
    for p in points:
        mapping.setdefault(p.region_id, len(mapping) + 1)
    return [Point(p.x, p.y, mapping[p.region_id], p.label) for p in points]


def rows_for_save(points: list[Point]) -> list[Point]:
    """Sort by (region_id, insertion order), then renumber ids to 1..k."""
    return renumber_regions(sorted(points, key=lambda p: p.region_id))


def nearest_point(points, click, to_canvas, radius) -> Point | None:
    """Point whose drawn position is nearest the click, within radius canvas px.

    to_canvas maps original -> canvas coords (None = not drawn). Ties go to the
    most recently added point, which is drawn on top.
    """
    best, best_d2 = None, radius * radius
    for p in points:
        c = to_canvas(p.x, p.y)
        if c is None:
            continue
        d2 = (c[0] - click[0]) ** 2 + (c[1] - click[1]) ** 2
        if d2 <= best_d2:
            best, best_d2 = p, d2
    return best


def region_first_points(points: list[Point]) -> dict[int, tuple[int, int]]:
    first: dict[int, tuple[int, int]] = {}
    for p in points:
        first.setdefault(p.region_id, (p.x, p.y))
    return first


def color_for_region(region_id: int) -> tuple[int, int, int]:
    """BGR color that depends only on region_id, so it is stable across sessions."""
    if 1 <= region_id <= len(PALETTE_20):
        return PALETTE_20[region_id - 1]
    hue = (region_id * GOLDEN_ANGLE) % 360.0
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.85, 1.0)
    return round(b * 255), round(g * 255), round(r * 255)


def palette_preview(n: int = 40, cols: int = 8, cell: int = 80) -> np.ndarray:
    """Swatch grid of regions 1..n, each labeled with its id."""
    rows = -(-n // cols)
    img = np.full((rows * cell, cols * cell, 3), 255, np.uint8)
    for rid in range(1, n + 1):
        r, c = divmod(rid - 1, cols)
        x0, y0 = c * cell, r * cell
        color = color_for_region(rid)
        cv2.rectangle(img, (x0 + 4, y0 + 4), (x0 + cell - 5, y0 + cell - 5), color, -1)
        luma = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        ink = (0, 0, 0) if luma > 128 else (255, 255, 255)
        cv2.putText(img, str(rid), (x0 + 12, y0 + cell // 2 + 8), FONT, 0.8, ink, 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- CSV I/O


def write_csv(path: Path, image_name: str, points: list[Point]) -> None:
    """Atomically write one row per point (header only if there are none)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for p in rows_for_save(points):
            w.writerow([image_name, p.x, p.y, p.region_id, p.label])
    os.replace(tmp, path)


def read_csv(path: Path) -> list[Point]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != CSV_HEADER:
            raise ValueError(f"{path}: expected header {CSV_HEADER}, got {header}")
        points = []
        for lineno, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(CSV_HEADER):
                raise ValueError(f"{path}:{lineno}: expected {len(CSV_HEADER)} columns")
            _, xs, ys, rs, label = row
            try:
                x, y, rid = int(xs), int(ys), int(rs)
            except ValueError:
                raise ValueError(f"{path}:{lineno}: non-integer x/y/region_id") from None
            if label not in LABELS:
                raise ValueError(f"{path}:{lineno}: unknown label {label!r}")
            if x < 0 or y < 0 or rid < 1:
                raise ValueError(f"{path}:{lineno}: out-of-range value")
            points.append(Point(x, y, rid, label))
    return points


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


# ---------------------------------------------------------------- display


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img[:, :, :3]


def stretch_params(img: np.ndarray) -> tuple[float, float]:
    """One lo/hi pair over all channels jointly, so chromaticity is preserved."""
    lo, hi = np.percentile(img, [LO_PCT, HI_PCT])
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def gamma_lut(lo: float, hi: float, gamma: float | None) -> np.ndarray:
    """65536-entry uint8 lookup table: clip((v-lo)/(hi-lo),0,1) ** (1/gamma).
    gamma None means linear (same as gamma 1)."""
    x = (np.arange(65536, dtype=np.float64) - lo) / (hi - lo)
    np.clip(x, 0.0, 1.0, out=x)
    if gamma is not None:
        np.power(x, 1.0 / gamma, out=x)
    return (x * 255.0 + 0.5).astype(np.uint8)


def to_display_8bit(
    img: np.ndarray, lo: float, hi: float, gamma: float | None, lut: np.ndarray | None = None
) -> np.ndarray:
    """Same affine stretch on every channel, then optional gamma.
    Integer images go through the LUT (fast); others are computed directly."""
    if img.dtype in (np.uint8, np.uint16):
        if lut is None:
            lut = gamma_lut(lo, hi, gamma)
        return lut[img]
    x = (img.astype(np.float32) - lo) / (hi - lo)
    np.clip(x, 0.0, 1.0, out=x)
    if gamma is not None:
        np.power(x, 1.0 / gamma, out=x)
    return (x * 255.0 + 0.5).astype(np.uint8)


def step_gamma(gamma: float, direction: int) -> float:
    """Gamma one step up (+1) or down (-1), clamped and rounded to avoid drift."""
    g = gamma + direction * GAMMA_STEP
    return round(min(max(g, GAMMA_MIN), GAMMA_MAX), 1)


def panel_side(w: int, h: int, mode: str = "auto") -> str:
    """'right' for portrait images, 'bottom' otherwise (unless forced)."""
    if mode in ("right", "bottom"):
        return mode
    return "right" if h > w else "bottom"


def compute_scale(
    w: int, h: int, max_w: int, max_h: int, side: str, panel: int | None = None
) -> float:
    """Scale that fits the image plus panel in max_w x max_h; never upscales.
    panel is the panel's width (right) or height (bottom); default nominal size."""
    if panel is None:
        panel = PANEL_RIGHT_W if side == "right" else PANEL_BOTTOM_H
    avail_w = max_w - (panel if side == "right" else 0)
    avail_h = max_h - (panel if side == "bottom" else 0)
    return min(1.0, max(1, avail_w) / w, max(1, avail_h) / h)


def fit_display_size(
    w: int, h: int, max_w: int, max_h: int, side: str = "bottom", panel: int | None = None
) -> tuple[int, int]:
    scale = compute_scale(w, h, max_w, max_h, side, panel)
    return max(1, round(w * scale)), max(1, round(h * scale))


def display_to_original(
    dx: int, dy: int, disp: tuple[int, int], size: tuple[int, int]
) -> tuple[int, int] | None:
    """Canvas pixel -> original pixel; None outside the image (e.g. HUD strip)."""
    (dw, dh), (w, h) = disp, size
    if dx < 0 or dy < 0 or dx >= dw or dy >= dh:
        return None
    return min(dx * w // dw, w - 1), min(dy * h // dh, h - 1)


def original_to_display(
    x: int, y: int, disp: tuple[int, int], size: tuple[int, int]
) -> tuple[int, int]:
    """Original pixel center -> canvas pixel."""
    (dw, dh), (w, h) = disp, size
    return int((x + 0.5) * dw / w), int((y + 0.5) * dh / h)


def zoom_crop_bounds(
    cx: int, cy: int, half: int, w: int, h: int
) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of a 2*half crop around (cx, cy), shifted to stay in the image."""
    size = 2 * half
    x0 = min(max(cx - half, 0), max(w - size, 0))
    y0 = min(max(cy - half, 0), max(h - size, 0))
    return x0, y0, min(x0 + size, w), min(y0 + size, h)


def zoom_to_original(
    zx: int, zy: int, bounds: tuple[int, int, int, int], zoom: int
) -> tuple[int, int] | None:
    x0, y0, x1, y1 = bounds
    if zx < 0 or zy < 0:
        return None
    x, y = x0 + zx // zoom, y0 + zy // zoom
    if x >= x1 or y >= y1:
        return None
    return x, y


def original_to_zoom(
    x: int, y: int, bounds: tuple[int, int, int, int], zoom: int
) -> tuple[int, int] | None:
    """Original pixel -> center of its block in the zoom window; None if cropped out."""
    x0, y0, x1, y1 = bounds
    if not (x0 <= x < x1 and y0 <= y < y1):
        return None
    return (x - x0) * zoom + zoom // 2, (y - y0) * zoom + zoom // 2


# ---------------------------------------------------------------- drawing


def draw_marker(canvas, c, shape, color, r, outline=(0, 0, 0), thickness=1):
    cx, cy = c
    if shape == "circle":
        cv2.circle(canvas, c, r, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, c, r, outline, thickness, cv2.LINE_AA)
    elif shape == "square":
        cv2.rectangle(canvas, (cx - r, cy - r), (cx + r, cy + r), color, -1)
        cv2.rectangle(canvas, (cx - r, cy - r), (cx + r, cy + r), outline, thickness)
    else:
        tri = np.array([[cx, cy - r - 1], [cx - r - 1, cy + r], [cx + r + 1, cy + r]], np.int32)
        cv2.fillPoly(canvas, [tri], color, cv2.LINE_AA)
        cv2.polylines(canvas, [tri], True, outline, thickness, cv2.LINE_AA)


def draw_text(canvas, text, org, color=(255, 255, 255), scale=0.45, thickness=1, outline=True):
    if outline:
        cv2.putText(canvas, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def text_width(text: str, scale: float = FONT_SCALE, thickness: int = 1) -> int:
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


def wrap_text(text: str, width_px: int, scale: float = FONT_SCALE, thickness: int = 1) -> list[str]:
    """Greedy word wrap to width_px; a single over-long word gets its own line."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}" if line else word
        if line and text_width(trial, scale, thickness) > width_px:
            lines.append(line)
            line = word
        else:
            line = trial
    if line or not lines:
        lines.append(line)
    return lines


def draw_ring(canvas, c, r):
    """Delete-preview ring around a marker of radius r: cyan on black."""
    cv2.circle(canvas, c, r + 5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(canvas, c, r + 5, (255, 255, 0), 1, cv2.LINE_AA)


def draw_markers(canvas, points, to_canvas, r, current_region=None):
    """Shape by label, color by region, region_id beside each region's first point.
    The current region's dots get a thicker white outline."""
    for p in points:
        c = to_canvas(p.x, p.y)
        if c is None:
            continue
        if p.region_id == current_region:
            draw_marker(canvas, c, MARKERS[p.label], color_for_region(p.region_id), r, (255, 255, 255), 2)
        else:
            draw_marker(canvas, c, MARKERS[p.label], color_for_region(p.region_id), r)
    for rid, (x, y) in region_first_points(points).items():
        c = to_canvas(x, y)
        if c is not None:
            draw_text(canvas, str(rid), (c[0] + r + 3, c[1] - r - 3), color_for_region(rid))


# ---------------------------------------------------------------- GUI

MOUSE_EVENT_NAMES = {
    getattr(cv2, n): n[len("EVENT_"):]
    for n in dir(cv2) if n.startswith("EVENT_") and not n.startswith("EVENT_FLAG_")
}
MOUSE_FLAG_NAMES = {
    getattr(cv2, n): n[len("EVENT_FLAG_"):] for n in dir(cv2) if n.startswith("EVENT_FLAG_")
}
BUTTON_FLAGS = cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_RBUTTON | cv2.EVENT_FLAG_MBUTTON


def describe_mouse(event: int, flags: int) -> tuple[str, str]:
    name = MOUSE_EVENT_NAMES.get(event, str(event))
    names = [v for k, v in sorted(MOUSE_FLAG_NAMES.items()) if flags & k]
    return name, "|".join(names) or "0"


def gui_backend() -> str:
    fn = getattr(cv2, "currentUIFramework", None)
    if fn is not None:
        try:
            return fn()
        except cv2.error:
            pass
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


HELP_GROUPS = [
    ("Labels", [
        "m = material (one reflectance, no variation: paint, a single brick, plastic)",
        "t = texture (fine-scale variation: grass, asphalt, speckled concrete)",
        "p = specular (shiny/metal)",
    ]),
    ("Regions", [
        "click = add point",
        "space = new region",
        "[ ] = previous/next region (wraps)",
        "r = pick a region by clicking one of its dots (right-click also works)",
    ]),
    ("Edit", ["x = delete point under the ring", "u = undo last action"]),
    ("View", ["z = zoom (Esc closes)", "g = gamma on/off", "+ / - = gamma up/down"]),
    ("Files", ["s = save", "n / b = next / previous image (autosaves)", "q = quit"]),
]

PANEL_BG = (40, 40, 40)
C_TEXT = (235, 235, 235)
C_HEAD = (255, 200, 120)  # light blue
C_DIM = (170, 170, 170)
C_MSG = (0, 230, 255)  # yellow
C_ADD = (120, 230, 120)
C_PICK = (0, 165, 255)  # orange


@dataclass
class PanelLine:
    text: str
    color: tuple[int, int, int] = C_TEXT
    indent: int = 0
    heading: bool = False
    swatch: tuple[str, tuple[int, int, int]] | None = None  # (marker shape, color)


def layout_lines(items: list[PanelLine], width: int) -> list[PanelLine]:
    """Word-wrap panel lines to width; continuation lines keep the indent."""
    out: list[PanelLine] = []
    for it in items:
        extra = 22 if it.swatch else 0
        thick = 2 if it.heading else 1
        for i, t in enumerate(wrap_text(it.text, width - it.indent - extra, FONT_SCALE, thick)):
            out.append(PanelLine(t, it.color, it.indent + extra, it.heading,
                                 it.swatch if i == 0 else None))
    return out


def help_items(groups) -> list[PanelLine]:
    items: list[PanelLine] = []
    for head, entries in groups:
        items.append(PanelLine(head, C_HEAD, heading=True))
        items += [PanelLine(e, C_TEXT, indent=12) for e in entries]
    return items


def draw_lines(canvas, lines: list[PanelLine], x: int, y: int) -> None:
    for i, ln in enumerate(lines):
        base = y + (i + 1) * LINE_H - 5
        if ln.swatch:
            shape, color = ln.swatch
            draw_marker(canvas, (x + ln.indent - 13, base - 5), shape, color, 7)
        draw_text(canvas, ln.text, (x + ln.indent, base), ln.color, FONT_SCALE,
                  2 if ln.heading else 1, outline=False)


class Annotator:
    MAIN = "annotate"
    ZOOMWIN = "zoom"
    ADD, PICK = "ADD", "PICK"

    def __init__(self, images: list[Path], out_dir: Path, max_w: int, max_h: int,
                 panel: str = "auto", debug_mouse: bool = False, clock=time.monotonic):
        self.images = images
        self.out_dir = out_dir
        self.max_w, self.max_h = max_w, max_h
        self.panel_mode = panel
        self.debug_mouse = debug_mouse
        self.clock = clock
        # display settings persist across images
        self.gamma_on = True
        self.gamma_value = GAMMA_DEFAULT
        self.msg, self.msg_time = "", -1e9
        self._zoom_open = False
        self.zoom_center = self.zoom_bounds = None
        self.hover: tuple[str, tuple[int, int]] | None = None
        self.idx = -1
        self.load(0)

    # -- messages

    def say(self, text: str) -> None:
        self.msg, self.msg_time = text, self.clock()

    def message(self) -> str:
        return self.msg if self.clock() - self.msg_time < MSG_SECONDS else ""

    # -- image / state

    def csv_path(self, path: Path) -> Path:
        return self.out_dir / f"{path.stem}.csv"

    def load(self, idx: int) -> None:
        path = self.images[idx]
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"cannot read image: {path}")
        csv_path = self.csv_path(path)
        points = read_csv(csv_path) if csv_path.exists() else []

        self.idx = idx
        self.raw = to_bgr(raw)
        self.h, self.w = self.raw.shape[:2]
        self.lo, self.hi = stretch_params(self.raw)  # once per load
        self._luts: dict[float | None, np.ndarray] = {}
        self._full: dict[float | None, np.ndarray] = {}
        self._scaled: dict[float | None, np.ndarray] = {}
        self.state = state_from_points(points)
        self.mode = self.ADD
        self.cursor: tuple[int, int] | None = None
        self.hover = None
        self.close_zoom()
        self.side = panel_side(self.w, self.h, self.panel_mode)
        panel = PANEL_RIGHT_W if self.side == "right" else self.bottom_columns(BOTTOM_MIN_W)[1]
        self.disp = fit_display_size(self.w, self.h, self.max_w, self.max_h, self.side, panel)
        self.say(f"loaded {len(points)} points" if points else f"image {idx + 1}/{len(self.images)}")

    def save(self) -> None:
        path = self.images[self.idx]
        write_csv(self.csv_path(path), path.name, self.state.points)
        self.say("saved")

    def gamma(self) -> float | None:
        """Effective display gamma (None = off/linear)."""
        return self.gamma_value if self.gamma_on else None

    def full_img(self) -> np.ndarray:
        g = self.gamma()
        if g not in self._full:
            if g not in self._luts:
                self._luts[g] = gamma_lut(self.lo, self.hi, g)
            self._full[g] = to_display_8bit(self.raw, self.lo, self.hi, g, self._luts[g])
        return self._full[g]

    def scaled_img(self) -> np.ndarray:
        g = self.gamma()
        if g not in self._scaled:
            full = self.full_img()
            if self.disp == (self.w, self.h):
                self._scaled[g] = full
            else:
                self._scaled[g] = cv2.resize(full, self.disp, interpolation=cv2.INTER_AREA)
        return self._scaled[g]

    def add(self, p: tuple[int, int] | None) -> None:
        if p is None:
            return
        if self.state.add_point(*p):
            self.say(f"added point to region {self.state.region_id}")
        else:
            self.say(f"pixel ({p[0]},{p[1]}) already used")

    # -- zoom

    def open_zoom(self) -> None:
        """Open zoom at the cursor, or re-center it if already open."""
        if self.cursor is None:
            self.say("move the cursor over the image first")
            return
        if not self._zoom_open:
            cv2.namedWindow(self.ZOOMWIN, WINDOW_FLAGS)
            cv2.setMouseCallback(self.ZOOMWIN, self.on_zoom_mouse)
            self._zoom_open = True
        self.zoom_center = self.cursor
        self.zoom_bounds = zoom_crop_bounds(*self.zoom_center, ZOOM_HALF, self.w, self.h)

    def close_zoom(self) -> None:
        if self._zoom_open:
            try:
                cv2.destroyWindow(self.ZOOMWIN)
            except cv2.error:
                pass
            self._zoom_open = False
        self.zoom_closed()

    def zoom_closed(self) -> None:
        """Reset zoom state (also used when the zoom window's X closed it)."""
        self.zoom_center = self.zoom_bounds = None
        if self.hover and self.hover[0] == self.ZOOMWIN:
            self.hover = None

    # -- mouse

    def log_mouse(self, window, event, x, y, flags) -> None:
        if not self.debug_mouse:
            return
        if event == cv2.EVENT_MOUSEMOVE and not flags & BUTTON_FLAGS:
            return
        name, fl = describe_mouse(event, flags)
        print(f"[mouse] win={window} event={name} x={x} y={y} flags={fl}", flush=True)

    def on_main_mouse(self, event, x, y, flags, param) -> None:
        self.log_mouse(self.MAIN, event, x, y, flags)
        p = display_to_original(x, y, self.disp, (self.w, self.h))
        self.mouse(self.MAIN, event, x, y, flags, p, self.to_display)

    def on_zoom_mouse(self, event, x, y, flags, param) -> None:
        self.log_mouse(self.ZOOMWIN, event, x, y, flags)
        if self.zoom_bounds is None:
            return
        p = zoom_to_original(x, y, self.zoom_bounds, ZOOM)
        self.mouse(self.ZOOMWIN, event, x, y, flags, p, self.to_zoom)

    def mouse(self, window, event, x, y, flags, p, to_canvas) -> None:
        """p is the original pixel under the mouse, None outside the image (panel)."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover = (window, (x, y))
            if p is not None:
                self.cursor = p
            return
        if p is None:  # clicks in the panel / outside the image are ignored
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == self.PICK or flags & cv2.EVENT_FLAG_CTRLKEY:
                self.pick_at((x, y), to_canvas)
            else:
                self.add(p)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.pick_at((x, y), to_canvas)

    def to_display(self, x: int, y: int) -> tuple[int, int]:
        return original_to_display(x, y, self.disp, (self.w, self.h))

    def to_zoom(self, x: int, y: int) -> tuple[int, int] | None:
        return original_to_zoom(x, y, self.zoom_bounds, ZOOM)

    def hover_target(self) -> Point | None:
        """Dot under the ring (x deletes it, a pick selects it): nearest to the
        mouse in the window it is over."""
        if self.hover is None:
            return None
        window, pos = self.hover
        if window == self.ZOOMWIN:
            if self.zoom_bounds is None:
                return None
            return nearest_point(self.state.points, pos, self.to_zoom, SELECT_RADIUS)
        return nearest_point(self.state.points, pos, self.to_display, SELECT_RADIUS)

    def pick_at(self, click: tuple[int, int], to_canvas) -> bool:
        """Select the region of the nearest dot; stay in PICK mode on a miss."""
        p = nearest_point(self.state.points, click, to_canvas, SELECT_RADIUS)
        if p is None:
            self.say("no dot near click")
            return False
        self.state.select_region(p.region_id)
        self.mode = self.ADD
        self.say(f"selected region {p.region_id}")
        return True

    # -- keys

    def handle_key(self, k: int) -> bool:
        """Return False to quit (only q quits)."""
        st = self.state
        if k in LABEL_KEYS:
            label = LABEL_KEYS[k]
            before = st.region_id
            st.set_label(label)  # opens max+1 if the current region has points
            if st.region_id != before:
                self.say(f"new region {st.region_id} started ({label})")
            else:
                self.say(f"label {label}")
        elif k == ord(" "):
            if st.new_region():
                self.say(f"region {st.region_id} started")
            else:
                self.say("current region is empty")
        elif k in (ord("["), ord("]")):
            rid = st.cycle_region(-1 if k == ord("[") else 1)
            self.mode = self.ADD
            self.say(f"selected region {rid}" if st.is_existing() else f"new region {rid}")
        elif k == ord("r"):
            if self.mode == self.PICK:
                self.mode = self.ADD
                self.say("pick cancelled")
            else:
                self.mode = self.PICK
                self.say("pick mode on")
        elif k == ord("u"):
            kind = st.undo()
            self.say(f"undid {kind}" if kind else "nothing to undo")
        elif k == ord("x"):
            t = self.hover_target()
            if t is None:
                self.say("no dot near cursor")
            else:
                st.delete_point(t.x, t.y)
                self.say("deleted point")
        elif k == ord("s"):
            self.save()
        elif k == ord("n"):
            if self.idx + 1 >= len(self.images):
                self.say("last image")
            else:
                self.save()
                self.load(self.idx + 1)
        elif k == ord("b"):
            if self.idx == 0:
                self.say("first image")
            else:
                self.save()
                self.load(self.idx - 1)
        elif k == ord("q"):
            return False
        elif k == ord("z"):
            self.open_zoom()
        elif k == 27:  # Esc: cancel pick, else close zoom; never quits
            if self.mode == self.PICK:
                self.mode = self.ADD
                self.say("pick cancelled")
            elif self._zoom_open or self.zoom_center is not None:
                self.close_zoom()
                self.say("zoom closed")
        elif k == ord("Z"):
            self.close_zoom()
        elif k == ord("g"):
            self.gamma_on = not self.gamma_on
            self.say(f"gamma {self.gamma_value:.1f} {'on' if self.gamma_on else 'off'}")
        elif k in (ord("+"), ord("="), ord("-"), ord("_")):
            if not self.gamma_on:
                self.gamma_on = True
            else:
                self.gamma_value = step_gamma(self.gamma_value, 1 if k in (ord("+"), ord("=")) else -1)
            self.say(f"gamma {self.gamma_value:.1f}")
        elif ord("A") <= k <= ord("Z"):
            self.say(f"unmapped key '{chr(k)}' (Caps Lock on?)")
        return True

    # -- rendering

    def state_items(self) -> list[PanelLine]:
        st = self.state
        n = len(st.region_points())
        existing = n > 0
        region = f"region {st.region_id} (existing, {n} pts)" if existing else f"region {st.region_id} (new)"
        if self.mode == self.PICK:
            mode = PanelLine("mode PICK - click a dot to select its region (r / Esc cancels)", C_PICK, heading=True)
        else:
            mode = PanelLine(
                f"mode ADD - ADDING TO {'existing' if existing else 'new'} region {st.region_id}",
                C_ADD, heading=True)
        g = f"gamma {self.gamma_value:.1f} {'on' if self.gamma_on else 'off'}"
        z = f"zoom {'on' if self.zoom_center is not None else 'off'}"
        return [
            PanelLine("Status", C_HEAD, heading=True),
            PanelLine(st.label, C_TEXT, heading=True,
                      swatch=(MARKERS[st.label], color_for_region(st.region_id))),
            PanelLine(region),
            PanelLine(f"regions {len(st.region_ids())}   points {len(st.points)}"),
            PanelLine(f"image {self.idx + 1}/{len(self.images)} (n = next, b = previous) "
                      f"{self.images[self.idx].name}"),
            PanelLine(f"{g}   {z}", C_DIM),
            mode,
            PanelLine(self.message() or " ", C_MSG),
        ]

    def bottom_columns(self, cw: int) -> tuple[list[list[PanelLine]], int]:
        """3-column bottom layout (status | labels+regions | edit+view+files) and
        its height: at least PANEL_BOTTOM_H, more if the text needs it."""
        col_w = (cw - 4 * PANEL_PAD) // 3
        cols = [
            layout_lines(self.state_items(), col_w),
            layout_lines(help_items(HELP_GROUPS[:2]), col_w),
            layout_lines(help_items(HELP_GROUPS[2:]), col_w),
        ]
        ph = max(PANEL_BOTTOM_H, 2 * PANEL_PAD + max(len(c) for c in cols) * LINE_H)
        return cols, ph

    def render(self) -> np.ndarray:
        dw, dh = self.disp
        if self.side == "right":
            width = PANEL_RIGHT_W - 2 * PANEL_PAD
            lines = layout_lines(self.state_items() + [PanelLine(" ")] + help_items(HELP_GROUPS), width)
            ph = 2 * PANEL_PAD + len(lines) * LINE_H
            canvas = np.zeros((max(dh, ph), dw + PANEL_RIGHT_W, 3), np.uint8)
            canvas[:, dw:] = PANEL_BG
            draw_lines(canvas, lines, dw + PANEL_PAD, PANEL_PAD)
        else:
            cw = max(dw, BOTTOM_MIN_W)
            cols, ph = self.bottom_columns(cw)
            col_w = (cw - 4 * PANEL_PAD) // 3
            canvas = np.zeros((dh + ph, cw, 3), np.uint8)
            canvas[dh:] = PANEL_BG
            for i, col in enumerate(cols):
                draw_lines(canvas, col, PANEL_PAD + i * (col_w + PANEL_PAD), dh + PANEL_PAD)
        canvas[:dh, :dw] = self.scaled_img()  # image at the origin: mapping unchanged
        st = self.state
        draw_markers(canvas, st.points, self.to_display, r=4, current_region=st.region_id)
        target = self.hover_target()
        if target is not None:
            draw_ring(canvas, self.to_display(target.x, target.y), 4)
        return canvas

    def render_zoom(self) -> np.ndarray:
        x0, y0, x1, y1 = self.zoom_bounds
        crop = self.full_img()[y0:y1, x0:x1]
        big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
        draw_markers(big, self.state.points, self.to_zoom, r=6, current_region=self.state.region_id)
        target = self.hover_target()
        if target is not None:
            c = self.to_zoom(target.x, target.y)
            if c is not None:
                draw_ring(big, c, 6)
        if self.cursor is not None:
            c = original_to_zoom(*self.cursor, self.zoom_bounds, ZOOM)
            if c is not None:
                a = (c[0] - ZOOM // 2, c[1] - ZOOM // 2)
                cv2.rectangle(big, a, (a[0] + ZOOM - 1, a[1] + ZOOM - 1), (255, 255, 255), 1)
        return big

    # -- main loop

    @staticmethod
    def window_visible(name: str) -> bool:
        try:
            return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def run(self) -> None:
        cv2.namedWindow(self.MAIN, WINDOW_FLAGS)
        cv2.setMouseCallback(self.MAIN, self.on_main_mouse)
        if self.debug_mouse:
            print(f"[debug] OpenCV {cv2.__version__}, GUI backend: {gui_backend()}", flush=True)
            print(f"[debug] windows created with WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL = {WINDOW_FLAGS}",
                  flush=True)
        try:
            while True:
                cv2.imshow(self.MAIN, self.render())
                if self.zoom_center is not None:
                    cv2.imshow(self.ZOOMWIN, self.render_zoom())
                k = cv2.waitKey(20) & 0xFF
                # check window close buttons before keys can open/close windows
                if not self.window_visible(self.MAIN):
                    break  # main window X: save and quit
                if self._zoom_open and not self.window_visible(self.ZOOMWIN):
                    self._zoom_open = False  # zoom window X: reset, don't crash
                    self.zoom_closed()
                if k != 255 and not self.handle_key(k):
                    break
        finally:
            self.save()
            cv2.destroyAllWindows()


def parse_size(s: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in s.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WxH, got {s!r}") from None
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"size too small: {s!r}")
    return w, h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image_dir", type=Path, nargs="?")
    ap.add_argument("out_dir", type=Path, nargs="?")
    ap.add_argument("--max-size", type=parse_size, default=(1600, 900), metavar="WxH",
                    help="max window size incl. the status/help panel (default 1600x900)")
    ap.add_argument("--panel", choices=("auto", "right", "bottom"), default="auto",
                    help="panel position; auto = right for portrait, bottom for landscape")
    ap.add_argument("--debug-mouse", action="store_true",
                    help="print mouse events and window flags to the terminal")
    ap.add_argument("--palette-preview", type=Path, metavar="OUT.png",
                    help="write a swatch grid of region colors 1-40 and exit")
    args = ap.parse_args(argv)

    if args.palette_preview is not None:
        if not cv2.imwrite(str(args.palette_preview), palette_preview(40)):
            print(f"cannot write {args.palette_preview}", file=sys.stderr)
            return 1
        print(f"wrote {args.palette_preview}")
        return 0
    if args.image_dir is None or args.out_dir is None:
        ap.error("image_dir and out_dir are required")

    images = list_images(args.image_dir)
    if not images:
        print(f"no images found in {args.image_dir}", file=sys.stderr)
        return 1
    stems: dict[str, Path] = {}
    for p in images:
        if p.stem in stems:
            print(f"CSV name collision: {stems[p.stem].name} and {p.name}", file=sys.stderr)
            return 1
        stems[p.stem] = p
    args.out_dir.mkdir(parents=True, exist_ok=True)

    Annotator(images, args.out_dir, *args.max_size, panel=args.panel,
              debug_mouse=args.debug_mouse).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
