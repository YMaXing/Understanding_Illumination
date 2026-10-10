"""Sparse point annotator for material / texture / specular ground truth.

Usage: uv run annotate.py <image_dir> <out_dir> [--max-size WxH]

Keys: m/t/p label, space new region, u undo, s save, n/b next/prev,
z zoom (re-center), Z/esc close zoom, g gamma, q quit.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LABELS = ("material", "texture", "specular")
LABEL_KEYS = {ord("m"): "material", ord("t"): "texture", ord("p"): "specular"}
MARKERS = {"material": "circle", "texture": "square", "specular": "triangle"}
CSV_HEADER = ["image", "x", "y", "region_id", "label"]
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

STRIP_H = 66  # HUD strip below the image (3 text lines)
HUD_MIN_W = 640  # wide enough for the one-line key legend (~450 px)
# no Qt toolbar / status bar / built-in wheel zoom
WINDOW_FLAGS = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL
ZOOM = 4
ZOOM_HALF = 64  # crop is 128x128 original pixels -> 512x512 window
LO_PCT, HI_PCT = 0.5, 99.5
GAMMA = 1 / 2.2

# BGR, distinct and readable on dark and light backgrounds
PALETTE = [
    (75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0),
    (48, 130, 245), (180, 30, 145), (240, 240, 70), (230, 50, 240),
    (60, 245, 210), (212, 190, 250), (128, 128, 0), (255, 190, 220),
]


# ---------------------------------------------------------------- state


@dataclass(frozen=True)
class Point:
    x: int
    y: int
    region_id: int
    label: str


@dataclass
class AnnotationState:
    points: list[Point] = field(default_factory=list)
    region_id: int = 1
    label: str = "material"

    def region_points(self, region_id: int | None = None) -> list[Point]:
        rid = self.region_id if region_id is None else region_id
        return [p for p in self.points if p.region_id == rid]

    def is_used(self, x: int, y: int) -> bool:
        return any(p.x == x and p.y == y for p in self.points)

    def add_point(self, x: int, y: int) -> bool:
        """Add a point to the current region; reject pixels used by any region."""
        if self.is_used(x, y):
            return False
        self.points.append(Point(x, y, self.region_id, self.label))
        return True

    def new_region(self) -> bool:
        """Start a new region; no-op if the current region is empty."""
        if not self.region_points():
            return False
        self.region_id = max([self.region_id] + [p.region_id for p in self.points]) + 1
        return True

    def set_label(self, label: str) -> None:
        """Relabel an empty region, or start a new region if it has points."""
        if label not in LABELS:
            raise ValueError(f"unknown label: {label}")
        if label == self.label:
            return
        self.new_region()
        self.label = label

    def undo(self) -> bool:
        """Remove the most recent point; fall back to the previous region if emptied."""
        if not self.points:
            return False
        p = self.points.pop()
        self.region_id, self.label = p.region_id, p.label
        if not self.region_points() and self.points:
            last = self.points[-1]
            self.region_id, self.label = last.region_id, last.label
        return True


def state_from_points(points: list[Point]) -> AnnotationState:
    """State for a reopened image: a fresh empty region after the existing ones."""
    if not points:
        return AnnotationState()
    return AnnotationState(
        points=list(points),
        region_id=max(p.region_id for p in points) + 1,
        label=points[-1].label,
    )


def renumber_regions(points: list[Point]) -> list[Point]:
    """Map region ids to 1..k in order of first appearance."""
    mapping: dict[int, int] = {}
    for p in points:
        mapping.setdefault(p.region_id, len(mapping) + 1)
    return [Point(p.x, p.y, mapping[p.region_id], p.label) for p in points]


def region_first_points(points: list[Point]) -> dict[int, tuple[int, int]]:
    first: dict[int, tuple[int, int]] = {}
    for p in points:
        first.setdefault(p.region_id, (p.x, p.y))
    return first


def region_color(region_id: int) -> tuple[int, int, int]:
    return PALETTE[(region_id - 1) % len(PALETTE)]


# ---------------------------------------------------------------- CSV I/O


def write_csv(path: Path, image_name: str, points: list[Point]) -> None:
    """Atomically write one row per point (header only if there are none)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for p in renumber_regions(points):
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


def to_display_8bit(img: np.ndarray, lo: float, hi: float, gamma: bool) -> np.ndarray:
    """Apply the same affine stretch to every channel, then optional gamma."""
    x = (img.astype(np.float32) - lo) / (hi - lo)
    np.clip(x, 0.0, 1.0, out=x)
    if gamma:
        np.power(x, GAMMA, out=x)
    return (x * 255.0 + 0.5).astype(np.uint8)


def fit_display_size(w: int, h: int, max_w: int, max_h: int) -> tuple[int, int]:
    """Display size that fits in max_w x (max_h - STRIP_H); never upscales."""
    avail_h = max(1, max_h - STRIP_H)
    scale = min(1.0, max_w / w, avail_h / h)
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


def draw_marker(canvas, c, shape, color, r):
    cx, cy = c
    outline = (0, 0, 0)
    if shape == "circle":
        cv2.circle(canvas, c, r, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, c, r, outline, 1, cv2.LINE_AA)
    elif shape == "square":
        cv2.rectangle(canvas, (cx - r, cy - r), (cx + r, cy + r), color, -1)
        cv2.rectangle(canvas, (cx - r, cy - r), (cx + r, cy + r), outline, 1)
    else:
        tri = np.array([[cx, cy - r - 1], [cx - r - 1, cy + r], [cx + r + 1, cy + r]], np.int32)
        cv2.fillPoly(canvas, [tri], color, cv2.LINE_AA)
        cv2.polylines(canvas, [tri], True, outline, 1, cv2.LINE_AA)


def draw_text(canvas, text, org, color=(255, 255, 255), scale=0.45):
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_markers(canvas, points, to_canvas, r):
    """Shape by label, color by region, region_id beside each region's first point."""
    for p in points:
        c = to_canvas(p.x, p.y)
        if c is not None:
            draw_marker(canvas, c, MARKERS[p.label], region_color(p.region_id), r)
    for rid, (x, y) in region_first_points(points).items():
        c = to_canvas(x, y)
        if c is not None:
            draw_text(canvas, str(rid), (c[0] + r + 3, c[1] - r - 3), region_color(rid))


# ---------------------------------------------------------------- GUI


class Annotator:
    MAIN = "annotate"
    ZOOMWIN = "zoom"

    def __init__(self, images: list[Path], out_dir: Path, max_w: int, max_h: int):
        self.images = images
        self.out_dir = out_dir
        self.max_w, self.max_h = max_w, max_h
        self.gamma = True
        self.status = ""
        self.idx = -1
        self.load(0)

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
        self.lo, self.hi = stretch_params(self.raw)
        self.disp = fit_display_size(self.w, self.h, self.max_w, self.max_h)
        self._full: dict[bool, np.ndarray] = {}
        self._scaled: dict[bool, np.ndarray] = {}
        self.state = state_from_points(points)
        self.cursor: tuple[int, int] | None = None
        self.close_zoom()
        self.status = f"loaded {len(points)} pts" if points else ""

    def save(self) -> None:
        path = self.images[self.idx]
        write_csv(self.csv_path(path), path.name, self.state.points)
        self.status = f"saved {len(self.state.points)} pts"

    def full_img(self) -> np.ndarray:
        if self.gamma not in self._full:
            self._full[self.gamma] = to_display_8bit(self.raw, self.lo, self.hi, self.gamma)
        return self._full[self.gamma]

    def scaled_img(self) -> np.ndarray:
        if self.gamma not in self._scaled:
            full = self.full_img()
            if self.disp == (self.w, self.h):
                self._scaled[self.gamma] = full
            else:
                self._scaled[self.gamma] = cv2.resize(full, self.disp, interpolation=cv2.INTER_AREA)
        return self._scaled[self.gamma]

    def add(self, p: tuple[int, int] | None) -> None:
        if p is None:
            return
        if self.state.add_point(*p):
            self.status = f"added ({p[0]},{p[1]})"
        else:
            self.status = f"pixel ({p[0]},{p[1]}) already used"

    # -- zoom

    def open_zoom(self) -> None:
        if self.cursor is None:
            self.status = "move the cursor over the image first"
            return
        if self.zoom_center is None:
            cv2.namedWindow(self.ZOOMWIN, WINDOW_FLAGS)
            cv2.setMouseCallback(self.ZOOMWIN, self.on_zoom_mouse)
        self.zoom_center = self.cursor
        self.zoom_bounds = zoom_crop_bounds(*self.zoom_center, ZOOM_HALF, self.w, self.h)

    def close_zoom(self) -> None:
        if getattr(self, "zoom_center", None) is not None:
            cv2.destroyWindow(self.ZOOMWIN)
        self.zoom_center = None
        self.zoom_bounds = None

    # -- mouse

    def on_main_mouse(self, event, x, y, flags, param) -> None:
        p = display_to_original(x, y, self.disp, (self.w, self.h))
        if event == cv2.EVENT_MOUSEMOVE and p is not None:
            self.cursor = p
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.add(p)

    def on_zoom_mouse(self, event, x, y, flags, param) -> None:
        if self.zoom_bounds is None:
            return
        p = zoom_to_original(x, y, self.zoom_bounds, ZOOM)
        if event == cv2.EVENT_MOUSEMOVE and p is not None:
            self.cursor = p
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.add(p)

    # -- keys

    def handle_key(self, k: int) -> bool:
        """Return False to quit."""
        st = self.state
        if k in LABEL_KEYS:
            st.set_label(LABEL_KEYS[k])
        elif k == ord(" "):
            if not st.new_region():
                self.status = "current region is empty"
        elif k == ord("u"):
            self.status = "undone" if st.undo() else "nothing to undo"
        elif k == ord("s"):
            self.save()
        elif k in (ord("n"), ord("b")):
            step = 1 if k == ord("n") else -1
            target = self.idx + step
            self.save()
            if 0 <= target < len(self.images):
                self.load(target)
            else:
                self.status = "already at last image" if step > 0 else "already at first image"
        elif k == ord("q"):
            return False
        elif k == ord("z"):
            self.open_zoom()
        elif k in (ord("Z"), 27):
            self.close_zoom()
        elif k == ord("g"):
            self.gamma = not self.gamma
        elif ord("A") <= k <= ord("Z"):
            self.status = f"unmapped key '{chr(k)}' (Caps Lock on?)"
        return True

    # -- rendering

    def render(self) -> np.ndarray:
        dw, dh = self.disp
        canvas = np.zeros((dh + STRIP_H, max(dw, HUD_MIN_W), 3), np.uint8)
        canvas[:dh, :dw] = self.scaled_img()
        size = (self.w, self.h)
        draw_markers(canvas, self.state.points, lambda x, y: original_to_display(x, y, self.disp, size), r=4)

        st = self.state
        n_region = len(st.region_points())
        draw_marker(canvas, (12, dh + 15), MARKERS[st.label], region_color(st.region_id), 6)
        draw_text(
            canvas,
            f"{st.label}  region {st.region_id} ({n_region} pts)  total {len(st.points)}"
            f"   [{self.idx + 1}/{len(self.images)}] {self.images[self.idx].name}",
            (26, dh + 20),
        )
        cur = f"({self.cursor[0]},{self.cursor[1]})" if self.cursor else "-"
        zoom = "on" if self.zoom_center is not None else "off"
        draw_text(
            canvas,
            f"gamma {'on' if self.gamma else 'off'}  zoom {zoom}  cursor {cur}   {self.status}",
            (8, dh + 40),
        )
        draw_text(
            canvas,
            "m/t/p label  space new region  u undo  z zoom  g gamma  s save  n/b next/prev  q quit",
            (8, dh + 60), (180, 180, 180), 0.4,
        )
        return canvas

    def render_zoom(self) -> np.ndarray:
        x0, y0, x1, y1 = self.zoom_bounds
        crop = self.full_img()[y0:y1, x0:x1]
        big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
        draw_markers(big, self.state.points, lambda x, y: original_to_zoom(x, y, self.zoom_bounds, ZOOM), r=6)
        if self.cursor is not None:
            c = original_to_zoom(*self.cursor, self.zoom_bounds, ZOOM)
            if c is not None:
                a = (c[0] - ZOOM // 2, c[1] - ZOOM // 2)
                cv2.rectangle(big, a, (a[0] + ZOOM - 1, a[1] + ZOOM - 1), (255, 255, 255), 1)
        return big

    # -- main loop

    def run(self) -> None:
        cv2.namedWindow(self.MAIN, WINDOW_FLAGS)
        cv2.setMouseCallback(self.MAIN, self.on_main_mouse)
        try:
            while True:
                cv2.imshow(self.MAIN, self.render())
                if self.zoom_center is not None:
                    cv2.imshow(self.ZOOMWIN, self.render_zoom())
                k = cv2.waitKey(20) & 0xFF
                # check window close buttons before keys can open/close windows
                if cv2.getWindowProperty(self.MAIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if (self.zoom_center is not None
                        and cv2.getWindowProperty(self.ZOOMWIN, cv2.WND_PROP_VISIBLE) < 1):
                    self.zoom_center = self.zoom_bounds = None
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
    if w <= 0 or h <= STRIP_H:
        raise argparse.ArgumentTypeError(f"size too small: {s!r}")
    return w, h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--max-size", type=parse_size, default=(1600, 900), metavar="WxH",
                    help="max window size incl. HUD strip (default 1600x900)")
    args = ap.parse_args(argv)

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

    Annotator(images, args.out_dir, *args.max_size).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
