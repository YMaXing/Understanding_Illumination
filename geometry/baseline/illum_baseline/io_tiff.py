"""TIFF loading, tag reading, and Orientation (tag 274) frame transforms.

Frames:
  stored  -- the pixel array as stored in the TIFF (row 0 = first stored row).
  upright -- the array after applying the Orientation tag, i.e. as a viewer displays it.

Orientation convention (TIFF 6.0 / EXIF tag 274), matching PIL.ImageOps.exif_transpose:
  1  identity                       5  transpose (swap rows/cols)
  2  mirror left-right              6  rotate 90 deg clockwise
  3  rotate 180                     7  transverse (transpose about the anti-diagonal)
  4  mirror top-bottom              8  rotate 90 deg counter-clockwise
`to_upright` applies the listed operation; `to_stored` is its exact inverse. Both act on
the first two axes, so (H, W) and (H, W, C) arrays are handled alike.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tifffile

ORIENTATION_TAG = 274

# Tags the audit reports by name. EXIF/DNG tags may sit in the main IFD or the EXIF sub-IFD.
NAMED_TAGS = (
    "ImageWidth", "ImageLength", "BitsPerSample", "SamplesPerPixel", "PhotometricInterpretation",
    "Orientation", "Make", "Model", "FocalLength", "FocalLengthIn35mmFilm", "Compression",
    "SampleFormat", "PlanarConfiguration", "CFAPattern", "CFARepeatPatternDim", "CFAPlaneColor",
    "BlackLevel", "BlackLevelRepeatDim", "WhiteLevel", "AsShotNeutral", "AsShotWhiteXY",
    "ColorMatrix1", "BaselineExposure", "DNGVersion", "ExposureTime", "FNumber",
    "ISOSpeedRatings", "Software", "ImageDescription",
)
# Per-strip/tile layout tags: large and uninformative, summarized by length only.
_LAYOUT_TAGS = {"StripOffsets", "StripByteCounts", "TileOffsets", "TileByteCounts"}


def _jsonable(v: Any) -> Any:
    if hasattr(v, "name") and hasattr(v, "value") and not isinstance(v, (str, bytes)):
        return v.name  # tifffile enums
    if isinstance(v, bytes):
        try:
            return v.decode("ascii").rstrip("\x00")
        except UnicodeDecodeError:
            return v.hex()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, np.generic):
        return v.item()
    return v


def read_tags(page: tifffile.TiffPage) -> dict[str, Any]:
    """All tags of a page as a JSON-able dict keyed by tag name, EXIF sub-IFD merged in."""
    tags: dict[str, Any] = {}
    for t in page.tags.values():
        name = t.name if t.name and not t.name.isdigit() else f"Tag{t.code}"
        if name in _LAYOUT_TAGS:
            tags[name] = f"<{len(t.value)} entries>"
        elif name == "ExifTag" and isinstance(t.value, dict):
            for k, v in t.value.items():
                tags.setdefault(str(k), _jsonable(v))
        else:
            tags[name] = _jsonable(t.value)
    return tags


def load_tiff(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (array in the stored frame, tags). Never writes to the file."""
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        arr = page.asarray()
        tags = read_tags(page)
        tags["_n_pages"] = len(tf.pages)
        tags["_n_subifds"] = len(page.subifds or ())
    return arr, tags


def orientation_of(tags: dict[str, Any]) -> int:
    """Orientation tag value; 1 if absent. Raises on values outside 1..8."""
    v = tags.get("Orientation", 1)
    v = {"TOPLEFT": 1, "TOPRIGHT": 2, "BOTRIGHT": 3, "BOTLEFT": 4,
         "LEFTTOP": 5, "RIGHTTOP": 6, "RIGHTBOT": 7, "LEFTBOT": 8}.get(v, v)
    v = int(v)
    if not 1 <= v <= 8:
        raise ValueError(f"invalid Orientation tag {v}")
    return v


def _t(x: np.ndarray) -> np.ndarray:
    return np.swapaxes(x, 0, 1)


_UPRIGHT = {
    1: lambda x: x,
    2: lambda x: x[:, ::-1],
    3: lambda x: x[::-1, ::-1],
    4: lambda x: x[::-1],
    5: _t,
    6: lambda x: np.rot90(x, k=-1),
    7: lambda x: _t(x[::-1, ::-1]),
    8: lambda x: np.rot90(x, k=1),
}
_STORED = {
    **{k: _UPRIGHT[k] for k in (1, 2, 3, 4, 5, 7)},  # involutions
    6: lambda x: np.rot90(x, k=1),
    8: lambda x: np.rot90(x, k=-1),
}


def to_upright(x: np.ndarray, orientation: int) -> np.ndarray:
    return np.ascontiguousarray(_UPRIGHT[orientation](x))


def to_stored(x: np.ndarray, orientation: int) -> np.ndarray:
    return np.ascontiguousarray(_STORED[orientation](x))


def upright_size(w_stored: int, h_stored: int, orientation: int) -> tuple[int, int]:
    """(W, H) in the upright frame."""
    return (h_stored, w_stored) if orientation >= 5 else (w_stored, h_stored)
