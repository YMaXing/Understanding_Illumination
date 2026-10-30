"""Spec tests 1-3: sRGB OETF, orientation round trip, network size."""

import numpy as np
import pytest
from PIL import Image, ImageOps

from illum_baseline.geometry import network_size
from illum_baseline.io_tiff import to_stored, to_upright, upright_size
from illum_baseline.preprocess import srgb_oetf


# 1. sRGB OETF
def test_oetf_anchor_points():
    assert srgb_oetf(0.0) == 0.0
    assert srgb_oetf(0.0031308) == pytest.approx(0.04045, abs=1e-6)
    assert srgb_oetf(1.0) == pytest.approx(1.0, abs=1e-12)


def test_oetf_monotone_and_continuous():
    x = np.linspace(0, 1, 200001)
    y = srgb_oetf(x)
    assert np.all(np.diff(y) > 0)
    k = 0.0031308
    assert 12.92 * k == pytest.approx(1.055 * k ** (1 / 2.4) - 0.055, abs=1e-6)


# 2. Orientation
@pytest.mark.parametrize("tag", range(1, 9))
@pytest.mark.parametrize("shape", [(5, 7), (5, 7, 3)])
def test_orientation_round_trip(tag, shape):
    x = np.random.default_rng(tag).integers(0, 1000, shape)
    up = to_upright(x, tag)
    assert up.shape[:2] == (shape[1], shape[0]) if tag >= 5 else up.shape[:2] == shape[:2]
    assert upright_size(shape[1], shape[0], tag) == (up.shape[1], up.shape[0])
    np.testing.assert_array_equal(to_stored(up, tag), x)


@pytest.mark.parametrize("tag", range(1, 9))
def test_orientation_matches_pil_exif_transpose(tag):
    """Independent reference: PIL applies the same TIFF/EXIF convention to display images upright."""
    x = np.random.default_rng(100 + tag).integers(0, 256, (5, 7, 3), dtype=np.uint8)
    img = Image.fromarray(x)
    exif = img.getexif()
    exif[0x0112] = tag
    img.info["exif"] = exif.tobytes()
    ref = np.asarray(ImageOps.exif_transpose(img))
    np.testing.assert_array_equal(to_upright(x, tag), ref)


def test_orientation_tag6_is_clockwise_rotation():
    # Tag 6: the stored row 0 is the upright right-hand column, read top to bottom.
    x = np.arange(6).reshape(2, 3)  # [[0,1,2],[3,4,5]]
    np.testing.assert_array_equal(to_upright(x, 6), [[3, 0], [4, 1], [5, 2]])


# 3. Network size
@pytest.mark.parametrize("wh, expected", [((1023, 768), (504, 378)), ((768, 1023), (378, 504))])
def test_network_size(wh, expected):
    wn, hn = network_size(*wh)
    assert (wn, hn) == expected
    assert wn % 14 == 0 and hn % 14 == 0


def test_focal_at_network_uses_both_scale_factors():
    from illum_baseline.geometry import focal_at_network, scale_factors

    s_x, s_y = scale_factors(1023, 768, *network_size(1023, 768))
    assert s_x != s_y  # 504/1023 vs 378/768, about 0.1% apart
    assert focal_at_network(700.0, 700.0, s_x, s_y) == pytest.approx(700.0 * (s_x + s_y) / 2)
    assert focal_at_network(700.0, 710.0, s_x, s_y) == pytest.approx((700.0 * s_x + 710.0 * s_y) / 2)
