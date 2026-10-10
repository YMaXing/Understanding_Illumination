import numpy as np
import pytest

from annotate import (
    CSV_HEADER,
    STRIP_H,
    ZOOM,
    ZOOM_HALF,
    AnnotationState,
    Point,
    display_to_original,
    fit_display_size,
    original_to_display,
    original_to_zoom,
    read_csv,
    region_first_points,
    renumber_regions,
    state_from_points,
    stretch_params,
    to_display_8bit,
    write_csv,
    zoom_crop_bounds,
    zoom_to_original,
)

W, H = 768, 1023  # size of the project's source TIFFs


# ---------------------------------------------------------------- stretch


def _ordered_image():
    rng = np.random.default_rng(0)
    base = rng.integers(100, 20000, size=(16, 16)).astype(np.float64)
    img = np.stack([base, base * 1.5, base * 2.0], axis=-1)  # B < G < R
    img[0, :4] = 30000  # a few gray pixels
    return img.astype(np.uint16)


@pytest.mark.parametrize("gamma", [True, False])
def test_joint_stretch_preserves_channel_ordering(gamma):
    img = _ordered_image()
    out = to_display_8bit(img, *stretch_params(img), gamma).astype(int)
    b, g, r = out[..., 0], out[..., 1], out[..., 2]
    assert np.all(b <= g) and np.all(g <= r)
    # strictly ordered somewhere (a per-channel stretch would not keep this structure)
    assert np.any((b < g) & (g < r))


def test_joint_stretch_keeps_gray_gray():
    img = _ordered_image()
    out = to_display_8bit(img, *stretch_params(img), True)
    gray = out[0, :4]
    assert np.all(gray[:, 0] == gray[:, 1]) and np.all(gray[:, 1] == gray[:, 2])


def test_stretch_params_is_joint_scalar_pair():
    img = np.zeros((10, 10, 3), np.uint16)
    img[..., 0] = 100
    img[..., 2] = 50000
    lo, hi = stretch_params(img)
    assert isinstance(lo, float) and isinstance(hi, float)
    assert lo == pytest.approx(np.percentile(img, 0.5))
    assert hi == pytest.approx(np.percentile(img, 99.5))


def test_constant_image_does_not_divide_by_zero():
    img = np.full((8, 8, 3), 1234, np.uint16)
    lo, hi = stretch_params(img)
    assert hi > lo
    out = to_display_8bit(img, lo, hi, True)
    assert out.dtype == np.uint8 and np.all(np.isfinite(out))


@pytest.mark.parametrize("dtype,shape", [(np.uint16, (5, 7, 3)), (np.uint8, (5, 7, 3)), (np.uint16, (5, 7))])
def test_display_shape_and_dtype(dtype, shape):
    img = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    out = to_display_8bit(img, *stretch_params(img), False)
    assert out.shape == shape and out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_gamma_brightens_midtones():
    img = np.linspace(0, 65535, 300, dtype=np.uint16).reshape(10, 10, 3)
    lo, hi = stretch_params(img)
    assert to_display_8bit(img, lo, hi, True).mean() > to_display_8bit(img, lo, hi, False).mean()


# ---------------------------------------------------------------- main-view mapping


def test_fit_never_upscales():
    assert fit_display_size(100, 80, 1600, 900) == (100, 80)


def test_fit_accounts_for_strip():
    dw, dh = fit_display_size(W, H, 800, 600)
    assert dh + STRIP_H <= 600 and dw <= 800
    assert dh == round(H * (600 - STRIP_H) / H)


def test_strip_and_outside_clicks_are_ignored():
    disp = fit_display_size(W, H, 800, 600)
    dw, dh = disp
    for dy in (dh, dh + 1, dh + STRIP_H - 1):
        assert display_to_original(10, dy, disp, (W, H)) is None
    assert display_to_original(dw, 10, disp, (W, H)) is None
    assert display_to_original(-1, 10, disp, (W, H)) is None
    assert display_to_original(0, 0, disp, (W, H)) == (0, 0)
    x, y = display_to_original(dw - 1, dh - 1, disp, (W, H))
    assert W - 1 - W / dw <= x <= W - 1 and H - 1 - H / dh <= y <= H - 1


@pytest.mark.parametrize("max_size", [(800, 600), (1600, 1200)])
def test_mapping_round_trip(max_size):
    disp = fit_display_size(W, H, *max_size)
    tol_x, tol_y = W / disp[0], H / disp[1]
    for x, y in [(0, 0), (W - 1, H - 1), (123, 456), (767, 0), (0, 1022), (400, 999)]:
        dx, dy = original_to_display(x, y, disp, (W, H))
        bx, by = display_to_original(dx, dy, disp, (W, H))
        assert abs(bx - x) <= tol_x and abs(by - y) <= tol_y


def test_identity_mapping_at_native_scale():
    disp = fit_display_size(W, H, 1600, 1200)
    assert disp == (W, H)
    for x, y in [(0, 0), (5, 9), (W - 1, H - 1)]:
        assert display_to_original(*original_to_display(x, y, disp, (W, H)), disp, (W, H)) == (x, y)


# ---------------------------------------------------------------- zoom


def test_zoom_bounds_around_center():
    assert zoom_crop_bounds(300, 400, ZOOM_HALF, W, H) == (236, 336, 364, 464)


def test_zoom_frozen_center_mapping():
    # center frozen at (300, 400); clicks map through that crop only
    b = zoom_crop_bounds(300, 400, ZOOM_HALF, W, H)
    assert zoom_to_original(ZOOM_HALF * ZOOM, ZOOM_HALF * ZOOM, b, ZOOM) == (300, 400)
    assert zoom_to_original(0, 0, b, ZOOM) == (236, 336)
    assert zoom_to_original(ZOOM - 1, ZOOM - 1, b, ZOOM) == (236, 336)  # same pixel block
    assert zoom_to_original(ZOOM, 0, b, ZOOM) == (237, 336)
    last = 2 * ZOOM_HALF * ZOOM - 1
    assert zoom_to_original(last, last, b, ZOOM) == (363, 463)
    assert zoom_to_original(last + 1, 0, b, ZOOM) is None
    assert zoom_to_original(-1, 0, b, ZOOM) is None


def test_zoom_clamped_near_border():
    b = zoom_crop_bounds(5, H - 3, ZOOM_HALF, W, H)
    x0, y0, x1, y1 = b
    assert (x0, x1) == (0, 2 * ZOOM_HALF)
    assert (y0, y1) == (H - 2 * ZOOM_HALF, H)
    # the frozen center is still addressable, and every window pixel maps inside the image
    zc = original_to_zoom(5, H - 3, b, ZOOM)
    assert zoom_to_original(*zc, b, ZOOM) == (5, H - 3)
    last = 2 * ZOOM_HALF * ZOOM - 1
    for zx, zy in [(0, 0), (last, last), (0, last), (last, 0)]:
        x, y = zoom_to_original(zx, zy, b, ZOOM)
        assert 0 <= x < W and 0 <= y < H


def test_zoom_crop_larger_than_image():
    assert zoom_crop_bounds(10, 10, ZOOM_HALF, 50, 40) == (0, 0, 50, 40)


def test_original_to_zoom_round_trip_and_outside():
    b = zoom_crop_bounds(300, 400, ZOOM_HALF, W, H)
    for x, y in [(236, 336), (363, 463), (300, 400), (250, 450)]:
        assert zoom_to_original(*original_to_zoom(x, y, b, ZOOM), b, ZOOM) == (x, y)
    assert original_to_zoom(235, 400, b, ZOOM) is None
    assert original_to_zoom(364, 400, b, ZOOM) is None


# ---------------------------------------------------------------- state


def test_duplicate_pixel_rejected_in_same_region():
    st = AnnotationState()
    assert st.add_point(10, 20)
    assert not st.add_point(10, 20)
    assert len(st.points) == 1


def test_duplicate_pixel_rejected_across_regions():
    st = AnnotationState()
    st.add_point(10, 20)
    st.new_region()
    assert not st.add_point(10, 20)
    st.set_label("specular")
    assert not st.add_point(10, 20)
    assert st.add_point(10, 21)


def test_pixel_reusable_after_undo():
    st = AnnotationState()
    st.add_point(10, 20)
    st.undo()
    assert st.add_point(10, 20)


def test_space_noop_on_empty_region():
    st = AnnotationState()
    assert not st.new_region()
    assert st.region_id == 1
    st.add_point(1, 1)
    assert st.new_region() and st.region_id == 2
    assert not st.new_region() and st.region_id == 2


def test_label_switch_rules():
    st = AnnotationState()
    st.set_label("texture")  # empty region: relabel only
    assert (st.region_id, st.label) == (1, "texture")
    st.add_point(1, 1)
    st.set_label("texture")  # same label: nothing
    assert st.region_id == 1
    st.set_label("specular")  # has points: new region
    assert (st.region_id, st.label) == (2, "specular")
    with pytest.raises(ValueError):
        st.set_label("bogus")


def test_global_undo_across_region_boundary():
    st = AnnotationState()
    st.add_point(1, 1)
    st.add_point(2, 2)
    st.set_label("texture")
    st.add_point(3, 3)
    assert (st.region_id, st.label) == (2, "texture")
    st.undo()  # region 2 now empty -> back to region 1
    assert (st.region_id, st.label) == (1, "material")
    st.undo()
    assert [(p.x, p.y) for p in st.points] == [(1, 1)]
    assert (st.region_id, st.label) == (1, "material")


def test_undo_from_fresh_empty_region_removes_previous_point():
    st = AnnotationState()
    st.add_point(1, 1)
    st.add_point(2, 2)
    st.new_region()
    st.set_label("specular")
    st.undo()
    assert [(p.x, p.y) for p in st.points] == [(1, 1)]
    assert (st.region_id, st.label) == (1, "material")


def test_undo_on_empty_state():
    st = AnnotationState()
    assert not st.undo()
    assert (st.region_id, st.label, st.points) == (1, "material", [])


def test_undo_last_point_keeps_its_region():
    st = AnnotationState()
    st.set_label("texture")
    st.add_point(1, 1)
    st.undo()
    assert (st.region_id, st.label, st.points) == (1, "texture", [])


def test_region_first_points():
    pts = [Point(5, 5, 2, "texture"), Point(1, 1, 1, "material"), Point(6, 6, 2, "texture")]
    assert region_first_points(pts) == {2: (5, 5), 1: (1, 1)}


# ---------------------------------------------------------------- CSV


def test_csv_round_trip(tmp_path):
    pts = [Point(1, 2, 1, "material"), Point(3, 4, 1, "material"), Point(5, 6, 2, "specular")]
    path = tmp_path / "img.csv"
    write_csv(path, "img.tif", pts)
    assert read_csv(path) == pts
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(CSV_HEADER)
    assert lines[1] == "img.tif,1,2,1,material"
    assert not list(tmp_path.glob("*.tmp"))


def test_csv_header_only_for_empty(tmp_path):
    path = tmp_path / "img.csv"
    write_csv(path, "img.tif", [])
    assert path.read_text().splitlines() == [",".join(CSV_HEADER)]
    assert read_csv(path) == []


def test_csv_write_renumbers_regions(tmp_path):
    pts = [Point(0, 0, 3, "texture"), Point(1, 1, 7, "material"), Point(2, 2, 3, "texture")]
    path = tmp_path / "img.csv"
    write_csv(path, "img.tif", pts)
    assert [p.region_id for p in read_csv(path)] == [1, 2, 1]


def test_renumber_regions():
    pts = [Point(0, 0, 1, "material"), Point(1, 1, 3, "texture"), Point(2, 2, 7, "specular")]
    assert [p.region_id for p in renumber_regions(pts)] == [1, 2, 3]


@pytest.mark.parametrize(
    "row",
    ["img.tif,1,2,1,metal", "img.tif,1.5,2,1,material", "img.tif,1,2,0,material", "img.tif,1,2,1"],
)
def test_csv_malformed_raises(tmp_path, row):
    path = tmp_path / "bad.csv"
    path.write_text(",".join(CSV_HEADER) + "\n" + row + "\n")
    with pytest.raises(ValueError, match="bad.csv:2"):
        read_csv(path)


def test_csv_bad_header_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x,y\n1,2\n")
    with pytest.raises(ValueError, match="header"):
        read_csv(path)


def test_state_from_points():
    st = state_from_points([])
    assert (st.region_id, st.label, st.points) == (1, "material", [])
    pts = [Point(0, 0, 1, "material"), Point(1, 1, 2, "specular")]
    st = state_from_points(pts)
    assert (st.region_id, st.label) == (3, "specular")
    assert st.region_points() == []
    assert not st.add_point(1, 1)  # loaded pixels count as used
