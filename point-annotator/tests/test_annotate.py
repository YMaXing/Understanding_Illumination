import numpy as np
import pytest

from annotate import (
    CSV_HEADER,
    PANEL_BOTTOM_H,
    PANEL_RIGHT_W,
    ZOOM,
    ZOOM_HALF,
    AnnotationState,
    Point,
    display_to_original,
    fit_display_size,
    original_to_display,
    nearest_point,
    original_to_zoom,
    read_csv,
    region_first_points,
    renumber_regions,
    rows_for_save,
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


@pytest.mark.parametrize("gamma", [2.2, None, 4.0])
def test_joint_stretch_preserves_channel_ordering(gamma):
    img = _ordered_image()
    out = to_display_8bit(img, *stretch_params(img), gamma).astype(int)
    b, g, r = out[..., 0], out[..., 1], out[..., 2]
    assert np.all(b <= g) and np.all(g <= r)
    # strictly ordered somewhere (a per-channel stretch would not keep this structure)
    assert np.any((b < g) & (g < r))


def test_joint_stretch_keeps_gray_gray():
    img = _ordered_image()
    out = to_display_8bit(img, *stretch_params(img), 2.2)
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
    out = to_display_8bit(img, lo, hi, 2.2)
    assert out.dtype == np.uint8 and np.all(np.isfinite(out))


@pytest.mark.parametrize("dtype,shape", [(np.uint16, (5, 7, 3)), (np.uint8, (5, 7, 3)), (np.uint16, (5, 7))])
def test_display_shape_and_dtype(dtype, shape):
    img = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    out = to_display_8bit(img, *stretch_params(img), None)
    assert out.shape == shape and out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_gamma_brightens_midtones():
    img = np.linspace(0, 65535, 300, dtype=np.uint16).reshape(10, 10, 3)
    lo, hi = stretch_params(img)
    assert to_display_8bit(img, lo, hi, 2.2).mean() > to_display_8bit(img, lo, hi, None).mean()


# ---------------------------------------------------------------- main-view mapping


def test_fit_never_upscales():
    assert fit_display_size(100, 80, 1600, 900) == (100, 80)


def test_fit_accounts_for_bottom_panel():
    dw, dh = fit_display_size(W, H, 800, 600, "bottom")
    assert dh + PANEL_BOTTOM_H <= 600 and dw <= 800
    assert dh == round(H * (600 - PANEL_BOTTOM_H) / H)


def test_bottom_panel_and_outside_clicks_are_ignored():
    disp = fit_display_size(W, H, 800, 600, "bottom")
    dw, dh = disp
    for dy in (dh, dh + 1, dh + PANEL_BOTTOM_H - 1):
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
    disp = fit_display_size(W, H, 2000, 1400)
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


def test_undo_last_point_resets_to_region_1_material():
    st = AnnotationState()
    st.set_label("texture")
    st.add_point(1, 1)
    st.undo()
    assert (st.region_id, st.label, st.points) == (1, "material", [])


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
    assert lines[1] == "img.tif,1,2,1,material,0"
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
    # sorted by region, insertion order kept within a region, ids renumbered
    assert [(p.x, p.region_id) for p in read_csv(path)] == [(0, 1), (2, 1), (1, 2)]


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


# ---------------------------------------------------------------- region selection


def _ident(x, y):
    return x, y


def test_nearest_point_hit_and_miss():
    pts = [Point(100, 100, 1, "material"), Point(200, 200, 2, "texture")]
    assert nearest_point(pts, (100, 100), _ident, 10) == pts[0]
    assert nearest_point(pts, (106, 108), _ident, 10) == pts[0]  # distance 10: hit
    assert nearest_point(pts, (111, 100), _ident, 10) is None  # distance 11: miss
    assert nearest_point(pts, (195, 198), _ident, 10) == pts[1]
    assert nearest_point([], (0, 0), _ident, 10) is None


def test_nearest_point_tie_goes_to_most_recent():
    pts = [Point(90, 100, 1, "material"), Point(110, 100, 2, "texture")]
    assert nearest_point(pts, (100, 100), _ident, 10) == pts[1]
    pts.reverse()
    assert nearest_point(pts, (100, 100), _ident, 10) == pts[1]


def test_nearest_point_tie_at_same_display_pixel():
    disp = fit_display_size(W, H, 800, 600)  # ~0.52 scale: neighbors share a display pixel
    def to_canvas(x, y):
        return original_to_display(x, y, disp, (W, H))
    x = next(x for x in range(300, 310) if to_canvas(x, 400) == to_canvas(x + 1, 400))
    pts = [Point(x, 400, 1, "material"), Point(x + 1, 400, 2, "specular")]
    assert nearest_point(pts, to_canvas(x, 400), to_canvas, 10) == pts[1]


def test_nearest_point_through_display_mapping():
    disp = fit_display_size(W, H, 800, 600)
    pts = [Point(100, 100, 1, "material"), Point(140, 100, 2, "texture")]  # ~21 display px apart
    def to_canvas(x, y):
        return original_to_display(x, y, disp, (W, H))
    cx, cy = to_canvas(140, 100)
    assert nearest_point(pts, (cx - 3, cy + 2), to_canvas, 10) == pts[1]
    assert nearest_point(pts, (cx + 15, cy), to_canvas, 10) is None


def test_nearest_point_skips_undrawn_points():
    b = zoom_crop_bounds(300, 400, ZOOM_HALF, W, H)
    pts = [Point(300, 400, 1, "material"), Point(10, 10, 2, "texture")]  # second is outside the crop
    def to_canvas(x, y):
        return original_to_zoom(x, y, b, ZOOM)
    assert nearest_point(pts, (0, 0), to_canvas, 10_000) == pts[0]


def _three_regions():
    st = AnnotationState()
    st.add_point(1, 1)             # region 1 material
    st.set_label("texture")
    st.add_point(2, 2)             # region 2 texture
    st.set_label("specular")
    st.add_point(3, 3)             # region 3 specular
    return st


def test_select_region_takes_its_label():
    st = _three_regions()
    assert st.select_region(2)
    assert (st.region_id, st.label, st.is_existing()) == (2, "texture", True)
    assert not st.select_region(9)
    assert (st.region_id, st.label) == (2, "texture")


def test_cycle_regions_wraps_through_new():
    st = _three_regions()
    st.new_region()
    assert st.region_id == 4 and not st.is_existing()
    assert st.cycle_region(1) == 1 and st.label == "material"
    assert st.cycle_region(1) == 2 and st.label == "texture"
    assert st.cycle_region(1) == 3
    assert st.cycle_region(1) == 4 and not st.is_existing()
    assert st.label == "specular"  # (new) keeps the current label
    assert st.cycle_region(-1) == 3  # [ from (new) -> k
    assert st.cycle_region(-1) == 2
    assert st.cycle_region(-1) == 1
    assert st.cycle_region(-1) == 4  # wrap to (new)


def test_cycle_with_no_points_stays_on_region_1():
    st = AnnotationState()
    assert st.cycle_region(1) == 1 and st.cycle_region(-1) == 1


def test_space_after_selecting_low_region_gives_max_plus_1():
    st = _three_regions()
    st.select_region(1)
    assert st.new_region()
    assert st.region_id == 4
    st.add_point(4, 4)
    assert st.region_ids() == [1, 2, 3, 4]


def test_label_key_after_selecting_existing_region():
    st = _three_regions()
    st.select_region(1)
    st.set_label("material")  # same label: stay
    assert (st.region_id, st.label) == (1, "material")
    st.set_label("texture")  # different label: open max+1, never relabel region 1
    assert (st.region_id, st.label, st.is_existing()) == (4, "texture", False)
    assert {p.label for p in st.region_points(1)} == {"material"}
    st.set_label("specular")  # new empty region: just relabel it
    assert (st.region_id, st.label) == (4, "specular")


def test_add_to_reselected_region_then_undo():
    st = _three_regions()
    st.select_region(1)
    st.add_point(5, 5)
    assert [p.region_id for p in st.points] == [1, 2, 3, 1]
    st.undo()  # last remaining point is region 3's
    assert (st.region_id, st.label) == (3, "specular")
    st.undo()
    assert (st.region_id, st.label) == (2, "texture")


def test_csv_sorted_and_round_trip(tmp_path):
    st = _three_regions()
    st.select_region(1)
    st.add_point(5, 5)
    st.select_region(2)
    st.add_point(6, 6)
    path = tmp_path / "img.csv"
    write_csv(path, "img.tif", st.points)
    got = read_csv(path)
    assert [(p.x, p.region_id) for p in got] == [(1, 1), (5, 1), (2, 2), (6, 2), (3, 3)]
    first = path.read_bytes()
    write_csv(path, "img.tif", got)
    assert path.read_bytes() == first


def test_rows_for_save_renumbering():
    pts = [Point(0, 0, 6, "texture"), Point(1, 1, 2, "material")]
    assert [(p.x, p.region_id) for p in rows_for_save(pts)] == [(1, 1), (0, 2)]
    pts = [Point(0, 0, 1, "material"), Point(1, 1, 6, "texture"), Point(2, 2, 2, "specular")]
    assert [(p.x, p.region_id) for p in rows_for_save(pts)] == [(0, 1), (2, 2), (1, 3)]  # 2 survives


# ---------------------------------------------------------------- delete + action undo


def test_delete_target_hit_miss_tie_display_mapping():
    disp = fit_display_size(W, H, 800, 600)
    def to_canvas(x, y):
        return original_to_display(x, y, disp, (W, H))
    pts = [Point(100, 100, 1, "material"), Point(140, 100, 2, "texture")]
    c0 = to_canvas(100, 100)
    assert nearest_point(pts, (c0[0] + 6, c0[1] + 8), to_canvas, 10) == pts[0]  # hit at 10
    assert nearest_point(pts, (c0[0] - 11, c0[1]), to_canvas, 10) is None  # miss at 11
    # nearest wins even when an older dot is also in range
    c1 = to_canvas(140, 100)
    mid = (c0[0] + c1[0]) // 2
    near0 = (mid - 2, c0[1])
    assert nearest_point(pts, near0, to_canvas, 15) == pts[0]
    # exact tie -> most recently added
    tied = [Point(100, 100, 1, "material"), Point(100, 100 + 2 * 3, 2, "texture")]
    def ident(x, y):
        return x, y
    assert nearest_point(tied, (100, 103), ident, 10) == tied[1]


def test_delete_target_through_zoom_mapping():
    b = zoom_crop_bounds(300, 400, ZOOM_HALF, W, H)
    def to_canvas(x, y):
        return original_to_zoom(x, y, b, ZOOM)
    pts = [Point(300, 400, 1, "material"), Point(302, 400, 2, "texture")]  # 8 zoom px apart
    c0, c1 = to_canvas(300, 400), to_canvas(302, 400)
    assert nearest_point(pts, (c0[0] + 1, c0[1]), to_canvas, 10) == pts[0]
    assert nearest_point(pts, ((c0[0] + c1[0]) // 2, c0[1]), to_canvas, 10) == pts[1]  # tie
    assert nearest_point(pts, (c1[0] + 11, c1[1]), to_canvas, 10) is None


def test_delete_from_non_current_region():
    st = _three_regions()
    st.add_point(4, 4)  # region 3 has 2 points; current = 3 specular
    assert st.delete_point(2, 2) == Point(2, 2, 2, "texture")
    assert (st.region_id, st.label) == (3, "specular")
    assert [(p.x, p.region_id) for p in st.points] == [(1, 1), (3, 3), (4, 3)]
    assert not st.is_used(2, 2)


def test_delete_only_point_of_current_region_keeps_it_current():
    st = _three_regions()
    st.select_region(2)
    st.delete_point(2, 2)
    assert (st.region_id, st.label, st.is_existing()) == (2, "texture", False)
    assert not st.new_region()  # space on empty region: no-op
    assert st.region_id == 2
    st.add_point(7, 7)  # refills the same region with its label
    assert st.region_points(2) == [Point(7, 7, 2, "texture")]


def test_delete_only_point_of_current_region_then_label_key_relabels():
    st = _three_regions()
    st.select_region(2)
    st.delete_point(2, 2)
    st.set_label("material")
    assert (st.region_id, st.label) == (2, "material")


def test_delete_only_point_of_non_current_region():
    st = _three_regions()
    st.select_region(1)
    st.delete_point(3, 3)  # empties region 3 (the max)
    assert (st.region_id, st.label) == (1, "material")
    assert st.region_ids() == [1, 2]
    assert st.new_region() and st.region_id == 3  # new max+1


def test_delete_missing_pixel_returns_none():
    st = _three_regions()
    n = len(st.history)
    assert st.delete_point(99, 99) is None
    assert len(st.history) == n


def test_undo_restores_deleted_point_and_prior_region():
    st = _three_regions()
    st.select_region(1)
    st.delete_point(2, 2)  # current was region 1 / material at delete time
    st.select_region(3)  # move elsewhere before undoing
    assert st.add_point(2, 2)  # pixel is free after delete...
    assert st.undo() == "add"
    assert st.undo() == "delete"
    assert st.points[1] == Point(2, 2, 2, "texture")  # original index, region, label
    assert (st.region_id, st.label) == (1, "material")  # pre-delete current region
    assert not st.add_point(2, 2)  # ...and taken again after undo


def test_undo_restores_emptied_region():
    st = _three_regions()
    st.delete_point(3, 3)  # only point of current region 3
    st.select_region(1)
    st.undo()
    assert st.region_points(3) == [Point(3, 3, 3, "specular")]
    assert (st.region_id, st.label) == (3, "specular")


def test_undo_after_mixed_add_and_delete():
    st = AnnotationState()
    st.add_point(1, 1)  # r1 material
    st.add_point(2, 2)  # r1 material
    st.set_label("texture")
    st.add_point(3, 3)  # r2 texture
    st.select_region(1)
    st.delete_point(2, 2)
    st.add_point(4, 4)  # r1 material
    assert [(p.x, p.region_id) for p in st.points] == [(1, 1), (3, 2), (4, 1)]

    assert st.undo() == "add"  # remove (4,4); follow new last point (3,3)
    assert [(p.x, p.y) for p in st.points] == [(1, 1), (3, 3)]
    assert (st.region_id, st.label) == (2, "texture")
    assert st.undo() == "delete"  # restore (2,2) at index 1; region 1 was current
    assert [(p.x, p.y) for p in st.points] == [(1, 1), (2, 2), (3, 3)]
    assert (st.region_id, st.label) == (1, "material")
    assert st.undo() == "add"  # remove (3,3)
    assert (st.region_id, st.label) == (1, "material")
    assert st.undo() == "add" and st.undo() == "add"
    assert st.points == [] and (st.region_id, st.label) == (1, "material")
    assert st.undo() is None


def test_reloaded_history_undoes_last_row_first():
    pts = [Point(0, 0, 1, "material"), Point(1, 1, 2, "specular"), Point(2, 2, 2, "specular")]
    st = state_from_points(pts)
    assert st.undo() == "add"
    assert [(p.x, p.y) for p in st.points] == [(0, 0), (1, 1)]
    assert (st.region_id, st.label) == (2, "specular")
    assert AnnotationState().undo() is None


def test_csv_contiguous_after_deleting_whole_region(tmp_path):
    st = _three_regions()
    st.select_region(2)
    st.add_point(5, 5)
    st.delete_point(2, 2)
    st.delete_point(5, 5)  # region 2 is gone
    path = tmp_path / "img.csv"
    write_csv(path, "img.tif", st.points)
    got = read_csv(path)
    assert [(p.x, p.region_id, p.label) for p in got] == [(1, 1, "material"), (3, 2, "specular")]
    # reloaded region-id text is drawn from the renumbered ids
    assert region_first_points(state_from_points(got).points) == {1: (1, 1), 2: (3, 3)}
    first = path.read_bytes()
    write_csv(path, "img.tif", got)
    assert path.read_bytes() == first


# ---------------------------------------------------------------- palette / colors

import itertools  # noqa: E402

import cv2  # noqa: E402

from annotate import (  # noqa: E402
    BOTTOM_MIN_W,
    GAMMA_DEFAULT,
    GOLDEN_ANGLE,
    MSG_SECONDS,
    PALETTE_20,
    Annotator,
    color_for_region,
    compute_scale,
    gamma_lut,
    palette_preview,
    panel_side,
    step_gamma,
    text_width,
    wrap_text,
)

ORIGINAL_8 = [
    (75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0),
    (48, 130, 245), (180, 30, 145), (240, 240, 70), (230, 50, 240),
]


def _lab(bgr):
    px = np.array(bgr, np.float32).reshape(1, 1, 3) / 255
    return cv2.cvtColor(px, cv2.COLOR_BGR2Lab).reshape(3).astype(float)


def test_palette_20_keeps_first_8_and_is_distinct():
    assert len(PALETTE_20) == 20
    assert PALETTE_20[:8] == ORIGINAL_8
    assert len(set(PALETTE_20)) == 20
    assert all(all(isinstance(v, int) and 0 <= v <= 255 for v in c) for c in PALETTE_20)


def test_palette_new_colors_are_bright():
    for c in PALETTE_20[8:]:
        assert _lab(c)[0] >= 45, c


def test_palette_min_pairwise_cielab_distance():
    labs = [_lab(c) for c in PALETTE_20]
    d = min(np.linalg.norm(a - b) for a, b in itertools.combinations(labs, 2))
    assert d >= 25


def test_color_for_region_palette_and_determinism():
    for rid in range(1, 21):
        assert color_for_region(rid) == PALETTE_20[rid - 1]
    first = [color_for_region(r) for r in range(1, 200)]
    assert first == [color_for_region(r) for r in range(1, 200)]  # stable across calls


def test_color_for_region_beyond_palette_golden_angle():
    import colorsys
    for rid in (21, 22, 57, 1000):
        b, g, r = color_for_region(rid)
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        assert v == pytest.approx(1.0, abs=1 / 255)
        assert s == pytest.approx(0.85, abs=0.01)
        expect = (rid * GOLDEN_ANGLE) % 360
        assert min(abs(h * 360 - expect), 360 - abs(h * 360 - expect)) < 1.0
    assert color_for_region(21) != color_for_region(22)


def test_palette_preview_grid():
    img = palette_preview(40)
    assert img.shape == (5 * 80, 8 * 80, 3) and img.dtype == np.uint8
    # cell for region 1 is filled with its color
    assert tuple(int(v) for v in img[8, 8]) == PALETTE_20[0]


# ---------------------------------------------------------------- gamma LUT


def test_gamma_lut_gamma_1_is_linear_identity():
    lo, hi = 1000.0, 50000.0
    assert np.array_equal(gamma_lut(lo, hi, 1.0), gamma_lut(lo, hi, None))
    lut = gamma_lut(0.0, 65535.0, None)
    v = np.arange(65536)
    assert np.all(np.abs(lut.astype(int) - np.round(v / 65535 * 255)) <= 1)


@pytest.mark.parametrize("gamma", [None, 1.0, 2.2, 5.0])
def test_gamma_lut_monotonic_and_clamped(gamma):
    lo, hi = 1000.0, 50000.0
    lut = gamma_lut(lo, hi, gamma)
    assert lut.shape == (65536,) and lut.dtype == np.uint8
    assert np.all(np.diff(lut.astype(int)) >= 0)
    assert np.all(lut[:1001] == 0) and np.all(lut[50000:] == 255)


def test_larger_gamma_brightens_shadows():
    a, b = gamma_lut(0, 65535, 2.2), gamma_lut(0, 65535, 3.0)
    assert b[3000] > a[3000] > gamma_lut(0, 65535, None)[3000]


def test_lut_matches_direct_formula():
    img = np.random.default_rng(1).integers(0, 65536, (20, 20, 3)).astype(np.uint16)
    lo, hi = stretch_params(img)
    x = np.clip((img.astype(np.float64) - lo) / (hi - lo), 0, 1) ** (1 / 2.2)
    out = to_display_8bit(img, lo, hi, 2.2)
    assert np.all(np.abs(out.astype(int) - np.round(x * 255)) <= 1)
    # float images take the direct path and agree
    assert np.all(np.abs(to_display_8bit(img.astype(np.float32), lo, hi, 2.2).astype(int) - out) <= 1)


def test_step_gamma():
    assert step_gamma(2.2, 1) == 2.4 and step_gamma(2.2, -1) == 2.0
    assert step_gamma(5.0, 1) == 5.0 and step_gamma(1.0, -1) == 1.0
    g = GAMMA_DEFAULT
    for _ in range(20):
        g = step_gamma(g, 1)
    assert g == 5.0
    for _ in range(20):
        g = step_gamma(g, -1)
    assert g == 1.0
    g = 1.0
    for _ in range(6):
        g = step_gamma(g, 1)
    assert g == 2.2  # no float drift


# ---------------------------------------------------------------- panel layout


def test_panel_side():
    assert panel_side(768, 1023) == "right"  # portrait
    assert panel_side(1023, 768) == "bottom"  # landscape
    assert panel_side(500, 500) == "bottom"
    assert panel_side(768, 1023, "bottom") == "bottom"
    assert panel_side(1023, 768, "right") == "right"


def test_compute_scale_with_panel():
    # right panel: width budget loses PANEL_RIGHT_W
    s = compute_scale(768, 1023, 1000, 2000, "right")
    assert s == pytest.approx((1000 - PANEL_RIGHT_W) / 768)
    # bottom panel: height budget loses PANEL_BOTTOM_H
    s = compute_scale(1023, 768, 3000, 600, "bottom")
    assert s == pytest.approx((600 - PANEL_BOTTOM_H) / 768)
    # never upscales
    assert compute_scale(100, 80, 3000, 3000, "right") == 1.0
    assert compute_scale(100, 80, 3000, 3000, "bottom") == 1.0
    # degenerate max-size stays positive
    assert compute_scale(768, 1023, 100, 100, "right") > 0


def test_wrap_text_fits_and_keeps_words():
    text = "m = material (one reflectance, no variation: paint, a single brick, plastic)"
    lines = wrap_text(text, 200)
    assert len(lines) > 1
    assert all(text_width(ln) <= 200 for ln in lines)
    assert " ".join(lines).split() == text.split()
    assert wrap_text("", 100) == [""]
    assert wrap_text("supercalifragilistic", 10) == ["supercalifragilistic"]


# ---------------------------------------------------------------- Annotator (no windows)


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _annotator(tmp_path, n=2, size=(120, 90), panel="auto", **kw):
    """Annotator over n synthetic uint16 images; never opens a window."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir(parents=True)
    w, h = size
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 65535, (h, w, 3)).astype(np.uint16)
        cv2.imwrite(str(img_dir / f"im{i}.png"), img)
    out = tmp_path / "out"
    out.mkdir(parents=True)
    clock = FakeClock()
    an = Annotator(sorted(img_dir.iterdir()), out, 1600, 900, panel=panel, clock=clock, **kw)
    return an, clock


def _click(an, x, y, flags=0, event=cv2.EVENT_LBUTTONDOWN):
    an.on_main_mouse(cv2.EVENT_MOUSEMOVE, x, y, 0, None)
    an.on_main_mouse(event, x, y, flags, None)


def _two_regions(an):
    _click(an, 10, 10)
    _click(an, 20, 10)
    an.handle_key(ord(" "))
    _click(an, 60, 50)
    assert an.state.region_ids() == [1, 2] and an.state.region_id == 2


def test_pick_mode_hit_selects_and_returns_to_add(tmp_path):
    an, _ = _annotator(tmp_path)
    _two_regions(an)
    an.handle_key(ord("r"))
    assert an.mode == an.PICK
    _click(an, 12, 11)  # near region 1's dot
    assert an.mode == an.ADD and an.state.region_id == 1
    assert an.message() == "selected region 1"
    _click(an, 30, 30)  # further left-clicks add to region 1
    assert an.state.region_points(1)[-1].x == 30


def test_pick_mode_miss_stays_in_pick_and_adds_nothing(tmp_path):
    an, _ = _annotator(tmp_path)
    _two_regions(an)
    n = len(an.state.points)
    an.handle_key(ord("r"))
    _click(an, 100, 80)
    assert an.mode == an.PICK and len(an.state.points) == n
    assert an.message() == "no dot near click"
    an.handle_key(27)  # Esc cancels pick
    assert an.mode == an.ADD


def test_pick_tie_goes_to_most_recent(tmp_path):
    an, _ = _annotator(tmp_path)
    _click(an, 10, 10)
    an.handle_key(ord(" "))
    _click(an, 16, 10)  # region 2, equidistant from (13,10)
    an.handle_key(ord(" "))
    an.handle_key(ord("r"))
    _click(an, 13, 10)
    assert an.state.region_id == 2


def test_ctrl_click_and_right_click_pick(tmp_path):
    an, _ = _annotator(tmp_path)
    _two_regions(an)
    _click(an, 11, 10, flags=cv2.EVENT_FLAG_CTRLKEY)
    assert an.state.region_id == 1 and an.mode == an.ADD
    assert len(an.state.points) == 3  # ctrl-click did not add a point
    _click(an, 61, 50, event=cv2.EVENT_RBUTTONDOWN)
    assert an.state.region_id == 2


def test_panel_clicks_ignored(tmp_path):
    an, _ = _annotator(tmp_path)  # 120x90 landscape -> bottom panel
    assert an.side == "bottom"
    dw, dh = an.disp
    _click(an, 10, dh + 5)
    _click(an, dw + 50, 10)
    assert an.state.points == []
    an.handle_key(ord("r"))
    _click(an, 10, dh + 5)
    assert an.mode == an.PICK  # panel click in pick mode ignored, no message change


def test_esc_never_quits(tmp_path):
    an, _ = _annotator(tmp_path)
    assert an.handle_key(27) is True
    an.handle_key(ord("r"))
    assert an.handle_key(27) is True and an.mode == an.ADD
    assert an.handle_key(ord("q")) is False


def test_esc_closes_zoom_state(tmp_path):
    an, _ = _annotator(tmp_path)
    an.zoom_center = (10, 10)  # as if open, without creating a window
    an.zoom_bounds = zoom_crop_bounds(10, 10, ZOOM_HALF, an.w, an.h)
    an.handle_key(27)
    assert an.zoom_center is None and an.zoom_bounds is None


def test_zoom_x_close_resets_state(tmp_path):
    an, _ = _annotator(tmp_path)
    an.zoom_center = (10, 10)
    an.zoom_bounds = zoom_crop_bounds(10, 10, ZOOM_HALF, an.w, an.h)
    an.hover = (an.ZOOMWIN, (5, 5))
    an.zoom_closed()  # what run() does when the zoom window's X is clicked
    assert an.zoom_center is None and an.hover is None
    assert an.hover_target() is None
    an.on_zoom_mouse(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)  # late event: no crash
    assert an.state.points == []


def test_next_on_last_and_back_on_first(tmp_path):
    an, _ = _annotator(tmp_path, n=2)
    an.handle_key(ord("b"))
    assert an.idx == 0 and an.message() == "first image"
    an.handle_key(ord("n"))
    assert an.idx == 1
    an.handle_key(ord("n"))
    assert an.idx == 1 and an.message() == "last image"


def test_gamma_keys(tmp_path):
    an, _ = _annotator(tmp_path)
    assert an.gamma() == 2.2
    an.handle_key(ord("+"))
    assert an.gamma() == 2.4
    an.handle_key(ord("_"))
    an.handle_key(ord("-"))
    assert an.gamma() == 2.0
    an.handle_key(ord("g"))
    assert an.gamma() is None and an.gamma_value == 2.0  # off keeps the value
    an.handle_key(ord("="))
    assert an.gamma() == 2.0  # + while off only turns gamma on
    an.handle_key(ord("="))
    assert an.gamma() == 2.2


def test_gamma_persists_across_images_and_caches(tmp_path):
    an, _ = _annotator(tmp_path, n=2)
    an.handle_key(ord("+"))
    a = an.full_img()
    assert an.full_img() is a  # cached per gamma
    an.handle_key(ord("n"))
    assert an.idx == 1 and an.gamma() == 2.4
    assert set(an._full) == set()  # caches reset per image
    an.render()
    assert set(an._full) == {2.4}


def test_message_expires(tmp_path):
    an, clock = _annotator(tmp_path)
    an.handle_key(ord("s"))
    assert an.message() == "saved"
    clock.t += MSG_SECONDS - 0.1
    assert an.message() == "saved"
    clock.t += 0.2
    assert an.message() == ""


def test_label_key_on_existing_region_message(tmp_path):
    an, _ = _annotator(tmp_path)
    _two_regions(an)
    an.handle_key(ord("r"))
    _click(an, 10, 10)  # select region 1 (material)
    an.handle_key(ord("t"))
    assert an.state.region_id == 3 and an.message() == "new region 3 started (texture)"
    an.handle_key(ord("p"))  # empty new region: just relabel
    assert an.state.region_id == 3 and an.message() == "label specular"
    an.handle_key(ord(" "))
    assert an.message() == "current region is empty"


def test_space_message(tmp_path):
    an, _ = _annotator(tmp_path)
    _click(an, 5, 5)
    an.handle_key(ord(" "))
    assert an.message() == "region 2 started"


def test_render_layout_bottom_and_right(tmp_path):
    an, _ = _annotator(tmp_path, size=(120, 90))
    canvas = an.render()
    dw, dh = an.disp
    assert an.side == "bottom" and canvas.shape[1] == max(dw, BOTTOM_MIN_W)
    assert canvas.shape[0] >= dh + PANEL_BOTTOM_H
    assert np.array_equal(canvas[:dh, :dw], an.scaled_img())  # image at the origin

    an2, _ = _annotator(tmp_path / "p", size=(90, 160))
    canvas = an2.render()
    dw, dh = an2.disp
    assert an2.side == "right" and canvas.shape[1] == dw + PANEL_RIGHT_W
    assert canvas.shape[0] >= dh
    assert np.array_equal(canvas[:dh, :dw], an2.scaled_img())


def test_debug_mouse_prints(tmp_path, capsys):
    an, _ = _annotator(tmp_path, debug_mouse=True)
    an.on_main_mouse(cv2.EVENT_MOUSEMOVE, 5, 5, 0, None)  # plain move: not logged
    an.on_main_mouse(cv2.EVENT_RBUTTONDOWN, 5, 5, cv2.EVENT_FLAG_RBUTTON, None)
    out = capsys.readouterr().out
    assert "event=RBUTTONDOWN" in out and "flags=RBUTTON" in out
    assert "MOUSEMOVE" not in out


def test_compute_scale_with_measured_panel():
    nominal = compute_scale(1023, 768, 3000, 900, "bottom")
    measured = compute_scale(1023, 768, 3000, 900, "bottom", panel=264)
    assert measured == pytest.approx((900 - 264) / 768) and measured < nominal


def test_landscape_window_fits_max_size(tmp_path):
    an, _ = _annotator(tmp_path, size=(1023, 768))
    an.max_w, an.max_h = 1600, 900
    an.load(0)
    canvas = an.render()
    assert an.side == "bottom" and canvas.shape[0] <= 900 and canvas.shape[1] <= 1600


# ---------------------------------------------------------------- super-regions

from annotate import (  # noqa: E402
    CSV_HEADER_V1,
    PALETTE_SUPER,
    color_for_super,
    read_csv_supers,
    rows_for_save_supers,
    super_list_lines,
)


def _bricks():
    """Regions 1, 2, 4 texture; region 3 material; region 1 current at the end."""
    st = AnnotationState()
    st.set_label("texture")
    st.add_point(10, 10)            # r1 texture
    st.new_region()
    st.add_point(20, 10)            # r2 texture
    st.set_label("material")
    st.add_point(30, 10)            # r3 material
    st.set_label("texture")
    st.add_point(40, 10)            # r4 texture
    st.add_point(41, 10)            # r4 second point
    st.select_region(1)
    return st


def _frozen(st):
    return list(st.points), st.region_id, st.label


def test_create_super_from_current_region():
    st = _bricks()
    before = _frozen(st)
    ok, msg = st.create_super()
    assert ok and st.super_of == {1: 1} and st.current_super == 1
    assert st.super_label(1) == "texture" and "S1" in msg
    assert _frozen(st) == before  # points, current region and label untouched


def test_create_rejected_for_member_and_empty_region():
    st = _bricks()
    st.create_super()
    n = len(st.history)
    ok, msg = st.create_super()
    assert not ok and "use a to move it" in msg and len(st.history) == n
    st.cycle_region(-1)  # wraps to the empty (new) region
    assert not st.is_existing()
    ok, msg = st.create_super()
    assert not ok and "no points" in msg and st.super_of == {1: 1}


def test_add_to_super_and_label_mismatch():
    st = _bricks()
    st.create_super()            # S1 = {1}
    st.select_region(2)
    ok, _ = st.add_to_super()
    assert ok and st.super_members(1) == [1, 2]
    st.select_region(3)          # material
    n = len(st.history)
    ok, msg = st.add_to_super()
    assert not ok and "label mismatch" in msg
    assert st.super_members(1) == [1, 2] and len(st.history) == n


def test_add_rejections():
    st = _bricks()
    ok, msg = st.add_to_super()
    assert not ok and msg == "no super-region selected"
    st.create_super()
    ok, msg = st.add_to_super()  # already a member
    assert not ok and "already in S1" in msg
    st.new_region()              # empty current region
    ok, msg = st.add_to_super()
    assert not ok and "no points" in msg


def test_move_between_supers():
    st = _bricks()
    st.create_super()            # S1 = {1}
    st.select_region(2)
    st.add_to_super()            # S1 = {1, 2}
    st.select_region(4)
    st.create_super()            # S2 = {4}, current
    st.select_region(2)
    ok, msg = st.add_to_super()
    assert ok and msg == "moved from S1 to S2"
    assert st.super_members(1) == [1] and st.super_members(2) == [2, 4]


def test_move_last_member_dissolves_old_super():
    st = _bricks()
    st.create_super()            # S1 = {1}
    st.select_region(2)
    st.create_super()            # S2 = {2}, current
    st.select_region(1)
    ok, msg = st.add_to_super()  # move region 1 from S1 to S2
    assert ok and msg == "moved from S1 to S2"
    assert st.super_ids() == [2] and st.current_super == 2


def test_drop_leaves_points_and_regions():
    st = _bricks()
    st.create_super()
    st.select_region(2)
    st.add_to_super()
    before = _frozen(st)
    ok, msg = st.drop_from_super()
    assert ok and st.super_of == {1: 1} and st.current_super == 1
    assert _frozen(st) == before
    ok, msg = st.drop_from_super()
    assert not ok and "no super-region" in msg


def test_super_vanishes_with_last_member_and_deselects():
    st = _bricks()
    st.create_super()
    ok, msg = st.drop_from_super()
    assert ok and "(S1 removed)" in msg
    assert st.super_of == {} and st.current_super is None and st.super_ids() == []


def test_next_super_id_is_max_plus_one_across_labels():
    st = _bricks()
    st.create_super()            # S1 texture
    st.select_region(3)
    st.create_super()            # S2 material: one id space
    assert st.super_ids() == [1, 2]
    st.select_region(1)
    st.drop_from_super()         # S1 gone
    st.select_region(4)
    st.create_super()
    assert st.super_ids() == [2, 3]


def test_single_member_super_is_kept():
    st = _bricks()
    st.create_super()
    st.select_region(2)
    assert st.super_ids() == [1] and st.current_super == 1


def test_deleting_last_point_leaves_super_and_undo_restores():
    st = _bricks()
    st.create_super()            # S1 = {1}
    st.select_region(2)
    st.add_to_super()            # S1 = {1, 2}
    st.select_region(4)          # current region elsewhere
    st.delete_point(20, 10)      # region 2's only point
    assert st.super_of == {1: 1} and st.region_ids() == [1, 3, 4]
    assert st.undo() == "delete"
    assert st.points[1] == Point(20, 10, 2, "texture")
    assert st.super_of == {1: 1, 2: 1} and st.current_super == 1
    assert (st.region_id, st.label) == (4, "texture")


def test_deleting_last_member_clears_current_super_and_undo_restores():
    st = _bricks()
    st.create_super()            # S1 = {1}, current
    st.delete_point(10, 10)      # region 1's only point
    assert st.super_of == {} and st.current_super is None
    st.undo()
    assert st.super_of == {1: 1} and st.current_super == 1


def test_deleting_non_last_point_keeps_membership():
    st = _bricks()
    st.select_region(4)
    st.create_super()
    st.delete_point(41, 10)
    assert st.super_of == {4: 1}


def test_undo_of_seeded_add_prunes_loaded_member():
    pts = [Point(0, 0, 1, "texture"), Point(5, 5, 2, "texture")]
    st = state_from_points(pts, {1: 1, 2: 1})
    assert st.current_super is None and st.next_super_id() == 2
    st.undo()                    # removes (5,5): region 2 gone
    assert st.super_of == {1: 1}
    st.undo()
    assert st.super_of == {} and st.super_ids() == []


def test_undo_super_actions_on_one_stack():
    st = _bricks()
    snaps = []
    def snap():
        snaps.append((dict(st.super_of), st.current_super, list(st.points)))
    snap(); st.create_super()                          # c
    snap(); st.select_region(2); st.add_to_super()     # a
    snap(); st.add_point(50, 50)                       # point add in region 2
    snap(); st.select_region(4); st.create_super()     # c (S2)
    snap(); st.select_region(2); st.add_to_super()     # a = move S1 -> S2
    snap(); st.drop_from_super()                       # d
    for kind in ("super", "super", "super", "add", "super", "super"):
        assert st.undo() == kind
        s, cur, pts = snaps.pop()
        assert (st.super_of, st.current_super, st.points) == (s, cur, pts)
    assert snaps == []


def test_select_and_cycle_supers_not_undoable():
    st = _bricks()
    st.create_super()
    st.select_region(3)
    st.create_super()
    st.select_region(4)
    st.create_super()            # S1, S2, S3; current S3
    n = len(st.history)
    assert st.cycle_super(1) == 1     # wraps
    assert st.cycle_super(-1) == 3    # wraps back
    assert st.cycle_super(-1) == 2
    assert st.select_super(1) and not st.select_super(9)
    assert len(st.history) == n


def test_cycle_supers_from_none_and_empty():
    st = _bricks()
    assert st.cycle_super(1) is None and st.current_super is None
    st.create_super()
    st.select_region(4)
    st.add_to_super()
    st.select_region(3)
    st.create_super()            # S1 {1,4}, S2 {3}
    st.current_super = None
    assert st.cycle_super(1) == 1
    st.current_super = None
    assert st.cycle_super(-1) == 2


def test_no_supers_means_identical_behavior():
    st = _bricks()
    assert st.super_of == {} and st.current_super is None
    assert all(a.supers in (None, ((), None)) for a in st.history)


# ---------------------------------------------------------------- super-regions: CSV


def test_csv_round_trip_with_supers(tmp_path):
    st = _bricks()
    st.create_super()
    st.select_region(4)
    st.add_to_super()
    st.select_region(3)
    st.create_super()
    path = tmp_path / "img.csv"
    write_csv(path, "img.tif", st.points, st.super_of)
    pts, sup = read_csv_supers(path)
    assert pts == rows_for_save(st.points)
    assert sup == {1: 1, 4: 1, 3: 2}
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(CSV_HEADER) and lines[0].endswith(",super_region_id")
    assert lines[1] == "img.tif,10,10,1,texture,1"
    assert lines[2] == "img.tif,20,10,2,texture,0"
    first = path.read_bytes()
    write_csv(path, "img.tif", pts, sup)
    assert path.read_bytes() == first


def test_csv_reads_5_column_files(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text(",".join(CSV_HEADER_V1) + "\nimg.tif,1,2,1,material\nimg.tif,3,4,2,texture\n")
    pts, sup = read_csv_supers(path)
    assert pts == [Point(1, 2, 1, "material"), Point(3, 4, 2, "texture")] and sup == {}
    assert read_csv(path) == pts


def test_csv_region_with_two_super_ids_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(",".join(CSV_HEADER) + "\nimg.tif,1,2,1,texture,1\nimg.tif,3,4,1,texture,2\n")
    with pytest.raises(ValueError, match=r"bad.csv:3: region 1 has super_region_id 2, but 1 on line 2"):
        read_csv_supers(path)


def test_csv_super_with_mixed_labels_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(",".join(CSV_HEADER) + "\nimg.tif,1,2,1,texture,1\nimg.tif,3,4,2,material,1\n")
    with pytest.raises(ValueError, match=r"bad.csv:3: super-region 1 mixes labels 'texture' \(line 2\)"):
        read_csv_supers(path)


@pytest.mark.parametrize("sid", ["-1", "x", "1.5"])
def test_csv_bad_super_value_raises(tmp_path, sid):
    path = tmp_path / "bad.csv"
    path.write_text(",".join(CSV_HEADER) + f"\nimg.tif,1,2,1,texture,{sid}\n")
    with pytest.raises(ValueError, match="bad.csv:2"):
        read_csv_supers(path)


def test_save_renumbers_regions_and_supers_with_gaps():
    # regions 3, 7, 9, 12 with gaps; supers 5 = {9, 12} (texture), 2 = {7} (material)
    pts = [Point(0, 0, 12, "texture"), Point(1, 1, 3, "texture"), Point(2, 2, 9, "texture"),
           Point(3, 3, 7, "material"), Point(4, 4, 12, "texture")]
    rows = rows_for_save_supers(pts, {9: 5, 12: 5, 7: 2})
    got = [(p.x, p.region_id, sid) for p, sid in rows]
    # regions -> 3:1, 7:2, 9:3, 12:4 ; supers ordered by smallest member: 2 ({2}) -> 1, 5 ({3,4}) -> 2
    assert got == [(1, 1, 0), (3, 2, 1), (2, 3, 2), (0, 4, 2), (4, 4, 2)]


def test_save_after_loaded_supers_round_trip(tmp_path):
    pts = [Point(0, 0, 1, "texture"), Point(1, 1, 2, "material"), Point(2, 2, 3, "texture")]
    path = tmp_path / "img.csv"
    write_csv(path, "img.tif", pts, {3: 4, 1: 4, 2: 9})
    p2, s2 = read_csv_supers(path)
    assert s2 == {1: 1, 3: 1, 2: 2}
    st = state_from_points(p2, s2)
    assert st.current_super is None and st.next_super_id() == 3


# ---------------------------------------------------------------- super-regions: display


def test_palette_super_distinct_and_spread():
    assert len(PALETTE_SUPER) == 12 and len(set(PALETTE_SUPER)) == 12
    labs = [_lab(c) for c in PALETTE_SUPER]
    d = min(np.linalg.norm(a - b) for a, b in itertools.combinations(labs, 2))
    assert d >= 25
    assert color_for_super(1) == PALETTE_SUPER[0] and color_for_super(13) == PALETTE_SUPER[0]
    assert color_for_super(12) == PALETTE_SUPER[11]


def test_super_list_lines_and_truncation():
    st = _bricks()
    assert super_list_lines(st) == ["(none yet)"]
    st.create_super()
    st.select_region(4)
    st.add_to_super()
    assert super_list_lines(st) == ["S1 texture: 1,4"]
    for rid in (2, 3):
        st.select_region(rid)
        st.create_super()
    lines = super_list_lines(st, max_lines=2)
    assert lines == ["S1 texture: 1,4", "+2 more"]
    assert super_list_lines(st, max_lines=3) == ["S1 texture: 1,4", "S2 texture: 2", "S3 material: 3"]


def _super_annotator(tmp_path):
    an, clock = _annotator(tmp_path)
    _two_regions(an)             # region 1: (10,10),(20,10); region 2: (60,50); current = 2
    an.handle_key(ord("c"))      # S1 = {2}
    return an, clock


def test_display_toggle_leaves_state_unchanged(tmp_path):
    an, _ = _super_annotator(tmp_path)
    st = an.state
    before = (list(st.points), dict(st.super_of), st.current_super, list(st.history), st.region_id)
    an.handle_key(ord("v"))
    assert st.show_supers is False and an.message() == "super display OFF"
    assert (list(st.points), dict(st.super_of), st.current_super, list(st.history), st.region_id) == before
    an.render()
    an.handle_key(ord("v"))
    assert st.show_supers is True


def test_super_keys_turn_display_on(tmp_path):
    an, _ = _super_annotator(tmp_path)
    an.handle_key(ord("v"))
    an.handle_key(ord("."))
    assert an.state.show_supers and an.message().endswith("super display on")


def test_show_supers_carries_over_images(tmp_path):
    an, _ = _annotator(tmp_path, n=2)
    an.handle_key(ord("v"))
    an.handle_key(ord("n"))
    assert an.idx == 1 and an.state.show_supers is False


def test_super_pick_hit_sets_current_super_not_region(tmp_path):
    an, _ = _super_annotator(tmp_path)
    an.handle_key(ord("["))      # current region 1
    assert an.state.region_id == 1
    an.handle_key(ord("y"))
    assert an.mode == an.SUPER_PICK
    an.state.current_super = None
    _click(an, 61, 50)           # region 2's dot (in S1)
    assert an.state.current_super == 1 and an.state.region_id == 1 and an.mode == an.ADD
    assert len(an.state.points) == 3


def test_super_pick_miss_and_region_without_super_stay(tmp_path):
    an, _ = _super_annotator(tmp_path)
    an.state.current_super = None
    an.handle_key(ord("y"))
    _click(an, 100, 80)
    assert an.mode == an.SUPER_PICK and an.message() == "no dot near click"
    _click(an, 11, 10)           # region 1, in no super
    assert an.mode == an.SUPER_PICK and an.message() == "region 1 is in no super-region"
    assert an.state.current_super is None and len(an.state.points) == 3
    an.handle_key(27)
    assert an.mode == an.ADD


def test_super_pick_tie_goes_to_most_recent(tmp_path):
    an, _ = _annotator(tmp_path)
    _click(an, 10, 10)
    an.handle_key(ord("c"))      # S1 = {1}
    an.handle_key(ord(" "))
    _click(an, 16, 10)
    an.handle_key(ord("c"))      # S2 = {2}
    an.state.current_super = None
    an.handle_key(ord("y"))
    _click(an, 13, 10)           # equidistant
    assert an.state.current_super == 2


def test_super_keys_messages(tmp_path):
    an, _ = _super_annotator(tmp_path)
    an.handle_key(ord("c"))
    assert "use a to move it" in an.message()
    an.handle_key(ord("["))      # region 1
    an.handle_key(ord("a"))
    assert an.message() == "added region 1 to S1"
    an.handle_key(ord("d"))
    assert an.message() == "dropped region 1 from S1"
    an.handle_key(ord("u"))
    assert an.message() == "undid super-region change" and an.state.super_of == {1: 1, 2: 1}
    an.handle_key(ord(","))
    assert an.message() == "selected S1 (material): regions 1,2"


def test_panel_super_lines(tmp_path):
    an, _ = _super_annotator(tmp_path)
    texts = [ln.text for ln in an.super_items()]
    assert "region 2: in S1" in texts
    assert "super: S1 (material), regions 2" in texts
    assert "display ON (v)" in texts and "S1 material: 2" in texts
    an.handle_key(ord("["))
    an.state.current_super = None
    texts = [ln.text for ln in an.super_items()]
    assert "region 1: in no super" in texts and "super: none selected" in texts


def test_saved_csv_from_annotator_has_supers(tmp_path):
    an, _ = _super_annotator(tmp_path)
    an.handle_key(ord("s"))
    pts, sup = read_csv_supers(an.csv_path(an.images[0]))
    assert sup == {2: 1} and len(pts) == 3
    an.handle_key(ord("n"))
    an.handle_key(ord("b"))
    assert an.state.super_of == {2: 1} and an.state.current_super is None
