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
