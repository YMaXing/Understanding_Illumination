"""Unit checks for the stripe diagnostics (T1) helpers."""

import numpy as np

from illum_baseline.diag import (shift_back, shift_input, spectral_peaks, stripe_crossings,
                                 transition_score, welch_columns, xcorr_v)


def test_shift_round_trip():
    x = np.random.default_rng(0).integers(0, 255, (40, 50, 3), dtype=np.uint8)
    s = shift_input(x, 7)
    assert s.shape == x.shape
    np.testing.assert_array_equal(s[7:, 7:], x[:-7, :-7])
    back = shift_back(s[..., 0].astype(float), 7)
    np.testing.assert_array_equal(back[:-7, :-7], x[:-7, :-7, 0])
    assert np.isnan(back[-7:]).all()


def test_spectrum_finds_period_and_xcorr_finds_lag():
    v = np.arange(200)[:, None] * np.ones((1, 30))
    e = 1e-3 * np.sin(2 * np.pi * v / 7.0)
    f, p, n = welch_columns(e, np.ones_like(e, bool), 64)
    assert n > 0 and abs(spectral_peaks(f, p)[0]["wavelength_px"] - 7.11) < 0.1  # nearest bin to 7 px
    sh = np.full_like(e, np.nan)
    sh[:-3] = e[3:]  # b[v] = a[v + 3]  ->  peak of sum a[v] b[v + lag] at lag -3
    xc = xcorr_v(e, sh, 5)
    assert max(xc, key=xc.get) == -3


def test_stripe_crossings_and_transition_score():
    v = np.arange(60)[:, None] * np.ones((1, 4))
    lor = 10 * np.sin(2 * np.pi * (v + 0.25) / 8.0)
    cross = stripe_crossings(lor, np.ones_like(lor, bool), 3.0)
    assert len(cross) == 4 * 14
    T = np.zeros((59, 4), bool)
    for c, z in cross:
        T[int(np.floor(z)), c] = True
    assert transition_score(cross, T) == 1.0
    assert transition_score(cross, T, shift=2) == 0.0  # crossings are 4 px apart: a 2-px shift misses all


def test_tta_shift_helpers_and_latin_set():
    from illum_baseline.depth import shift_image, tta_shifts, unshift

    x = np.random.default_rng(1).normal(size=(30, 40))
    for dx, dy in [(7, 0), (0, 7), (7, 7), (-5, 3), (4, -6)]:
        back = unshift(shift_image(x, dx, dy), dx, dy)
        m = np.isfinite(back)
        assert m.sum() == (30 - abs(dy)) * (40 - abs(dx))
        np.testing.assert_array_equal(back[m], x[m])
    lat = tta_shifts("latin14")
    assert sorted(d % 14 for d, _ in lat) == list(range(14)) and sorted(d % 14 for _, d in lat) == list(range(14))
    # A 7-px ripple along v averages out over the latin14 y-phases.
    v = np.arange(28)
    assert abs(np.mean([np.sin(2 * np.pi * (v[10] - dy) / 7) for _, dy in lat])) < 1e-12
    assert tta_shifts("spec") == [(0, 0), (7, 0), (0, 7), (7, 7)]
