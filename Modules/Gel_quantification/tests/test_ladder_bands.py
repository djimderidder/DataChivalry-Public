"""US-14: identify the 10 ladder bands.

Product owner's recipe:
  1. lane profile along the ladder's centre line, at most the lane width
  2. rolling-disc background, then peaks -- lower the bar until we have 10,
     with a minimum distance from the ladder's band spacing
  3. more than 10: check for spillage from the neighbouring lanes
  4. assign kDa top to bottom; the thick bands (75/50/25) confirm it
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gelanalyzer.config import LADDERS, GelConfig
from gelanalyzer.io import GelImage
from gelanalyzer.ladder import find_ladder, identify_ladder_bands
from gelanalyzer.plotting import plot_ladder
from gelanalyzer.profiles import lane_profile, rolling_disc_background
from synthetic import make_gel

BASE = dict(n_lanes=10, width=900, height=600, margin=(40, 30, 50, 35), comb=True,
            bands_per_lane=6, seed=7)
PRECISION_PLUS = LADDERS["Precision Plus Unstained"]


def gel_with_ladder(ladder_lane=3, **kwargs):
    truth = make_gel(**{**BASE, **kwargs}, ladder_lane=ladder_lane)
    config = GelConfig(n_lanes=10, ladder_lanes=(ladder_lane + 1,), ladder=PRECISION_PLUS)
    return truth, GelImage(pixels=truth.image, config=config)


def true_ladder_ys(truth, lane=3):
    return sorted(b.y for b in truth.bands if b.lane == lane)


# --- building blocks ------------------------------------------------------------

def test_lane_profile_follows_a_slanted_line_and_averages_across_the_width():
    pixels = np.zeros((100, 100))
    for y in range(100):
        pixels[y, 20 + y // 5] = 10.0          # a slanted 1-px line: x = 20 + y/5
    ys, values = lane_profile(pixels, x_at=lambda y: 20 + np.asarray(y) / 5, width_at=lambda y: 0 * np.asarray(y) + 1,
                              y_start=0, y_end=100, fraction=1.0)
    assert len(ys) == 100
    assert values.min() > 0                     # the line is found on every row


def test_rolling_disc_removes_a_slow_background_but_keeps_narrow_peaks():
    x = np.arange(500)
    background = 100 + 0.2 * x                  # slow trend
    peaks = 50 * np.exp(-0.5 * ((x - 150) / 4) ** 2) + 80 * np.exp(-0.5 * ((x - 350) / 4) ** 2)
    estimate = rolling_disc_background(background + peaks, radius=30)
    assert estimate == pytest.approx(background, abs=7)
    corrected = background + peaks - estimate
    assert corrected[150] == pytest.approx(50, abs=7) and corrected[350] == pytest.approx(80, abs=7)


# --- identification -------------------------------------------------------------

def test_finds_all_ten_bands_in_order_with_their_kda():
    truth, gel = gel_with_ladder(fan=40)
    result = identify_ladder_bands(gel, find_ladder(gel))
    assert result.complete
    assert [b.y for b in result.bands] == pytest.approx(true_ladder_ys(truth), abs=3)
    assert [b.kda for b in result.bands] == list(PRECISION_PLUS.kda)


def test_thick_reference_bands_confirm_the_assignment():
    truth, gel = gel_with_ladder()
    result = identify_ladder_bands(gel, find_ladder(gel))
    assert result.thick_bands_ok
    assert {b.kda for b in result.bands if b.thick} == {75, 50, 25}


def test_spillage_from_a_neighbour_is_rejected():
    # an overloaded, extra-wide band in lane 5 (index 4) spills into the ladder lane (index 3)
    plain, _ = gel_with_ladder()
    ys = true_ladder_ys(plain)
    y_between = (ys[1] + ys[2]) / 2
    x_spill = plain.lane_centers[4] - 0.45 * plain.lane_pitch     # reaches well into the ladder lane
    truth, gel = gel_with_ladder(extra_bands=({"x": x_spill, "y": y_between, "intensity": 3.0, "width": 70},))
    result = identify_ladder_bands(gel, find_ladder(gel))
    assert len(result.bands) == 10
    assert [b.y for b in result.bands] == pytest.approx(true_ladder_ys(truth), abs=3)
    assert result.thick_bands_ok
    assert any(abs(y - y_between) < 5 for y in result.spillage)


def test_needs_a_ladder_definition():
    truth = make_gel(**BASE, ladder_lane=3)
    gel = GelImage(pixels=truth.image, config=GelConfig(n_lanes=10, ladder_lanes=(4,)))     # no ladder type
    with pytest.raises(ValueError, match="ladder"):
        identify_ladder_bands(gel, find_ladder(gel))


def test_diagnostic_plot_shows_the_profile(save_plot):
    truth, gel = gel_with_ladder(fan=40)
    ladder = find_ladder(gel)
    fig = plot_ladder(gel, ladder, identification=identify_ladder_bands(gel, ladder))
    ids = {a.get_gid() for ax in fig.axes for a in ax.get_children()}
    assert {"ladder-profile", "ladder-background", "ladder-kda"} <= ids
    save_plot("ladder_bands_synthetic", fig)
