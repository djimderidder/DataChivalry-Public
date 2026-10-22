"""US-13: find and trace the ladder lane.

Product owner's recipe:
  1. crop from the middle of the left neighbour lane to the middle of the right
     neighbour (an edge lane: up to the gel border)
  2. background first, then threshold -> blobs
  3. the blobs roughly below each other in the middle are the ladder: a line
     through them gives the lane centre, their widths give the lane broadness
     (a trapezoid: width at top and bottom)
  4. lane end = lowest ladder band + the distance between the last two bands
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gelanalyzer.config import GelConfig
from gelanalyzer.io import GelImage
from gelanalyzer.ladder import find_ladder
from gelanalyzer.plotting import plot_ladder
from synthetic import make_gel

BASE = dict(n_lanes=10, width=900, height=600, margin=(40, 30, 50, 35), comb=True,
            bands_per_lane=6, seed=7)


def gel_with_ladder(ladder_lane=3, **kwargs):
    """Synthetic gel + GelImage with config. ladder_lane is 0-based here, 1-based in the config."""
    truth = make_gel(**{**BASE, **kwargs}, ladder_lane=ladder_lane)
    image = GelImage(pixels=truth.image, config=GelConfig(n_lanes=10, ladder_lanes=(ladder_lane + 1,)))
    return truth, image


def ladder_band_ys(truth, lane):
    return [b.y for b in truth.bands if b.lane == lane]


def test_centre_line_runs_through_the_ladder():
    truth, gel = gel_with_ladder(fan=40)
    ladder = find_ladder(gel)
    for y in ladder_band_ys(truth, 3):
        assert ladder.x_at(y) == pytest.approx(truth.lane_x_at(3, y), abs=3)


def test_finds_the_ladder_bands_and_only_those():
    truth, gel = gel_with_ladder(fan=40)
    ladder = find_ladder(gel)
    assert len(ladder.bands) >= 8
    for band in ladder.bands:                                    # nothing from the neighbours
        assert band.x == pytest.approx(truth.lane_x_at(3, band.y), abs=0.15 * truth.lane_pitch)


def test_width_is_the_band_width():
    truth, gel = gel_with_ladder()
    ladder = find_ladder(gel)
    y_mid = np.mean(ladder_band_ys(truth, 3))
    assert ladder.width_at(y_mid) == pytest.approx(40, rel=0.25)    # synthetic bands are 40 px wide


def test_lane_end_is_lowest_band_plus_last_spacing():
    truth, gel = gel_with_ladder()
    ladder = find_ladder(gel)
    ys = sorted(ladder_band_ys(truth, 3))
    assert ladder.y_end == pytest.approx(ys[-1] + (ys[-1] - ys[-2]), abs=5)


def test_lane_starts_at_the_well_floor():
    truth, gel = gel_with_ladder()
    assert find_ladder(gel).y_start == pytest.approx(truth.comb_bottom, abs=4)


def test_crop_is_from_neighbour_middle_to_neighbour_middle():
    truth, gel = gel_with_ladder(ladder_lane=3)
    x0, x1, _, _ = find_ladder(gel).crop
    assert x0 == pytest.approx(truth.lane_centers[2], abs=3)
    assert x1 == pytest.approx(truth.lane_centers[4], abs=3)


def test_ladder_in_the_first_lane_crops_to_the_gel_border():
    truth, gel = gel_with_ladder(ladder_lane=0)
    ladder = find_ladder(gel)
    assert ladder.crop[0] == pytest.approx(truth.gel_area[0], abs=4)
    for y in ladder_band_ys(truth, 0):
        assert ladder.x_at(y) == pytest.approx(truth.lane_x_at(0, y), abs=3)


def test_trapezoid_corners():
    truth, gel = gel_with_ladder()
    ladder = find_ladder(gel)
    (tl, tr, br, bl) = ladder.trapezoid()
    assert tl[1] == tr[1] == ladder.y_start and bl[1] == br[1] == ladder.y_end
    assert tr[0] - tl[0] == pytest.approx(ladder.width_at(ladder.y_start))


def test_lane_can_be_given_explicitly():
    truth, gel = gel_with_ladder(ladder_lane=6)
    assert find_ladder(gel, lane=7).lane == 7


def test_needs_a_config_or_a_lane():
    truth = make_gel(**BASE, ladder_lane=3)
    with pytest.raises(ValueError, match="config"):
        find_ladder(GelImage(pixels=truth.image))


def test_diagnostic_plot(save_plot):
    truth, gel = gel_with_ladder(fan=40)
    ladder = find_ladder(gel)
    fig = plot_ladder(gel, ladder)
    ids = {a.get_gid() for ax in fig.axes for a in ax.get_children()}
    assert {"ladder-trapezoid", "ladder-centre", "ladder-bands", "ladder-end"} <= ids
    save_plot("ladder_synthetic", fig)
