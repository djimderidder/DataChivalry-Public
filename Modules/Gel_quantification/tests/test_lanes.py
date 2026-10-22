"""Lane detection.

US-11: detect_lanes(pixels, n_lanes) -- the main method, uses the comb wells.
US-1:  lanes_from_profile(pixels)    -- fallback without comb or lane count.
"""

import numpy as np
import pytest

from gelanalyzer.lanes import detect_lanes, lanes_from_profile
from synthetic import make_gel

# --- US-11: from the comb ----------------------------------------------------------


def test_detect_lanes_on_gel_with_comb():
    gel = make_gel(n_lanes=10, width=900, height=500, margin=(40, 30, 50, 35), comb=True, seed=3)
    lanes = detect_lanes(gel.image, n_lanes=10)
    assert lanes == pytest.approx(gel.lane_centers, abs=3)


def test_detect_lanes_returns_n_lanes_left_to_right():
    gel = make_gel(n_lanes=12, width=1000, height=500, margin=(30, 25, 30, 30), comb=True, seed=5)
    lanes = detect_lanes(gel.image, n_lanes=12)
    assert len(lanes) == 12
    assert lanes == sorted(lanes)


# --- US-1: brightness profile fallback (no comb) -----------------------------------


def test_finds_correct_number_of_lanes(simple_gel):
    lanes = lanes_from_profile(simple_gel.image)
    assert len(lanes) == len(simple_gel.lane_centers)


def test_lane_positions_are_accurate(simple_gel):
    lanes = lanes_from_profile(simple_gel.image)
    for found, true in zip(lanes, simple_gel.lane_centers):
        assert found == pytest.approx(true, abs=5)


def test_lanes_are_sorted_left_to_right(simple_gel):
    lanes = lanes_from_profile(simple_gel.image)
    assert lanes == sorted(lanes)


def test_works_on_noisy_gel(noisy_gel):
    lanes = lanes_from_profile(noisy_gel.image)
    assert len(lanes) == len(noisy_gel.lane_centers)


def test_blank_image_has_no_lanes():
    assert lanes_from_profile(np.zeros((100, 200))) == []
