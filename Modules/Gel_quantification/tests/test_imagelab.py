"""US-9: read Image Lab's own lane/band analysis from a .scn file.

This is REFERENCE data for validating our detectors -- the user selected
lanes and bands of interest by hand in Image Lab. The analysis pipeline itself
never depends on it (users will upload gels without opening Image Lab).
"""

import numpy as np
import pytest

from gelanalyzer.imagelab import read_imagelab_analysis
from scn_factory import lane_analysis_xml, write_fake_scn

FRAME = (10, 20, 110, 220)   # left, top, right, bottom -> 100 px wide, 200 px high
LANES = [
    dict(label="sample A", top_anchor=0.25, bottom_anchor=0.35, width=20, std=False, bands=[0.5]),
    dict(label="L", top_anchor=0.75, bottom_anchor=0.75, width=30, std=True, bands=[0.1, 0.9]),
]


@pytest.fixture
def analysis(tmp_path):
    pixels = np.zeros((240, 120), dtype=np.uint16)
    path = write_fake_scn(tmp_path / "gel.scn", pixels, lane_analysis=lane_analysis_xml(FRAME, LANES))
    return read_imagelab_analysis(path)


def test_file_without_analysis_returns_none(tmp_path):
    path = write_fake_scn(tmp_path / "gel.scn", np.zeros((5, 5), dtype=np.uint16))
    assert read_imagelab_analysis(path) is None


def test_frame(analysis):
    assert analysis.frame == (10, 20, 110, 220)


def test_lanes_are_numbered_from_1_left_to_right(analysis):
    assert [lane.number for lane in analysis.lanes] == [1, 2]
    assert [lane.label for lane in analysis.lanes] == ["sample A", "L"]


def test_lane_centre_at_top_and_bottom(analysis):
    lane = analysis.lanes[0]
    assert lane.x_top == pytest.approx(35)      # 10 + 0.25 * 100
    assert lane.x_bottom == pytest.approx(45)   # 10 + 0.35 * 100


def test_lane_x_at_any_height_follows_the_slant(analysis):
    lane = analysis.lanes[0]
    assert lane.x_at(120) == pytest.approx(40)  # halfway down -> halfway between 35 and 45


def test_lane_width(analysis):
    assert analysis.lanes[1].width == 30


def test_band_positions_in_pixels(analysis):
    assert analysis.lanes[0].band_ys == pytest.approx([120])       # 20 + 0.5 * 200
    assert analysis.lanes[1].band_ys == pytest.approx([40, 200])


def test_standard_lane_and_ladder(analysis):
    assert analysis.standard_lane.number == 2
    assert analysis.standard_name == "Bio-Rad Precision Plus"
    assert analysis.standard_kda == [250, 150, 100, 75, 50, 37, 25, 20, 15, 10]
