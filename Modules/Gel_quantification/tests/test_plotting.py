"""Comparison plot: gel image + Image Lab reference lanes + our detected lanes.

We can't easily assert that a plot "looks right" -- that's what looking at it
is for. These tests check its structure: the right things are drawn, numbered
left to right. Artists get a `gid` (an id) so tests can find them.
"""

import matplotlib

matplotlib.use("Agg")  # draw without a screen

import numpy as np
import pytest

from gelanalyzer.imagelab import ImageLabAnalysis, ReferenceLane
from gelanalyzer.io import GelImage
from gelanalyzer.plotting import plot_lanes


@pytest.fixture
def gel():
    return GelImage(pixels=np.random.default_rng(0).random((100, 200)))


@pytest.fixture
def reference():
    lanes = [
        ReferenceLane(number=1, label="a", x_top=50, x_bottom=55, y_top=10, y_bottom=90,
                      width=20, is_standard=False, band_ys=[40]),
        ReferenceLane(number=2, label="L", x_top=150, x_bottom=150, y_top=10, y_bottom=90,
                      width=20, is_standard=True, band_ys=[20, 50, 80]),
    ]
    return ImageLabAnalysis(frame=(30, 10, 170, 90), lanes=lanes)


def gids(ax, prefix):
    found = [a.get_gid() for a in ax.get_children() if (a.get_gid() or "").startswith(prefix)]
    return sorted(found)


def texts(ax, prefix):
    return sorted(t.get_text() for t in ax.texts if (t.get_gid() or "").startswith(prefix))


def test_shows_the_gel_image(gel):
    ax = plot_lanes(gel)
    assert len(ax.images) == 1


def test_draws_one_box_per_reference_lane(gel, reference):
    ax = plot_lanes(gel, reference=reference)
    assert gids(ax, "reference-lane-") == ["reference-lane-1", "reference-lane-2"]


def test_reference_lanes_are_numbered_and_ladder_is_marked(gel, reference):
    ax = plot_lanes(gel, reference=reference)
    assert texts(ax, "reference-number-") == ["1", "2 (L)"]


def test_draws_reference_bands(gel, reference):
    ax = plot_lanes(gel, reference=reference)
    assert len(gids(ax, "reference-band-")) == 4      # 1 + 3 bands


def test_draws_detected_lanes_numbered_left_to_right(gel):
    ax = plot_lanes(gel, detected=[120.0, 40.0, 80.0])   # unsorted on purpose
    assert gids(ax, "detected-lane-") == ["detected-lane-1", "detected-lane-2", "detected-lane-3"]
    number_1 = next(t for t in ax.texts if t.get_gid() == "detected-number-1")
    assert number_1.get_position()[0] == pytest.approx(40.0)   # lane 1 is the leftmost


def test_legend_names_both_sources(gel, reference):
    ax = plot_lanes(gel, detected=[60.0], reference=reference)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Image Lab" in label for label in labels)
    assert any("detected" in label for label in labels)


def test_can_draw_into_an_existing_axes(gel):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    assert plot_lanes(gel, ax=ax) is ax


def test_draws_the_gel_area(gel):
    from gelanalyzer.border import GelArea
    ax = plot_lanes(gel, area=GelArea(left=10, top=5, right=190, bottom=95))
    assert gids(ax, "gel-area") == ["gel-area"]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("gel area" in label for label in labels)


def test_draws_traced_lanes_with_their_evidence(gel):
    from gelanalyzer.tracing import BandPoint, Lanes
    lanes = Lanes.straight([60.0, 140.0], y_start=10, y_end=90)
    lanes.points = [BandPoint(lane=0, y=40, x=61, width=30, strength=10)]
    ax = plot_lanes(gel, traced=lanes)
    assert gids(ax, "traced-lane-") == ["traced-lane-1", "traced-lane-2"]
    assert texts(ax, "traced-number-") == ["1", "2"]
    assert gids(ax, "band-points") == ["band-points"]
    # broadness indicator: a width bar at the top and bottom of every lane
    assert len(gids(ax, "traced-width-")) == 4
