"""US-12: follow the lanes down the gel.

Domain rules from the product owner, each one a test:
  * lanes rarely run straight down: they shift, fan out, sometimes curve
  * neighbouring lanes move together; lanes never overlap
  * spread is uneven, so lane width changes from top to bottom
  * where a lane has no clear bands, its neighbours tell where it is
"""

import numpy as np
import pytest

from gelanalyzer.tracing import Lanes, trace_lanes
from synthetic import make_gel

BASE = dict(n_lanes=10, width=900, height=600, margin=(40, 30, 50, 35), comb=True,
            bands_per_lane=6, seed=7)


def heights(gel, n=8):
    """Heights from just below the wells to near the gel bottom."""
    return np.linspace(gel.comb_bottom + 20, gel.gel_area[3] - 30, n)


def max_error(gel, lanes, lanes_to_check=None):
    ks = range(len(gel.lane_centers)) if lanes_to_check is None else lanes_to_check
    return max(abs(lanes.x_at(y)[k] - gel.lane_x_at(k, y)) for y in heights(gel) for k in ks)


def test_straight_gel_gives_straight_lanes():
    gel = make_gel(**BASE)
    assert max_error(gel, trace_lanes(gel.image, n_lanes=10)) < 4


@pytest.mark.parametrize("deformation", [dict(shift=30), dict(fan=45), dict(fan=40, bend=12), dict(shift=-20, fan=30)],
                         ids=["shift", "fan", "fan+bend", "shift+fan"])
def test_follows_lanes_that_do_not_run_straight(deformation):
    gel = make_gel(**BASE, **deformation)
    assert max_error(gel, trace_lanes(gel.image, n_lanes=10)) < 6


def test_empty_lanes_follow_their_neighbours():
    gel = make_gel(**BASE, fan=45, empty_lanes=(3, 4))
    lanes = trace_lanes(gel.image, n_lanes=10)
    assert max_error(gel, lanes, lanes_to_check=[3, 4]) < 8


def test_lanes_start_at_the_comb():
    gel = make_gel(**BASE, fan=45)
    lanes = trace_lanes(gel.image, n_lanes=10)
    assert lanes.x_at(gel.comb_bottom) == pytest.approx(gel.lane_centers, abs=3)
    assert lanes.x_at(gel.gel_area[1]) == pytest.approx(gel.lane_centers, abs=3)   # in the wells: no deformation


def test_lanes_never_overlap_and_width_follows_the_spread():
    gel = make_gel(**BASE, fan=45)
    lanes = trace_lanes(gel.image, n_lanes=10)
    for y in heights(gel):
        edges = lanes.boundaries_at(y)
        assert len(edges) == 11
        assert np.all(np.diff(edges) > 0)                       # ordered: no overlap
    top, bottom = heights(gel)[0], heights(gel)[-1]
    assert lanes.width_at(bottom).mean() > lanes.width_at(top).mean()   # fanning out -> wider lanes


def test_x_at_accepts_many_heights():
    gel = make_gel(**BASE)
    lanes = trace_lanes(gel.image, n_lanes=10)
    ys = heights(gel)
    assert lanes.x_at(ys).shape == (len(ys), 10)


def test_straight_lanes_from_comb_only():
    lanes = Lanes.straight([100.0, 200.0, 300.0], y_start=50, y_end=450)
    assert lanes.x_at(300) == pytest.approx([100, 200, 300])
    assert lanes.boundaries_at(300) == pytest.approx([50, 150, 250, 350])
    assert lanes.width_at(300) == pytest.approx([100, 100, 100])
