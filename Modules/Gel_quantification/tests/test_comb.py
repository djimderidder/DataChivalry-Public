"""US-11 (part 1): find the comb wells.

The comb is machined, so its wells are EXACTLY evenly spaced. Instead of
looking for n separate peaks, we fit a grid: first well position + pitch.
Wells are wider than the teeth between them, which tells wells from teeth
whatever the stain (wells lighter in Coomassie, darker in stain-free).
"""

import pytest

from gelanalyzer.border import GelArea
from gelanalyzer.comb import find_comb
from synthetic import make_gel

COOMASSIE_LIKE = dict(n_lanes=10, width=900, height=500, margin=(40, 30, 50, 35), comb=True, seed=3)
STAIN_FREE_LIKE = dict(n_lanes=10, width=900, height=500, margin=(40, 0, 50, 35), comb=True,
                       well_level=0.12, well_walls=0.0, seed=4)


@pytest.mark.parametrize("kind", [COOMASSIE_LIKE, STAIN_FREE_LIKE], ids=["coomassie", "stain-free"])
def test_finds_every_well(kind):
    gel = make_gel(**kind)
    comb = find_comb(gel.image, n_lanes=10)
    assert comb.centers == pytest.approx(gel.lane_centers, abs=3)
    assert comb.pitch == pytest.approx(gel.lane_pitch, abs=1)


def test_returns_exactly_n_sorted_wells():
    comb = find_comb(make_gel(**COOMASSIE_LIKE).image, n_lanes=10)
    assert len(comb.centers) == 10
    assert comb.centers == sorted(comb.centers)


def test_noisy_gel():
    gel = make_gel(**{**COOMASSIE_LIKE, "noise": 0.06})
    comb = find_comb(gel.image, n_lanes=10)
    assert comb.centers == pytest.approx(gel.lane_centers, abs=4)


def test_uses_a_given_gel_area():
    gel = make_gel(**COOMASSIE_LIKE)
    area = GelArea(*gel.gel_area)
    comb = find_comb(gel.image, n_lanes=10, area=area)
    assert comb.centers == pytest.approx(gel.lane_centers, abs=3)


def test_needs_at_least_two_lanes():
    with pytest.raises(ValueError, match="n_lanes"):
        find_comb(make_gel(**COOMASSIE_LIKE).image, n_lanes=1)


@pytest.mark.parametrize("kind", [COOMASSIE_LIKE, STAIN_FREE_LIKE], ids=["coomassie", "stain-free"])
def test_knows_where_the_wells_end(kind):
    """Lanes start at the well floors, so tracing needs to know where they are."""
    gel = make_gel(**kind)
    comb = find_comb(gel.image, n_lanes=10)
    assert comb.bottom == pytest.approx(gel.comb_bottom, abs=4)
