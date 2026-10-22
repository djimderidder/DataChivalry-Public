"""US-10: find the gel area (the gel itself, without background around it).

Idea: the gel edge shows up as the OUTERMOST strong line on each side. A line
only counts as the border if the area beyond it is *lower* than inside
(background = least signal). That rejects e.g. the well floors when the image
is cropped at the gel top: above them are the wells, not background.
"""

import pytest

from gelanalyzer.border import GelArea, find_gel_area
from synthetic import make_gel

TOL = 3  # px, on synthetic gels we know the truth exactly


def assert_area(found: GelArea, truth, tol=TOL):
    assert (found.left, found.top, found.right, found.bottom) == pytest.approx(truth, abs=tol)


def test_gel_with_background_on_all_sides():
    gel = make_gel(n_lanes=8, width=700, height=450, margin=(40, 30, 50, 35), seed=1)
    assert_area(find_gel_area(gel.image), gel.gel_area)


def test_comb_and_bands_do_not_fool_it():
    gel = make_gel(n_lanes=10, width=900, height=500, margin=(40, 30, 50, 35), comb=True, seed=3)
    assert_area(find_gel_area(gel.image), gel.gel_area)


def test_cropped_top_with_dark_wells_is_not_mistaken_for_an_edge():
    # stain-free-like: no background above the gel, wells MORE intense than the gel
    gel = make_gel(n_lanes=10, width=900, height=500, margin=(40, 0, 50, 35), comb=True,
                   well_level=0.12, well_walls=0.0, seed=4)
    area = find_gel_area(gel.image)
    assert area.top == 0
    assert_area(area, gel.gel_area)


def test_fully_cropped_gel_is_the_whole_image():
    gel = make_gel(n_lanes=6, seed=1)          # no margin at all
    assert_area(find_gel_area(gel.image), (0, 0, 600, 400))


def test_area_size():
    area = GelArea(left=10, top=20, right=110, bottom=220)
    assert (area.width, area.height) == (100, 200)
