"""Integration tests on real gels from the lab.

The gel files are NOT in git (unpublished data, and large). Put them in
tests/data/real/ to run these; without them, the tests are skipped.
Run only these with:  pytest -m real_data
"""

from pathlib import Path

import numpy as np
import pytest

from gelanalyzer.border import find_gel_area
from gelanalyzer.imagelab import read_imagelab_analysis
from gelanalyzer.io import load_gel
from gelanalyzer.lanes import detect_lanes
from gelanalyzer.tracing import trace_lanes

REAL = Path(__file__).parent / "data" / "real"

# What we know about each gel (from the product owner + Image Lab).
# gel_area (left, top, right, bottom) was read by eye from zoomed crops with a pixel grid
# -> loose tolerance (30 px = 1.7 mm).
GELS = {
    "2025-11-10_15h03m52s.scn": dict(shape=(1561, 1882), application="Coomassie Blue",
                                     n_lanes=15, ladder_lane=4, gel_area=(95, 50, 1828, 1522)),   # bottom: the straight line (confirmed by PO)
    "2025-11-10_11h58m54s.scn": dict(shape=(1311, 1680), application="Stain Free Gel",
                                     n_lanes=15, ladder_lane=5, gel_area=(48, 0, 1620, 1258)),  # ladder lane 5 (confirmed)    # top cropped; bottom just below loading-front smear
    "2025-11-13_07h03m48s.scn": dict(shape=(1450, 1796), application="Coomassie Blue",
                                     n_lanes=15, ladder_lane=10,    # damaged gel
                                     gel_area=(42, 20, 1750, 1420)),   # bottom damaged: 1400..1440
}


@pytest.mark.real_data
@pytest.mark.parametrize("name", GELS)
def test_real_scn_loads(name):
    path = REAL / name
    if not path.exists():
        pytest.skip(f"{name} not present in tests/data/real/")
    gel = load_gel(path)
    expected = GELS[name]
    assert gel.pixels.shape == expected["shape"]
    assert gel.metadata["application"] == expected["application"]
    assert gel.pixel_size_mm == pytest.approx((0.0564, 0.0564), abs=1e-3)


@pytest.mark.real_data
@pytest.mark.parametrize("name", GELS)
def test_real_imagelab_reference(name):
    path = REAL / name
    if not path.exists():
        pytest.skip(f"{name} not present in tests/data/real/")
    expected = GELS[name]
    analysis = read_imagelab_analysis(path)
    assert len(analysis.lanes) == expected["n_lanes"]
    assert analysis.standard_lane.number == expected["ladder_lane"]
    assert analysis.standard_kda == [250, 150, 100, 75, 50, 37, 25, 20, 15, 10]
    assert len(analysis.standard_lane.band_ys) == 10
    centres = [lane.x_center for lane in analysis.lanes]
    assert centres == sorted(centres)                     # numbered left to right


@pytest.mark.real_data
@pytest.mark.parametrize("name", GELS)
def test_real_gel_area(name):
    path = REAL / name
    if not path.exists():
        pytest.skip(f"{name} not present in tests/data/real/")
    area = find_gel_area(load_gel(path).pixels)
    found = (area.left, area.top, area.right, area.bottom)
    assert found == pytest.approx(GELS[name]["gel_area"], abs=30)


@pytest.mark.real_data
@pytest.mark.parametrize("name", GELS)
def test_real_lanes_match_image_lab(name):
    """US-11 acceptance: 15/15 lanes, each within 25 px (a quarter lane) of Image Lab,
    measured at the top of Image Lab's lane frame (just below the wells)."""
    path = REAL / name
    if not path.exists():
        pytest.skip(f"{name} not present in tests/data/real/")
    expected = GELS[name]
    lanes = detect_lanes(load_gel(path).pixels, n_lanes=expected["n_lanes"])
    reference = [lane.x_top for lane in read_imagelab_analysis(path).lanes]
    assert lanes == pytest.approx(reference, abs=25)


# US-12 ratchet: mean distance (px) to Image Lab's lanes at the bands the user picked.
# Image Lab's hand-drawn lines are within 2-6 px of the real band centres there.
# Thresholds = today's result + a little slack: future changes may not make it worse.
TRACING_RATCHET = {
    "2025-11-10_15h03m52s.scn": 8.0,    # today 6.3  (straight lanes: 29.1)
    "2025-11-10_11h58m54s.scn": 19.0,   # today 17.0 (straight lanes: 28.4) - few clear bands
    "2025-11-13_07h03m48s.scn": 9.0,    # today 7.1  (straight lanes: 16.3)
}


@pytest.mark.real_data
@pytest.mark.parametrize("name", GELS)
def test_real_traced_lanes_follow_the_bands(name):
    path = REAL / name
    if not path.exists():
        pytest.skip(f"{name} not present in tests/data/real/")
    lanes = trace_lanes(load_gel(path).pixels, n_lanes=GELS[name]["n_lanes"])
    traced, straight = [], []
    for k, lane in enumerate(read_imagelab_analysis(path).lanes):
        for y in lane.band_ys:
            traced.append(abs(lanes.x_at(y)[k] - lane.x_at(y)))
            straight.append(abs(lanes.comb_centers[k] - lane.x_at(y)))
    assert np.mean(traced) < TRACING_RATCHET[name]
    assert np.mean(traced) < 0.7 * np.mean(straight)          # clearly better than straight lanes


@pytest.mark.real_data
@pytest.mark.parametrize("name", GELS)
def test_real_ladder(name, save_plot):
    """US-13 acceptance: ladder centre line on the Image Lab ladder bands (mean < 5 px,
    max < 10), at least 7 of 10 bands, lane end below the lowest ladder band."""
    from gelanalyzer.ladder import find_ladder
    from gelanalyzer.plotting import plot_ladder

    path = REAL / name
    if not path.exists():
        pytest.skip(f"{name} not present in tests/data/real/")
    expected = GELS[name]
    gel = load_gel(path, config={"n_lanes": expected["n_lanes"], "ladder_lanes": [expected["ladder_lane"]],
                                 "ladder": "Precision Plus Unstained"})
    ladder = find_ladder(gel)
    reference = read_imagelab_analysis(path).lanes[expected["ladder_lane"] - 1]
    ref_ys = reference.band_ys
    save_plot(f"ladder_{path.stem}", plot_ladder(gel, ladder, reference_ys=ref_ys,
                                                 reference_xs=[reference.x_at(y) for y in ref_ys]))

    errors = [abs(ladder.x_at(y) - reference.x_at(y)) for y in ref_ys]
    assert np.mean(errors) < 5 and np.max(errors) < 10
    assert len(ladder.bands) >= 7
    assert max(ref_ys) < ladder.y_end <= find_gel_area(gel.pixels).bottom
    mid = (ladder.y_start + ladder.y_end) / 2
    assert 0.4 * ladder.pitch < ladder.width_at(mid) < 1.0 * ladder.pitch


@pytest.mark.real_data
@pytest.mark.parametrize("name", GELS)
def test_real_ladder_bands(name, save_plot):
    """US-14 acceptance: all 10 ladder bands found, sizes assigned, thick bands (75/50/25)
    are the strongest, heights close to the bands the user picked in Image Lab."""
    from gelanalyzer.ladder import find_ladder, identify_ladder_bands
    from gelanalyzer.plotting import plot_ladder

    path = REAL / name
    if not path.exists():
        pytest.skip(f"{name} not present in tests/data/real/")
    expected = GELS[name]
    gel = load_gel(path, config={"n_lanes": expected["n_lanes"], "ladder_lanes": [expected["ladder_lane"]],
                                 "ladder": "Precision Plus Unstained"})
    ladder = find_ladder(gel)
    result = identify_ladder_bands(gel, ladder)
    reference = read_imagelab_analysis(path).lanes[expected["ladder_lane"] - 1]
    save_plot(f"ladder_bands_{path.stem}",
              plot_ladder(gel, ladder, reference_ys=reference.band_ys,
                          reference_xs=[reference.x_at(y) for y in reference.band_ys], identification=result))

    assert result.complete and result.thick_bands_ok
    assert [b.kda for b in result.bands] == [250, 150, 100, 75, 50, 37, 25, 20, 15, 10]
    diff = np.abs(np.array([b.y for b in result.bands]) - np.array(reference.band_ys))
    assert diff.mean() < 4
    assert diff.max() < 16       # 13-11: 10 kDa Image Lab mark sits ~15 px above the visible band; ours is on it
