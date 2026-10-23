import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from fluostar.selection import WellSelection, select_trace_window

WELLS = ["A9", "B9", "C9", "H10", "H11"]


def test_click_cycle_matches_the_original_selector():
    """click: off -> selected -> base (light blue) -> fibril (dark blue) -> off."""
    s = WellSelection(WELLS)
    s.click("A9")
    assert s.state("A9") == "selected"
    s.click("A9")
    assert s.state("A9") == "base"
    s.click("A9")
    assert s.state("A9") == "fibril"
    s.click("A9")
    assert s.state("A9") == "off"


def test_only_one_base_and_one_fibril_control():
    s = WellSelection(WELLS)
    for w in ("H11", "H11"):          # H11 -> selected -> base
        s.click(w)
    for w in ("H10", "H10"):          # H10 -> selected -> base taken -> fibril
        s.click(w)
    assert s.base == "H11" and s.fibril == "H10"
    s.click("C9")
    s.click("C9")                     # both controls taken: back to off
    assert s.state("C9") == "off"


def test_roi_in_plate_order():
    s = WellSelection(WELLS)
    for w in ("C9", "A9", "H11", "H11", "H10", "H10"):
        s.click(w)
    roi = s.roi()
    assert list(roi["well"]) == ["A9", "C9"]
    assert set(roi["nBase"]) == {"H11"}
    assert set(roi["nFibril"]) == {"H10"}


def test_roi_without_controls_gives_a_clear_error():
    s = WellSelection(WELLS)
    s.click("A9")
    with pytest.raises(ValueError, match="control"):
        s.roi()


def test_unknown_well():
    with pytest.raises(KeyError):
        WellSelection(WELLS).click("Z9")


def test_trace_window_returns_nearest_indices_of_the_guesses():
    """Without an interactive backend the window closes at once: the guesses are returned."""
    x = np.arange(100) * 16.0
    start, end = select_trace_window(x, -x, start_guess=161, end_guess=800)
    assert (start, end) == (10, 50)
