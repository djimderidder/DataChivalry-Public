import pytest

from thtdecay.settings import ThtSettings


def test_defaults_match_the_original_scripts():
    s = ThtSettings()
    assert (s.gain, s.equilibration_s, s.initial_window_s, s.final_window_s) == (3, 600, 3600, 3600)
    assert (s.smoothing_window, s.smoothing_poly) == (51, 3)
    assert not s.smooth_initial and not s.smooth_final


@pytest.mark.parametrize("mode,initial,final", [
    ("off", False, False), ("initial", True, False), ("final", False, True), ("both", True, True)])
def test_old_mode_mapping(mode, initial, final):
    s = ThtSettings.from_mode(mode)
    assert (s.smooth_initial, s.smooth_final) == (initial, final)


def test_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        ThtSettings.from_mode("smooth")


@pytest.mark.parametrize("kwargs", [
    {"smoothing_window": 50},                       # must be odd
    {"smoothing_window": 5, "smoothing_poly": 5},   # poly < window
    {"initial_window_s": 0},
    {"max_time_s": 300},                            # before the equilibration cut
    {"normalize_points": 0},
])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        ThtSettings(**kwargs)


def test_with_changes():
    s = ThtSettings().with_changes(gain=1)
    assert s.gain == 1 and ThtSettings().gain == 3
