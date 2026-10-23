import numpy as np
import pytest
from scipy import stats

from atpase import AtpaseSettings, atpase_rate, auto_stop, fit_window


@pytest.fixture
def trace():
    t = np.arange(0, 7200, 16.0)
    y = np.maximum(-0.1 - 0.01 * t / 60, -1.2)  # linear decrease, then flat (NADH depleted)
    return t, y


def test_rate_matches_the_original_formula(trace):
    """Old: linregress(Time[s:e]/60, Data[s:e]); rate = -1/(eps * conc * 1e-6) * slope."""
    t, y = trace
    result = atpase_rate(t, y, 10, 200, enzyme_uM=4.0)
    old_slope = stats.linregress(t[10:200] / 60, y[10:200]).slope
    assert result.rate == pytest.approx(-old_slope / (6220 * 4.0 * 1e-6))
    assert result.slope == pytest.approx(-0.01)


def test_path_length_scales_the_rate(trace):
    t, y = trace
    full = atpase_rate(t, y, 0, 200, 4.0).rate
    half = atpase_rate(t, y, 0, 200, 4.0, AtpaseSettings(path_length_cm=0.5)).rate
    assert half == pytest.approx(2 * full)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), "-"])
def test_invalid_enzyme_concentration(trace, bad):
    t, y = trace
    with pytest.raises(ValueError):
        atpase_rate(t, y, 0, 100, bad)


def test_auto_stop_matches_the_original_rule(trace):
    """Old: first point <= 0.95 * mean(last 5)."""
    t, y = trace
    old = np.where(y <= 0.95 * np.mean(y[-5:]))[0][0]
    assert auto_stop(y) == old


def test_auto_stop_searches_after_start(trace):
    _, y = trace
    y = y.copy()
    y[3] = -5.0                       # an early outlier below the threshold
    assert auto_stop(y) == 3          # old behaviour from the start...
    assert auto_stop(y, start=10) > 10  # ...but never before the chosen start


def test_auto_stop_without_depletion_uses_the_whole_trace():
    y = np.linspace(0, -1, 50)
    assert auto_stop(y, AtpaseSettings(auto_threshold=1.0, auto_n_last=1)) == 49


def test_each_bound_is_taken_from_the_config_when_filled_in(trace):
    _, y = trace
    auto = auto_stop(y)
    assert fit_window(y, "-", "-") == (0, auto, "auto/auto")
    assert fit_window(y, 150, "-") == (150, auto, "config/auto")
    assert fit_window(y, float("nan"), 300.0) == (0, 300, "auto/config")
    assert fit_window(y, "20", "60") == (20, 60, "config/config")


def test_invalid_window(trace):
    _, y = trace
    with pytest.raises(ValueError, match="window"):
        fit_window(y, 300, 100)
