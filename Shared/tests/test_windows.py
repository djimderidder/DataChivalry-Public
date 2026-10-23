import math

import numpy as np
import pytest
from scipy import stats

from fluostar.windows import Window, fit_linear, last_seconds, window_mean


@pytest.fixture
def t():
    return np.arange(0, 3600 * 3 + 1, 60.0)  # every minute for 3 h


def test_default_window_is_everything(t):
    assert Window().mask(t).all()


def test_until_matches_old_absolute_time_mask(t):
    """ThT initial window: the old code used `Time_s <= window_s` on absolute time."""
    w = Window(end_s=3600)
    np.testing.assert_array_equal(w.mask(t), t <= 3600)


def test_last_seconds_matches_old_endpoint_mask(t):
    w = last_seconds(t, 3600)
    np.testing.assert_array_equal(w.mask(t), t >= t.max() - 3600)


def test_from_indices_uses_python_slice_semantics(t):
    """ATPase: the old code fitted Time[starti:endi] (endi excluded)."""
    w = Window.from_indices(t, 5, 20)
    assert w.mask(t).sum() == 15
    assert w.indices(t) == (5, 20)


def test_from_indices_validation(t):
    with pytest.raises(ValueError):
        Window.from_indices(t, 10, 10)
    with pytest.raises(ValueError):
        Window.from_indices(t, 0, len(t) + 1)


def test_start_after_end():
    with pytest.raises(ValueError, match="start"):
        Window(10, 5)


def test_fit_linear_matches_linregress_in_minutes(t):
    rng = np.random.default_rng(0)
    y = 2.0 - 0.01 * t / 60 + rng.normal(0, 0.01, t.size)
    w = Window(600, 3600)

    fit = fit_linear(t, y, w, time_unit_s=60)
    ref = stats.linregress(t[w.mask(t)] / 60, y[w.mask(t)])

    assert fit.slope == pytest.approx(ref.slope)
    assert fit.intercept == pytest.approx(ref.intercept)
    assert fit.r_value == pytest.approx(ref.rvalue)
    assert fit.n_points == w.mask(t).sum()


def test_fit_slope_scales_with_time_unit(t):
    y = 1 - 0.5 * t / 3600
    per_min = fit_linear(t, y, Window(), time_unit_s=60).slope
    per_hour = fit_linear(t, y, Window(), time_unit_s=3600).slope
    assert per_hour == pytest.approx(per_min * 60)
    assert per_hour == pytest.approx(-0.5)


def test_fit_line_evaluates_the_fit(t):
    y = 3 + 2 * t / 60
    fit = fit_linear(t, y, Window(), time_unit_s=60)
    np.testing.assert_allclose(fit.line(t), y)


def test_fit_ignores_nan_points(t):
    y = 1 - 0.01 * t / 60
    y[3] = np.nan
    fit = fit_linear(t, y, Window())
    assert fit.slope == pytest.approx(-0.01)
    assert fit.n_points == t.size - 1


def test_fit_needs_two_points(t):
    with pytest.raises(ValueError, match="2"):
        fit_linear(t, t, Window(0, 0))


def test_window_mean(t):
    y = np.where(t > 3600, 5.0, 1.0)
    assert window_mean(t, y, Window(3700, math.inf)) == pytest.approx(5.0)
