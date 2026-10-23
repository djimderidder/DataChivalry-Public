import numpy as np
import pytest

from fluotime.irf import irf_background, preprocess_irf, shift_irf


@pytest.fixture
def raw_irf():
    t = np.arange(200) * 0.05
    rng = np.random.default_rng(0)
    peak = 1e4 * np.exp(-0.5 * ((t - 3.0) / 0.08) ** 2)
    return t, rng.poisson(peak + 3.0).astype(float)


# --- background / preprocessing ---------------------------------------------

def test_mean_background_matches_original_constant_model_fit(raw_irf):
    """The original scripts fitted lmfit's ConstantModel to the whole IRF.
    That fit is exactly the mean of the trace; keep that behaviour."""
    from lmfit.models import ConstantModel

    t, irf = raw_irf
    model = ConstantModel()
    original = model.fit(irf, model.guess(irf, x=t), x=t).params["c"].value

    assert irf_background(irf, method="mean") == pytest.approx(original, rel=1e-6)


def test_percentile_background(raw_irf):
    _, irf = raw_irf
    assert irf_background(irf, method="percentile", percentile=5) == pytest.approx(
        np.percentile(irf, 5))


@pytest.mark.parametrize("method", ["mean_threshold", "percentile_subtract", "none"])
def test_preprocess_is_normalised_and_non_negative(raw_irf, method):
    _, irf = raw_irf
    out = preprocess_irf(irf, method=method)
    assert out.sum() == pytest.approx(1.0)
    assert np.all(out >= 0)


def test_mean_threshold_zeroes_bins_below_mean_without_subtracting(raw_irf):
    _, irf = raw_irf
    out = preprocess_irf(irf, method="mean_threshold")
    keep = irf >= irf.mean()
    assert np.all(out[~keep] == 0)
    # kept bins are proportional to the raw counts (no background subtracted)
    np.testing.assert_allclose(out[keep], irf[keep] / irf[keep].sum())


def test_percentile_subtract_removes_baseline(raw_irf):
    _, irf = raw_irf
    out = preprocess_irf(irf, method="percentile_subtract", percentile=50)
    assert np.count_nonzero(out) < 0.5 * irf.size


def test_preprocess_unknown_method(raw_irf):
    _, irf = raw_irf
    with pytest.raises(ValueError, match="method"):
        preprocess_irf(irf, method="magic")


def test_preprocess_empty_irf_raises():
    with pytest.raises(ValueError, match="empty"):
        preprocess_irf(np.zeros(10), method="none")


def test_preprocess_does_not_modify_input(raw_irf):
    _, irf = raw_irf
    copy = irf.copy()
    preprocess_irf(irf, method="percentile_subtract")
    np.testing.assert_array_equal(irf, copy)


# --- shifting ---------------------------------------------------------------

def test_zero_shift_is_identity():
    t = np.arange(10) * 0.1
    irf = np.array([0, 0, 1, 3, 1, 0, 0, 0, 0, 0], float)
    np.testing.assert_allclose(shift_irf(t, irf, 0.0), irf)


def test_whole_bin_shift_moves_peak_and_fills_zero():
    t = np.arange(10) * 0.1
    irf = np.array([0, 0, 1, 3, 1, 0, 0, 0, 0, 2], float)
    out = shift_irf(t, irf, 0.2)  # +2 bins: later in time
    np.testing.assert_allclose(out, [0, 0, 0, 0, 1, 3, 1, 0, 0, 0], atol=1e-12)


def test_half_bin_shift_interpolates_linearly():
    t = np.arange(5) * 1.0
    irf = np.array([0, 2, 0, 0, 0], float)
    out = shift_irf(t, irf, 0.5)
    np.testing.assert_allclose(out, [0, 1, 1, 0, 0])


def test_shift_onto_a_different_time_grid():
    t_irf = np.arange(10) * 0.1
    irf = np.zeros(10)
    irf[3] = 1.0
    t_target = np.arange(20) * 0.05
    out = shift_irf(t_target, irf, 0.0, t_irf=t_irf)
    assert t_target[np.argmax(out)] == pytest.approx(0.3)
