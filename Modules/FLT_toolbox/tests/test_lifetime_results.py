import numpy as np
import pytest

from fluotime.lifetime.results import fractions, mean_lifetime


def test_amplitude_fractions():
    np.testing.assert_allclose(fractions([3, 1], [4, 1], kind="amplitude"), [0.75, 0.25])


def test_intensity_fractions():
    # a_i * tau_i = 12, 1
    np.testing.assert_allclose(fractions([3, 1], [4, 1], kind="intensity"), [12 / 13, 1 / 13])


def test_fractions_zero_total():
    np.testing.assert_allclose(fractions([0, 0], [4, 1]), [0, 0])


def test_fractions_bad_kind():
    with pytest.raises(ValueError, match="kind"):
        fractions([1], [1], kind="area")


def test_amplitude_weighted_mean_lifetime():
    assert mean_lifetime([1, 1], [4, 2], kind="amplitude") == pytest.approx(3.0)


def test_intensity_weighted_mean_lifetime():
    # sum a tau^2 / sum a tau = (16 + 4) / (4 + 2)
    assert mean_lifetime([1, 1], [4, 2], kind="intensity") == pytest.approx(20 / 6)
