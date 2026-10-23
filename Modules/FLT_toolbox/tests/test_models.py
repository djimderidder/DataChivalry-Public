import numpy as np
import pytest

from fluotime.models import multi_exp_kernel, reconvolution_model


@pytest.fixture
def delta_irf():
    t = np.arange(2000) * 0.01
    irf = np.zeros_like(t)
    irf[100] = 1.0  # a single-bin IRF at t = 1 ns
    return t, irf


def test_kernel_single_exponential():
    t = np.linspace(0, 10, 101)
    k = multi_exp_kernel(t, amplitudes=[2.0], taus=[3.0])
    np.testing.assert_allclose(k, 2.0 * np.exp(-t / 3.0))


def test_kernel_starts_at_first_time_point():
    t = np.linspace(5, 15, 101)
    k = multi_exp_kernel(t, amplitudes=[1.0], taus=[2.0])
    assert k[0] == pytest.approx(1.0)


def test_kernel_sums_components():
    t = np.linspace(0, 10, 101)
    k = multi_exp_kernel(t, [1.0, 3.0], [2.0, 0.5])
    np.testing.assert_allclose(k, np.exp(-t / 2) + 3 * np.exp(-t / 0.5))


def test_kernel_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        multi_exp_kernel(np.arange(5.0), [1.0, 2.0], [1.0])


def test_reconvolution_with_delta_irf_gives_shifted_exponential(delta_irf):
    t, irf = delta_irf
    dt = t[1] - t[0]
    y = reconvolution_model(t, irf, amplitudes=[1000.0], taus=[2.0])

    expected = np.zeros_like(t)
    after = t >= t[100]
    expected[after] = 1000.0 * dt * np.exp(-(t[after] - t[100]) / 2.0)
    np.testing.assert_allclose(y, expected, atol=1e-9)


def test_reconvolution_area_is_amplitude_times_tau(delta_irf):
    """With a normalised IRF the total counts in the decay equal sum(A_i * tau_i)."""
    t, irf = delta_irf
    y = reconvolution_model(t, irf, amplitudes=[500.0, 200.0], taus=[1.5, 0.5])
    assert y.sum() == pytest.approx(500 * 1.5 + 200 * 0.5, rel=1e-2)


def test_background_and_scatter_terms(delta_irf):
    t, irf = delta_irf
    base = reconvolution_model(t, irf, [100.0], [1.0])
    y = reconvolution_model(t, irf, [100.0], [1.0], background=3.0, scatter=50.0)
    np.testing.assert_allclose(y - base, 3.0 + 50.0 * irf)


def test_shift_delays_the_decay(delta_irf):
    t, irf = delta_irf
    y0 = reconvolution_model(t, irf, [100.0], [1.0])
    y1 = reconvolution_model(t, irf, [100.0], [1.0], shift=0.5)
    assert t[np.argmax(y1)] - t[np.argmax(y0)] == pytest.approx(0.5)
