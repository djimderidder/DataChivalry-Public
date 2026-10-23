import numpy as np

from fluotime.synthetic import gaussian_irf, make_synthetic_decay, make_synthetic_spectrum


def test_gaussian_irf_is_normalised_and_centred():
    t = np.arange(400) * 0.05
    irf = gaussian_irf(t, center=5.0, fwhm=0.2)
    assert irf.sum() == np.float64(1.0) or abs(irf.sum() - 1) < 1e-12
    assert t[np.argmax(irf)] == 5.0


def test_synthetic_decay_is_reproducible_with_seed():
    a = make_synthetic_decay("x", amplitudes=(1e5,), taus=(4.0,), seed=7)
    b = make_synthetic_decay("x", amplitudes=(1e5,), taus=(4.0,), seed=7)
    np.testing.assert_array_equal(a.counts, b.counts)
    assert a.irf is not None and a.irf.max() > 0


def test_synthetic_spectrum_channels():
    s = make_synthetic_spectrum(r=0.1, g=0.9)
    assert set(s.channels) == {"VV", "VH", "HH", "HV"}
