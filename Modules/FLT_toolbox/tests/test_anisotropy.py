import numpy as np
import pytest

from fluotime.anisotropy import (
    AnisotropyConfig,
    analyse_anisotropy,
    anisotropy,
    anisotropy_spectrum,
    g_factor,
    smooth_spectrum,
    subtract_blank,
    window_average,
)
from fluotime.io import Spectrum


# --- formulas -----------------------------------------------------------------

def test_g_factor():
    np.testing.assert_allclose(g_factor(np.array([9.0, 4.0]), np.array([10.0, 4.0])), [0.9, 1.0])


def test_g_factor_zero_hh_gives_nan():
    assert np.isnan(g_factor(np.array([1.0]), np.array([0.0]))[0])


def test_anisotropy_formula():
    # VV = I(1+2r)/3, VH = I(1-r)/(3G)  ->  r recovered for any I, G
    r_true, g, intensity = 0.2, 0.8, 300.0
    vv = intensity * (1 + 2 * r_true) / 3
    vh = intensity * (1 - r_true) / 3 / g
    assert anisotropy(np.array([vv]), np.array([vh]), g)[0] == pytest.approx(r_true)


def test_anisotropy_limits():
    assert anisotropy(np.array([1.0]), np.array([0.0]), 1.0)[0] == pytest.approx(1.0)
    assert anisotropy(np.array([1.0]), np.array([1.0]), 1.0)[0] == pytest.approx(0.0)


# --- processing steps ---------------------------------------------------------

def _spec(vv, vh, wl=None):
    wl = np.arange(len(vv), dtype=float) if wl is None else wl
    return Spectrum(wavelength=wl, channels={"VV": np.asarray(vv, float), "VH": np.asarray(vh, float)})


def test_subtract_blank_interpolates_and_does_not_mutate():
    sample = _spec([10, 10, 10], [5, 5, 5], wl=np.array([500.0, 501, 502]))
    blank = _spec([1, 3], [0, 2], wl=np.array([500.0, 502]))

    out = subtract_blank(sample, blank, channels=("VV", "VH"))

    np.testing.assert_allclose(out["VV"], [9, 8, 7])
    np.testing.assert_allclose(out["VH"], [5, 4, 3])
    np.testing.assert_allclose(sample["VV"], [10, 10, 10])  # input untouched


def test_subtract_blank_missing_channel():
    sample = _spec([1, 1], [1, 1])
    blank = _spec([0, 0], [0, 0])
    with pytest.raises(KeyError, match="HH"):
        subtract_blank(sample, blank, channels=("VV", "HH"))


def test_smoothing_reduces_noise_and_keeps_wavelength():
    rng = np.random.default_rng(0)
    s = _spec(100 + rng.normal(0, 5, 200), 50 + rng.normal(0, 5, 200))
    out = smooth_spectrum(s, sigma=3)
    assert out["VV"].std() < s["VV"].std()
    np.testing.assert_array_equal(out.wavelength, s.wavelength)


def test_smoothing_blank_too_equals_smoothing_the_difference(spectrum_pair):
    sample, blank = spectrum_pair
    a = anisotropy_spectrum(sample, blank, smooth_sigma=2, smooth_blank=True)
    diff = subtract_blank(sample, blank, channels=("VV", "VH"))
    b = anisotropy_spectrum(diff, None, smooth_sigma=2)
    np.testing.assert_allclose(a["r"], b["r"], rtol=1e-9)


# --- full spectrum ------------------------------------------------------------

def test_anisotropy_spectrum_recovers_known_values(spectrum_pair):
    sample, blank = spectrum_pair
    df = anisotropy_spectrum(sample, blank, smooth_sigma=2)

    mean_r, _, n = window_average(df, center=520, half_width=5)
    assert n > 0
    assert mean_r == pytest.approx(0.15, abs=0.01)
    assert df["G"].loc[(df.wavelength > 510) & (df.wavelength < 530)].mean() == pytest.approx(0.9, abs=0.02)


def test_without_blank_subtraction_r_is_biased(spectrum_pair):
    sample, blank = spectrum_pair
    with_blank = window_average(anisotropy_spectrum(sample, blank), 520, 5)[0]
    without = window_average(anisotropy_spectrum(sample, None), 520, 5)[0]
    assert abs(without - 0.15) > abs(with_blank - 0.15)


def test_anisotropy_spectrum_columns(spectrum_pair):
    sample, blank = spectrum_pair
    df = anisotropy_spectrum(sample, blank, smooth_sigma=None)
    assert list(df.columns) == ["wavelength", "r", "G", "VV", "VH", "HH", "HV"]


def test_window_average_empty_window(spectrum_pair):
    df = anisotropy_spectrum(*spectrum_pair)
    mean, std, n = window_average(df, center=900, half_width=1)
    assert n == 0 and np.isnan(mean)


def test_analyse_anisotropy_summary(spectrum_pair):
    sample, blank = spectrum_pair
    sample.metadata["concentration"] = 0.4
    cfg = AnisotropyConfig(target_wavelength=520, half_width=5)

    spectra, summary = analyse_anisotropy([sample], blank, cfg)

    assert list(spectra) == [sample.name]
    assert list(summary.columns[:2]) == ["name", "concentration"]
    assert summary["r_mean"].iloc[0] == pytest.approx(0.15, abs=0.01)
