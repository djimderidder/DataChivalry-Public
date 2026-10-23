"""Shared fixtures.

All test data is synthetic: decays are simulated from known parameters, so the
tests can check that the fitting code *recovers* those parameters. No measured
data is needed (or stored) in the repository.
"""

import numpy as np
import pytest

from fluotime.synthetic import make_synthetic_decay, make_synthetic_spectrum

TRUE_TAUS = (4.1, 1.5)


@pytest.fixture
def time_axis():
    """Uniform TCSPC-like time axis: 1024 bins of 0.05 ns."""
    return np.arange(1024) * 0.05


@pytest.fixture
def two_biexp_decays():
    """Two datasets sharing lifetimes, different amplitude ratios, same IRF."""
    common = dict(taus=TRUE_TAUS, shift=0.10, background=2.0, scatter=500.0)
    d0 = make_synthetic_decay("ds_a", amplitudes=(3.0e5, 0.6e5), seed=1, **common)
    d1 = make_synthetic_decay("ds_b", amplitudes=(1.5e5, 2.0e5), seed=2, **common)
    return [d0, d1]


@pytest.fixture
def two_biexp_decays_different_shifts():
    """Two datasets whose decays are offset differently from the measured IRF."""
    common = dict(taus=TRUE_TAUS, background=2.0, scatter=500.0)
    d0 = make_synthetic_decay("ds_a", amplitudes=(3.0e5, 0.6e5), shift=0.05, seed=3, **common)
    d1 = make_synthetic_decay("ds_b", amplitudes=(1.5e5, 2.0e5), shift=-0.10, seed=4, **common)
    return [d0, d1]


@pytest.fixture
def spectrum_pair():
    """A synthetic sample spectrum with known anisotropy/G and a matching blank."""
    sample = make_synthetic_spectrum(r=0.15, g=0.9, blank_level=50.0, seed=5)
    blank = make_synthetic_spectrum(r=0.0, g=0.9, peak=0.0, blank_level=50.0,
                                    channels=("VV", "VH"), noise=False)
    return sample, blank
