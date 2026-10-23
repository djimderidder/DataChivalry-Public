import numpy as np
import pytest

from atpase.synthetic import make_atpase_plate, nadh_absorbance


def test_nadh_falls_linearly_then_goes_flat():
    t = np.arange(0, 7200, 16.0)
    n = nadh_absorbance(t, rate_per_min=0.5, enzyme_uM=4.0)
    v = 0.5 * 6220 * 4e-6                       # absorbance per minute
    early = t < 600
    np.testing.assert_allclose(np.diff(n[early]) / (16 / 60), -v, rtol=1e-3)
    assert n[-1] == pytest.approx(0, abs=1e-3)


def test_lag_slows_the_start():
    t = np.arange(0, 1200, 16.0)
    fast = nadh_absorbance(t, 0.5, 4.0)
    lagged = nadh_absorbance(t, 0.5, 4.0, lag_s=900)
    assert (fast[0] - fast[20]) > 2 * (lagged[0] - lagged[20])


def test_plate_has_control_buffer_and_samples():
    plate = make_atpase_plate(seed=0)
    assert {"H8", "H7"} <= set(plate.wells)
    assert all(s.well in plate.wells for s in plate.samples)
