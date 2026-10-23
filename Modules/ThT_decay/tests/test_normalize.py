import numpy as np
import pandas as pd
import pytest

from fluostar.io import load_fluostar
from thtdecay.kinetics import prepare
from thtdecay.normalize import normalize_tht, normalized_trace
from thtdecay.settings import ThtSettings


def test_formula_on_simple_numbers():
    sample = np.array([110.0, 60.0])   # minus base -> 100, 50
    fibril = np.array([210.0, 190.0])  # minus base -> 200, 180
    base = np.array([10.0, 10.0])
    np.testing.assert_allclose(normalized_trace(sample, fibril, base), [1.0, 0.5 / 0.9])


def test_normalised_trace_starts_at_one(mock_data):
    data = mock_data.select_gain(3)
    roi = pd.DataFrame({"well": ["A9", "B9"], "nFibril": ["H10"] * 2, "nBase": ["H11"] * 2})
    norm = normalize_tht(data, roi)
    assert list(norm.columns) == ["Time [s]", "A9", "B9"]
    assert norm["A9"].iloc[0] == pytest.approx(1.0)


def test_noise_free_normalisation_recovers_the_power_law(noise_free_experiment):
    """Control drift and well-to-well amplitude differences are divided out exactly."""
    _, csv_path, _, plate = noise_free_experiment
    data = load_fluostar(csv_path).select_gain(3)
    roi = pd.DataFrame({"well": list(plate.truth), "nFibril": "H10", "nBase": "H11"})
    norm = normalize_tht(data, roi)
    for well, truth in plate.truth.items():
        np.testing.assert_allclose(norm[well], truth / truth[0], atol=2e-4)  # CSV rounds to counts


def test_mean_of_first_points_option(mock_data):
    data = mock_data.select_gain(3)
    roi = pd.DataFrame({"well": ["A9"], "nFibril": ["H10"], "nBase": ["H11"]})
    first = normalize_tht(data, roi, n_points=1)["A9"]
    mean5 = normalize_tht(data, roi, n_points=5)["A9"]
    assert first.iloc[0] == pytest.approx(1.0)
    assert mean5.iloc[:5].mean() == pytest.approx(1.0, abs=0.01)
    assert not np.allclose(first, mean5)


def test_roi_well_names_with_spaces(mock_data):
    data = prepare(mock_data, ThtSettings())
    roi = pd.DataFrame({"well": [" A9"], "nFibril": ["H10 "], "nBase": [" H11"]})
    assert "A9" in normalize_tht(data, roi).columns


def test_needs_a_single_gain(mock_data):
    roi = pd.DataFrame({"well": ["A9"], "nFibril": ["H10"], "nBase": ["H11"]})
    with pytest.raises(ValueError, match="select_gain"):
        normalize_tht(mock_data, roi)
