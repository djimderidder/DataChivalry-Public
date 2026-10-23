"""Fixtures: mock plates written as real FLUOstar exports (see thtdecay.synthetic)."""

import pytest

from fluostar.io import load_fluostar
from thtdecay.synthetic import write_mock_tht_experiment


@pytest.fixture
def mock_experiment(tmp_path):
    """(raw_data_root, csv_path, config_path, MockPlate) of a noisy mock titration."""
    csv_path, config_path, plate = write_mock_tht_experiment(tmp_path, seed=1)
    return tmp_path, csv_path, config_path, plate


@pytest.fixture
def noise_free_experiment(tmp_path):
    csv_path, config_path, plate = write_mock_tht_experiment(tmp_path, seed=2, noise=0.0)
    return tmp_path, csv_path, config_path, plate


@pytest.fixture
def mock_data(mock_experiment):
    return load_fluostar(mock_experiment[1])
