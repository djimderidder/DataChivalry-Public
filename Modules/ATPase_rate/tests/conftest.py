"""Fixtures: mock NADH-coupled plates written as real FLUOstar exports."""

import pandas as pd
import pytest

from atpase.synthetic import write_mock_atpase_experiment


@pytest.fixture
def mock_experiment(tmp_path):
    """(raw_data_root, csv_path, config DataFrame, MockAtpasePlate)."""
    csv_path, config_path, plate = write_mock_atpase_experiment(tmp_path, seed=1)
    return tmp_path, csv_path, pd.read_excel(config_path), plate


@pytest.fixture
def noise_free_experiment(tmp_path):
    csv_path, config_path, plate = write_mock_atpase_experiment(tmp_path, seed=1, noise=0.0)
    return tmp_path, csv_path, pd.read_excel(config_path), plate
