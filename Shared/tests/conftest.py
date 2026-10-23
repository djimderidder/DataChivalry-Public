"""Shared fixtures: small mock FLUOstar exports written in the real CSV format."""

import numpy as np
import pytest

from fluostar.synthetic import write_fluostar_csv


@pytest.fixture
def three_gain_csv(tmp_path):
    """Fluorescence export with 3 gains, 4 wells and one overflow value."""
    t = np.array([0.0, 82, 164, 246, 400, 600, 900])
    base = {"A9": 1000.0, "B9": 2000.0, "H10": 1500.0, "H11": 50.0}
    wells = {}
    for well, level in base.items():
        per_gain = np.array([level * s * (1 - t / 1e4) for s in (0.01, 0.1, 1.0)])
        wells[well] = (f"content {well};", per_gain)
    wells["B9"][1][2, 3] = 5e6  # will be written as 'overflow'
    path = tmp_path / "plate.csv"
    write_fluostar_csv(path, t, wells, gains=(1, 2, 3), overflow=1e6, decimals=0,
                       test_name="ThT mock")
    return path


@pytest.fixture
def absorbance_csv(tmp_path):
    """Absorbance export (one channel, no gain number), like an ATPase run."""
    t = np.arange(10) * 16.0
    wells = {"E8": ("2 uM X; 4 uM A;", 1.6 - 0.001 * t), "H8": ("2 uM X;", 1.7 - 0.0001 * t)}
    path = tmp_path / "atpase.csv"
    write_fluostar_csv(path, t, wells, gains=None, measurement="Absorbance",
                       channel="340", decimals=3, test_name="ATPase mock")
    return path
