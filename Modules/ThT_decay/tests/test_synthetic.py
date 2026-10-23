import numpy as np
import pandas as pd
import pytest

from fluostar.io import load_fluostar
from thtdecay.synthetic import make_tht_plate, powerlaw_decay, write_mock_tht_experiment


def test_powerlaw_shape():
    t = np.array([0.0, 3600, 3 * 3600])
    np.testing.assert_allclose(powerlaw_decay(t, alpha=0.6, t0_s=3600), [1, 2 ** -0.6, 4 ** -0.6])
    assert powerlaw_decay(1e9, plateau=0.3) == pytest.approx(0.3, abs=1e-3)


def test_controls_barely_drift_and_base_is_much_lower():
    plate = make_tht_plate(noise=0.0, seed=0)
    fibril = plate.wells["H10"][1][-1]   # gain 3
    base = plate.wells["H11"][1][-1]
    assert fibril[-1] / fibril[0] == pytest.approx(1 - 0.002 * 18, rel=1e-6)  # tiny negative slope
    assert base.mean() < 0.1 * fibril.mean()


def test_well_amplitudes_vary():
    plate = make_tht_plate(noise=0.0, seed=3)
    starts = [plate.wells[w][1][-1][0] for w in plate.truth]
    assert np.std(starts) / np.mean(starts) > 0.05


def test_gains_are_scaled_copies():
    plate = make_tht_plate(noise=0.0, seed=0)
    g1, g2, g3 = plate.wells["A9"][1]
    np.testing.assert_allclose(g3 / g1, 750, rtol=1e-9)


def test_written_experiment_is_readable(tmp_path):
    csv_path, config_path, plate = write_mock_tht_experiment(tmp_path)
    data = load_fluostar(csv_path)
    config = pd.read_excel(config_path)
    assert data.gains == [1, 2, 3]
    assert set(plate.truth) <= set(data.wells)
    assert list(config["Well"]) == list(plate.truth)
    assert (tmp_path / config["folder"].iloc[0] / config["filename"].iloc[0]).exists()
