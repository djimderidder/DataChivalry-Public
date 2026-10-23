import numpy as np
import pandas as pd
import pytest

from atpase import AtpaseSettings, analyse_config, background_subtracted, update_config
from fluostar.io import load_fluostar


def test_background_subtraction(mock_experiment):
    _, csv_path, _, _ = mock_experiment
    data = load_fluostar(csv_path)
    np.testing.assert_allclose(background_subtracted(data, " E8", " H8"), data.well("E8") - data.well("H8"))
    np.testing.assert_allclose(background_subtracted(data, "E8", "-"), data.well("E8"))
    np.testing.assert_allclose(background_subtracted(data, "E8", float("nan")), data.well("E8"))


def test_noise_free_rates_are_recovered(noise_free_experiment):
    root, _, config, plate = noise_free_experiment
    _, summary = analyse_config(config, root, AtpaseSettings())
    for (_, row), truth in zip(summary.iterrows(), plate.samples):
        assert row["rate"] == pytest.approx(truth.rate, rel=0.01)


def test_noisy_rates_are_recovered(mock_experiment):
    root, _, config, plate = mock_experiment
    _, summary = analyse_config(config, root, AtpaseSettings())
    for (_, row), truth in zip(summary.iterrows(), plate.samples):
        assert row["rate"] == pytest.approx(truth.rate, rel=0.05)


def test_config_starti_skips_the_lag(noise_free_experiment):
    """D8 has a slow start; its config Starti skips it. Fitting from 0 underestimates the rate."""
    root, _, config, plate = noise_free_experiment
    lagged = [i for i, s in enumerate(plate.samples) if s.lag_s][0]
    _, with_start = analyse_config(config, root, AtpaseSettings(), rows=[lagged])
    no_start = config.copy()
    no_start.loc[lagged, "Starti"] = "-"
    _, from_zero = analyse_config(no_start, root, AtpaseSettings(), rows=[lagged])

    truth = plate.samples[lagged].rate
    assert with_start["window"].iloc[0] == "config/auto"
    assert abs(with_start["rate"].iloc[0] - truth) < abs(from_zero["rate"].iloc[0] - truth)


def test_summary_columns(mock_experiment):
    root, _, config, _ = mock_experiment
    traces, summary = analyse_config(config, root, AtpaseSettings(), rows=[0, 3])
    assert list(summary["row"]) == [0, 3]
    assert {"well", "X", "M", "A", "rate", "slope_per_min", "r_squared", "starti", "endi", "window"} <= set(summary)
    assert [t.row for t in traces] == [0, 3]


def test_missing_enzyme_column(mock_experiment):
    root, _, config, _ = mock_experiment
    with pytest.raises(ValueError, match="enzyme column 'H'"):
        analyse_config(config, root, AtpaseSettings(enzyme_column="H"))


def test_update_config_returns_a_new_table(mock_experiment):
    _, _, config, _ = mock_experiment
    picks = pd.DataFrame({"row": [1], "starti": [12], "endi": [300], "rate": [0.42]})
    updated = update_config(config, picks)
    assert (updated.loc[1, "Starti"], updated.loc[1, "Endi"], updated.loc[1, "Rate"]) == (12, 300, 0.42)
    assert config.loc[1, "Starti"] == "-"   # original untouched


def test_config_template_has_the_required_columns():
    from pathlib import Path
    from atpase.batch import REQUIRED_COLUMNS
    from fluostar.config import read_config
    template = Path(__file__).parents[1] / "input" / "ATPase config template.xlsx"
    config = read_config(template, required=REQUIRED_COLUMNS + (AtpaseSettings().enzyme_column,))
    assert config["Starti"].iloc[0] == "-"
