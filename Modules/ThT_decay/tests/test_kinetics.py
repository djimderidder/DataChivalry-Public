import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.signal import savgol_filter

from fluostar.io import load_fluostar
from thtdecay.kinetics import analyse_config, prepare, process_plate, process_well, tht_kinetics
from thtdecay.settings import ThtSettings
from thtdecay.synthetic import powerlaw_decay

ROI_A9 = {"well": "A9", "nFibril": "H10", "nBase": "H11"}


@pytest.fixture
def trace():
    t = np.arange(660.0, 18 * 3600, 120.0)       # after the 600 s equilibration cut
    return t, powerlaw_decay(t, t0_s=3000) / powerlaw_decay(t[0], t0_s=3000)


def test_initial_rate_matches_the_original_formula(trace):
    """Old code: linregress(t/60, y) over t <= 3600 s, rate = slope * 60 * 100 (%/h)."""
    t, y = trace
    k = tht_kinetics(t, y, ThtSettings())
    mask = t <= 3600
    old = stats.linregress(t[mask] / 60, y[mask]).slope * 60 * 100
    assert k.rate_percent_per_hour == pytest.approx(old)


def test_endpoint_and_drift_match_the_original_formula(trace):
    t, y = trace
    k = tht_kinetics(t, y, ThtSettings())
    final = y[t >= t.max() - 3600]
    assert k.endpoint_percent == pytest.approx(final.mean() * 100)
    assert k.drift == pytest.approx(final[0] - final[-1])
    assert k.end_time_h == pytest.approx(t.max() / 3600)


def test_smoothing_options(trace):
    t, y = trace
    rng = np.random.default_rng(0)
    noisy = y + rng.normal(0, 0.02, y.size)
    smoothed = savgol_filter(noisy, 51, 3)

    off = tht_kinetics(t, noisy, ThtSettings())
    both = tht_kinetics(t, noisy, ThtSettings.from_mode("both"))
    mask = t <= 3600
    assert both.rate_percent_per_hour == pytest.approx(
        stats.linregress(t[mask] / 3600, smoothed[mask]).slope * 100)
    assert both.endpoint_percent == pytest.approx(smoothed[t >= t.max() - 3600].mean() * 100)
    assert off.rate_percent_per_hour != pytest.approx(both.rate_percent_per_hour)


def test_window_uses_absolute_time(trace):
    """Equilibration is cut first; the 1 h window then covers 600-3600 s (50 min)."""
    t, y = trace
    k = tht_kinetics(t, y, ThtSettings())
    assert k.initial_fit.n_points == np.sum(t <= 3600)


def test_prepare_selects_gain_and_trims_without_changing_input(mock_data):
    settings = ThtSettings(max_time_s=14 * 3600)
    p = prepare(mock_data, settings)
    assert p.gains == [3]
    assert p.time_s.min() > 600 and p.time_s.max() <= 14 * 3600
    assert mock_data.gains == [1, 2, 3]                    # unchanged
    np.testing.assert_allclose(prepare(p, settings).time_s, p.time_s)  # re-running is harmless


def _true_kinetics(plate, prepared, well, settings):
    t = prepared.time_s
    truth = plate.truth[well]
    return tht_kinetics(t, truth[np.isin(plate.time_s, t)] / truth[plate.time_s > settings.equilibration_s][0],
                        settings)


def test_noise_free_plate_gives_the_true_kinetics(noise_free_experiment):
    _, csv_path, _, plate = noise_free_experiment
    settings = ThtSettings()
    prepared = prepare(load_fluostar(csv_path), settings)
    for well in plate.truth:
        _, k = process_well(prepared, dict(ROI_A9, well=well), settings)
        expected = _true_kinetics(plate, prepared, well, settings)
        assert k.rate_percent_per_hour == pytest.approx(expected.rate_percent_per_hour, abs=0.05)
        assert k.endpoint_percent == pytest.approx(expected.endpoint_percent, abs=0.05)


def test_noisy_plate_recovers_kinetics_within_noise(mock_experiment):
    """1 % noise per point, ~25 points in the 50-min window: rate uncertainty ~1.2 %/h."""
    _, csv_path, _, plate = mock_experiment
    settings = ThtSettings()
    prepared = prepare(load_fluostar(csv_path), settings)
    for well in plate.truth:
        _, k = process_well(prepared, dict(ROI_A9, well=well), settings)
        expected = _true_kinetics(plate, prepared, well, settings)
        assert k.rate_percent_per_hour == pytest.approx(expected.rate_percent_per_hour, abs=3.0)
        assert k.endpoint_percent == pytest.approx(expected.endpoint_percent, abs=2.0)


def test_faster_wells_have_steeper_rates(mock_experiment):
    """Mock titration: t0 decreases from A9 to E9, so rates get more negative."""
    _, csv_path, _, plate = mock_experiment
    settings = ThtSettings()
    prepared = prepare(load_fluostar(csv_path), settings)
    roi = pd.DataFrame({"well": list(plate.truth), "nFibril": "H10", "nBase": "H11"})
    _, kinetics, summary = process_plate(prepared, roi, settings)
    rates = summary["rate_percent_per_hour"].to_numpy()
    assert np.all(np.diff(rates) < 0)
    assert list(summary["well"]) == list(plate.truth)
    assert summary["content"].iloc[0].startswith("2 uM X")


def test_process_well_rejects_multi_row_roi(mock_data):
    prepared = prepare(mock_data, ThtSettings())
    roi = pd.DataFrame({"well": ["A9", "B9"], "nFibril": "H10", "nBase": "H11"})
    with pytest.raises(ValueError, match="single"):
        process_well(prepared, roi, ThtSettings())


def test_analyse_config_batch(mock_experiment):
    root, _, config_path, plate = mock_experiment
    config = pd.read_excel(config_path)
    results, summary = analyse_config(config, root, ThtSettings(), rows=[0, 2])

    assert list(summary["row"]) == [0, 2]
    assert list(summary["well"]) == ["A9", "C9"]
    assert {"exp", "date", "X", "A", "B", "rate_percent_per_hour", "endpoint_percent",
            "drift", "end_time_h"} <= set(summary.columns)
    assert set(results) == {0, 2}


def test_analyse_config_missing_control(mock_experiment):
    root, _, config_path, _ = mock_experiment
    config = pd.read_excel(config_path)
    config.loc[0, "Control T"] = "-"
    with pytest.raises(ValueError, match="Control T"):
        analyse_config(config, root, ThtSettings(), rows=[0])


def test_config_template_has_the_required_columns():
    from pathlib import Path
    from fluostar.config import read_config
    template = Path(__file__).parents[1] / "input" / "ThT config template.xlsx"
    read_config(template, required=("folder", "filename", "Well", "Control F", "Control T"))
