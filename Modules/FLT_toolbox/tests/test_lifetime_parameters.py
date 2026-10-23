import pytest

from fluotime.lifetime.config import FitConfig
from fluotime.lifetime.parameters import build_parameters, dataset_labels


def _config(**kw):
    base = dict(n_exp=2, taus=[{"value": 4.1, "min": 3.7, "max": 4.2}, 2.8])
    base.update(kw)
    return FitConfig.from_dict(base)


def test_dataset_labels():
    assert dataset_labels(3) == ["ds0", "ds1", "ds2"]


def test_global_parameters(two_biexp_decays):
    p = build_parameters(two_biexp_decays, _config())

    assert {"tau1", "tau2", "shift"} <= set(p)
    for lbl in ("ds0", "ds1"):
        assert {f"A1_{lbl}", f"A2_{lbl}", f"bkg_{lbl}", f"scatter_{lbl}"} <= set(p)
    assert "shift_ds0" not in p
    assert (p["tau1"].min, p["tau1"].max) == (3.7, 4.2)


def test_amplitudes_are_non_negative_and_guessed_from_counts(two_biexp_decays):
    p = build_parameters(two_biexp_decays, _config(amplitude_guess_factors=[2.0, 1.0]))
    assert p["A1_ds0"].min == 0
    assert p["A1_ds0"].value > 0
    # factor scales the guess: total/(n_exp*tau_i) * factor_i
    total = two_biexp_decays[0].counts.sum()
    assert p["A1_ds0"].value == pytest.approx(2.0 * total / (2 * 4.1))
    assert p["A2_ds0"].value == pytest.approx(1.0 * total / (2 * 2.8))


def test_per_dataset_shift(two_biexp_decays):
    p = build_parameters(two_biexp_decays, _config(shift_mode="per_dataset",
                                                   shift={"value": 0, "min": -0.5, "max": 0.5}))
    assert "shift" not in p
    assert (p["shift_ds1"].min, p["shift_ds1"].max) == (-0.5, 0.5)


def test_anchored_shift_bounds_around_reference(two_biexp_decays):
    p = build_parameters(two_biexp_decays, _config(shift_mode="anchored", anchor_window=0.05),
                         shift_reference=0.2)
    assert p["shift_ds0"].value == pytest.approx(0.2)
    assert p["shift_ds0"].min == pytest.approx(0.15)
    assert p["shift_ds0"].max == pytest.approx(0.25)


def test_anchored_requires_reference(two_biexp_decays):
    with pytest.raises(ValueError, match="shift_reference"):
        build_parameters(two_biexp_decays, _config(shift_mode="anchored"))


def test_fixed_tau(two_biexp_decays):
    cfg = _config(taus=[{"value": 4.1, "vary": False}, 2.8])
    p = build_parameters(two_biexp_decays, cfg)
    assert p["tau1"].vary is False
