"""Parameter-recovery tests: simulate decays with known parameters, fit, compare."""

import numpy as np
import pytest

from fluotime.lifetime.config import FitConfig
from fluotime.lifetime.fit import fit_global
from fluotime.lifetime.results import fractions

from conftest import TRUE_TAUS

TRUE_AMPS = [(3.0e5, 0.6e5), (1.5e5, 2.0e5)]


def _config(**kw):
    base = dict(
        n_exp=2,
        taus=[{"value": 3.8, "min": 0.05, "max": 10}, {"value": 2.0, "min": 0.05, "max": 10}],
        shift={"value": 0.0, "min": -0.5, "max": 0.5},
        background={"value": 1.0, "min": 0, "max": 100},
        scatter={"value": 100.0, "min": 0, "max": 1e5},
    )
    base.update(kw)
    return FitConfig.from_dict(base)


def _assert_recovered(result, amps=TRUE_AMPS, rtol_tau=0.02, atol_frac=0.02):
    np.testing.assert_allclose(sorted(result.taus, reverse=True), TRUE_TAUS, rtol=rtol_tau)
    order = np.argsort(result.taus)[::-1]
    for i, true in enumerate(amps):
        fitted = result.amplitudes(i)[order]
        np.testing.assert_allclose(fractions(fitted, TRUE_TAUS), fractions(true, TRUE_TAUS),
                                   atol=atol_frac)


@pytest.mark.parametrize("irf_source", ["reference", "own"])
def test_global_shift_recovers_parameters(two_biexp_decays, irf_source):
    result = fit_global(two_biexp_decays, _config(irf_source=irf_source))

    assert result.success
    _assert_recovered(result)
    assert result.shift(0) == pytest.approx(0.10, abs=0.01)
    assert result.shift(0) == result.shift(1)


def test_per_dataset_shift_recovers_individual_shifts(two_biexp_decays_different_shifts):
    result = fit_global(two_biexp_decays_different_shifts, _config(shift_mode="per_dataset"))

    _assert_recovered(result)
    assert result.shift(0) == pytest.approx(0.05, abs=0.01)
    assert result.shift(1) == pytest.approx(-0.10, abs=0.01)


def test_anchored_shift_stays_within_window(two_biexp_decays_different_shifts):
    result = fit_global(two_biexp_decays_different_shifts,
                        _config(shift_mode="anchored", anchor_window=0.05))

    ref = result.shift_reference
    assert ref == pytest.approx(0.05, abs=0.01)  # dataset 0 is the reference
    for i in range(2):
        assert ref - 0.05 - 1e-9 <= result.shift(i) <= ref + 0.05 + 1e-9


def test_tau_constraints_are_respected(two_biexp_decays):
    cfg = _config(taus=[{"value": 4.0, "min": 3.9, "max": 4.0}, 1.5])
    result = fit_global(two_biexp_decays, cfg)
    assert 3.9 <= result.taus[0] <= 4.0


def test_fixed_tau_is_not_changed(two_biexp_decays):
    cfg = _config(taus=[{"value": 4.1, "vary": False}, 2.0])
    result = fit_global(two_biexp_decays, cfg)
    assert result.taus[0] == 4.1
    assert result.taus[1] == pytest.approx(1.5, rel=0.03)


def test_monoexponential_on_biexponential_data_has_worse_chi2(two_biexp_decays):
    good = fit_global(two_biexp_decays, _config())
    mono = fit_global(two_biexp_decays, _config(n_exp=1, taus=[3.0]))
    assert mono.reduced_chi2(0) > 2 * good.reduced_chi2(0)


def test_good_fit_has_reduced_chi2_near_one(two_biexp_decays):
    result = fit_global(two_biexp_decays, _config())
    for i in range(2):
        assert 0.7 < result.reduced_chi2(i) < 1.5


@pytest.mark.parametrize("weighting", ["neyman", "pearson", "none"])
def test_weighting_options_run(two_biexp_decays, weighting):
    result = fit_global(two_biexp_decays, _config(weighting=weighting))
    np.testing.assert_allclose(sorted(result.taus, reverse=True), TRUE_TAUS, rtol=0.05)


def test_curves_and_residuals_have_data_shape(two_biexp_decays):
    result = fit_global(two_biexp_decays, _config())
    n = two_biexp_decays[0].time.size
    assert result.curve(0).shape == (n,)
    assert result.weighted_residuals(1).shape == (n,)
    assert result.irf_on_grid(0).sum() == pytest.approx(1.0, rel=1e-3)


def test_summary_table(two_biexp_decays):
    two_biexp_decays[0].metadata["concentration"] = 0.0
    two_biexp_decays[1].metadata["concentration"] = 1.0
    df = fit_global(two_biexp_decays, _config()).summary()

    assert list(df["name"]) == ["ds_a", "ds_b"]
    assert list(df["concentration"]) == [0.0, 1.0]
    for col in ("tau1", "tau2", "a1_frac", "a2_frac", "f1_frac", "tau_amp", "tau_int",
                "shift", "background", "scatter", "chi2_red"):
        assert col in df.columns
    np.testing.assert_allclose(df["a1_frac"] + df["a2_frac"], 1.0)


def test_own_irf_requires_irf_in_every_dataset(two_biexp_decays):
    two_biexp_decays[1].irf = None
    with pytest.raises(ValueError, match="IRF"):
        fit_global(two_biexp_decays, _config(irf_source="own"))


def test_reference_index_out_of_range(two_biexp_decays):
    with pytest.raises(ValueError, match="reference_index"):
        fit_global(two_biexp_decays, _config(reference_index=5))


def test_no_datasets():
    with pytest.raises(ValueError, match="at least one"):
        fit_global([], _config())
