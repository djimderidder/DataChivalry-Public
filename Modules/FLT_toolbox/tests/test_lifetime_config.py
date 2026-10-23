import math

import pytest

from fluotime.lifetime.config import FitConfig, ParamSpec


# --- ParamSpec --------------------------------------------------------------

def test_paramspec_from_number():
    p = ParamSpec.from_any(4.1)
    assert (p.value, p.min, p.max, p.vary) == (4.1, -math.inf, math.inf, True)


def test_paramspec_from_dict():
    p = ParamSpec.from_any({"value": 4.1, "min": 3.7, "max": 4.2})
    assert (p.value, p.min, p.max) == (4.1, 3.7, 4.2)


def test_paramspec_value_outside_bounds():
    with pytest.raises(ValueError, match="outside"):
        ParamSpec(value=5.0, min=3.7, max=4.2)


def test_paramspec_min_above_max():
    with pytest.raises(ValueError, match="min"):
        ParamSpec(value=1.0, min=2.0, max=0.0)


def test_paramspec_fixed():
    p = ParamSpec.from_any({"value": 4.1, "vary": False})
    assert p.vary is False


# --- FitConfig --------------------------------------------------------------

def test_minimal_config_has_defaults():
    cfg = FitConfig(n_exp=1, taus=[4.0])
    assert cfg.irf_source == "reference"
    assert cfg.shift_mode == "global"
    assert cfg.irf_preprocessing == "mean_threshold"
    assert cfg.weighting == "neyman"
    assert isinstance(cfg.taus[0], ParamSpec)
    assert cfg.amplitude_guess_factors == [1.0]


def test_number_of_taus_must_match_n_exp():
    with pytest.raises(ValueError, match="n_exp"):
        FitConfig(n_exp=2, taus=[4.0])


def test_n_exp_range():
    with pytest.raises(ValueError, match="n_exp"):
        FitConfig(n_exp=0, taus=[])


@pytest.mark.parametrize("field,value", [
    ("irf_source", "nearest"),
    ("shift_mode", "free"),
    ("irf_preprocessing", "magic"),
    ("weighting", "poisson-ish"),
])
def test_unknown_options_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        FitConfig(n_exp=1, taus=[4.0], **{field: value})


def test_from_dict_with_nested_specs():
    cfg = FitConfig.from_dict({
        "n_exp": 2,
        "taus": [{"value": 4.1, "min": 3.7, "max": 4.2}, {"value": 2.8, "min": 0.05, "max": 3.8}],
        "shift": {"value": 0.0, "min": -1.0, "max": 1.0},
        "shift_mode": "anchored",
        "anchor_window": 0.05,
    })
    assert cfg.taus[1].max == 3.8
    assert cfg.shift.min == -1.0
    assert cfg.shift_mode == "anchored"


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="tau_guess"):
        FitConfig.from_dict({"n_exp": 1, "taus": [4.0], "tau_guess": 4})


def test_with_overrides_returns_new_config():
    cfg = FitConfig(n_exp=1, taus=[4.0])
    new = cfg.with_overrides(shift_mode="per_dataset")
    assert new.shift_mode == "per_dataset"
    assert cfg.shift_mode == "global"


def test_from_toml_file(tmp_path):
    f = tmp_path / "fit.toml"
    f.write_text(
        'n_exp = 2\n'
        'irf_source = "own"\n'
        'amplitude_guess_factors = [70, 1]\n'
        '[[taus]]\nvalue = 4.1\nmin = 3.7\nmax = 4.2\n'
        '[[taus]]\nvalue = 2.8\nmin = 0.05\nmax = 3.8\n'
        '[background]\nvalue = 2.5\nmin = 0\nmax = 100\n'
    )
    cfg = FitConfig.from_file(f)
    assert cfg.irf_source == "own"
    assert cfg.taus[0].min == 3.7
    assert cfg.background.max == 100
    assert cfg.amplitude_guess_factors == [70, 1]


def test_from_json_file(tmp_path):
    f = tmp_path / "fit.json"
    f.write_text('{"n_exp": 1, "taus": [{"value": 4.0, "min": 3, "max": 5}]}')
    assert FitConfig.from_file(f).taus[0].value == 4.0


def test_shipped_example_configs_are_valid():
    from pathlib import Path
    configs = sorted((Path(__file__).parents[1] / "configs").glob("*.toml"))
    assert configs, "expected example configs in configs/"
    for path in configs:
        FitConfig.from_file(path)
