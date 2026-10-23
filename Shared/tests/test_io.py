import numpy as np
import pytest

from fluostar.io import FluostarData, load_fluostar


def test_load_multi_gain_export(three_gain_csv):
    data = load_fluostar(three_gain_csv)

    assert isinstance(data, FluostarData)
    assert data.gains == [1, 2, 3]
    assert data.wells == ["A9", "B9", "H10", "H11"]          # leading spaces stripped
    assert data.layout["H11"] == "content H11;"
    assert data.meta["Test name"] == "ThT mock"
    assert list(data.data.columns[:2]) == ["Gain", "Time [s]"]
    assert len(data.data) == 3 * 7


def test_folder_and_name_arguments(three_gain_csv):
    data = load_fluostar(three_gain_csv.parent, three_gain_csv.name)
    assert data.file_name == three_gain_csv.name
    assert data.file_folder == three_gain_csv.parent


def test_overflow_becomes_nan(three_gain_csv):
    data = load_fluostar(three_gain_csv).select_gain(3)
    assert np.isnan(data.well("B9")[3])
    assert np.isfinite(data.well("B9")).sum() == 6


def test_absorbance_export_has_a_single_gain(absorbance_csv):
    data = load_fluostar(absorbance_csv)
    assert data.gains == [1]
    np.testing.assert_allclose(data.time_s, np.arange(10) * 16.0)
    assert data.well(" E8")[0] == pytest.approx(1.6)  # well names are cleaned on lookup too


def test_select_gain_returns_a_new_object(three_gain_csv):
    data = load_fluostar(three_gain_csv)
    g3 = data.select_gain(3)

    assert g3.gains == [3]
    assert data.gains == [1, 2, 3]  # original unchanged
    assert g3.well("A9")[0] == pytest.approx(1000.0)


def test_select_unknown_gain(three_gain_csv):
    with pytest.raises(ValueError, match="gain 7"):
        load_fluostar(three_gain_csv).select_gain(7)


def test_time_and_well_need_a_single_gain(three_gain_csv):
    data = load_fluostar(three_gain_csv)
    with pytest.raises(ValueError, match="select_gain"):
        data.time_s
    with pytest.raises(ValueError, match="select_gain"):
        data.well("A9")


def test_trim_keeps_start_exclusive_end_inclusive(three_gain_csv):
    """Same as the old scripts: data[t > equilibration] and data[t <= max_time]."""
    data = load_fluostar(three_gain_csv).select_gain(3)
    trimmed = data.trim(start_s=82, end_s=600)

    np.testing.assert_allclose(trimmed.time_s, [164, 246, 400, 600])
    np.testing.assert_allclose(data.time_s, [0, 82, 164, 246, 400, 600, 900])  # unchanged


def test_trim_is_safe_to_repeat(three_gain_csv):
    data = load_fluostar(three_gain_csv).select_gain(3)
    once = data.trim(start_s=100)
    twice = data.trim(start_s=100).trim(start_s=100)
    np.testing.assert_allclose(once.time_s, twice.time_s)


def test_unknown_well(three_gain_csv):
    with pytest.raises(KeyError, match="Z1"):
        load_fluostar(three_gain_csv).select_gain(1).well("Z1")


def test_file_without_well_header(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("User: x,,\nsomething,else,\n1,2,3\n")
    with pytest.raises(ValueError, match="Well"):
        load_fluostar(f)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fluostar(tmp_path / "nope.csv")
