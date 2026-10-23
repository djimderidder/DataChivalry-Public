import math
from pathlib import Path

import pandas as pd
import pytest

from fluostar.config import clean_well, is_missing, raw_file_path, read_config, select_rows


@pytest.mark.parametrize("value", [None, math.nan, "-", "", "  ", " - ", "nan", pd.NA])
def test_missing_values(value):
    assert is_missing(value)


@pytest.mark.parametrize("value", [0, 0.0, 150, "H8", "150"])
def test_present_values(value):
    assert not is_missing(value)


def test_clean_well():
    assert clean_well(" A9") == "A9"
    assert clean_well("H10 ") == "H10"


def test_raw_file_path_accepts_windows_separators(tmp_path):
    row = {"folder": r"202505 tht disag\20250509", "filename": "plate.csv"}
    assert raw_file_path(row, tmp_path) == tmp_path / "202505 tht disag" / "20250509" / "plate.csv"


def test_read_config_xlsx_and_csv(tmp_path):
    df = pd.DataFrame({"folder": ["f"], "filename": ["x.csv"], "Well": [" A9"]})
    df.to_excel(tmp_path / "c.xlsx", index=False)
    df.to_csv(tmp_path / "c.csv", index=False)
    for name in ("c.xlsx", "c.csv"):
        assert read_config(tmp_path / name)["Well"].iloc[0] == " A9"


def test_read_config_requires_columns(tmp_path):
    pd.DataFrame({"folder": ["f"]}).to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(ValueError, match="filename"):
        read_config(tmp_path / "c.csv", required=("folder", "filename"))


def test_select_rows():
    df = pd.DataFrame({"a": range(5)})
    assert select_rows(df) == [0, 1, 2, 3, 4]
    assert select_rows(df, [3, 1]) == [3, 1]
    with pytest.raises(IndexError):
        select_rows(df, [7])
