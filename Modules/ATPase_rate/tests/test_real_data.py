"""Checks against a real export and your own analysed config (both NOT in git).

Put in tests/data/real/ (git-ignored):
  * the raw FLUOstar CSV export(s)
  * config.xlsx: config rows for those files with the previously found 'Rate'
The raw file is looked up as <real>/<folder>/<filename>, or directly as <real>/<filename>.
Old spreadsheets keep the ATPase concentration in column 'H': set ENZYME_COLUMN below.
Run only these with:  pytest -m real_data
"""

from pathlib import Path

import pandas as pd
import pytest

from atpase import AtpaseSettings, analyse_config
from fluostar.config import is_missing, raw_file_path, read_config

REAL = Path(__file__).parent / "data" / "real"
CONFIG = REAL / "config.xlsx"
ENZYME_COLUMN = "H"

pytestmark = [pytest.mark.real_data,
              pytest.mark.skipif(not CONFIG.exists(), reason=f"no {CONFIG}")]


@pytest.fixture(scope="module")
def config():
    config = read_config(CONFIG)
    # let the test find files placed directly in tests/data/real/
    for k, row in config.iterrows():
        if not raw_file_path(row, REAL).exists():
            config.loc[k, "folder"] = "."
    return config


def test_rates_match_the_config(config):
    _, summary = analyse_config(config, REAL, AtpaseSettings(enzyme_column=ENZYME_COLUMN))
    for (_, row), (_, result) in zip(config.iterrows(), summary.iterrows()):
        if not is_missing(row.get("Rate")):
            assert result["rate"] == pytest.approx(float(row["Rate"]), abs=0.01)
