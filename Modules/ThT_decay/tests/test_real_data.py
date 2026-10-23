"""Checks against real exports and your own analysed config (both NOT in git).

Put in tests/data/real/ (git-ignored):
  * the raw FLUOstar CSV export(s)
  * config.xlsx: config rows for those files, with the previously found values in
    the columns 'end' (endpoint, fraction) and 'end_time' (hours)
The raw file is looked up as <real>/<folder>/<filename>, or directly as <real>/<filename>.
Run only these with:  pytest -m real_data
"""

from pathlib import Path

import numpy as np
import pytest

from fluostar.config import is_missing, raw_file_path, read_config
from fluostar.io import load_fluostar
from thtdecay import ThtSettings, prepare, process_well
from thtdecay.kinetics import roi_from_config_row

REAL = Path(__file__).parent / "data" / "real"
CONFIG = REAL / "config.xlsx"

pytestmark = [pytest.mark.real_data,
              pytest.mark.skipif(not CONFIG.exists(), reason=f"no {CONFIG}")]


def _raw(row):
    path = raw_file_path(row, REAL)
    return path if path.exists() else REAL / str(row["filename"]).strip()


@pytest.fixture(scope="module")
def config():
    return read_config(CONFIG)


def test_every_config_row_can_be_analysed(config):
    settings = ThtSettings()
    for _, row in config.iterrows():
        _, k = process_well(prepare(load_fluostar(_raw(row)), settings), roi_from_config_row(row), settings)
        assert np.isfinite(k.rate_percent_per_hour) and np.isfinite(k.endpoint_percent)


def test_endpoints_match_the_config(config):
    settings = ThtSettings(max_time_s=None)
    for _, row in config.iterrows():
        if is_missing(row.get("end")):
            continue
        _, k = process_well(prepare(load_fluostar(_raw(row)), settings), roi_from_config_row(row), settings)
        assert k.endpoint_percent / 100 == pytest.approx(float(row["end"]), abs=0.01)
        if not is_missing(row.get("end_time")):
            assert k.end_time_h == pytest.approx(float(row["end_time"]), abs=0.05)
