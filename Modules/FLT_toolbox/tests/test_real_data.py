"""Smoke tests on measured lifetime data.

The measured .dat files are NOT in git (unpublished data). Put lifetime files
(columns: time [ns], decay, IRF) in tests/data/real/ to run these, or point the
FLUOTIME_DATA environment variable at another folder. Without data, the tests
are skipped. Run only these with:  pytest -m real_data
"""

import os
from pathlib import Path

import numpy as np
import pytest

from fluotime.io import load_decay
from fluotime.lifetime.config import FitConfig
from fluotime.lifetime.fit import fit_global

REAL = Path(os.environ.get("FLUOTIME_DATA", Path(__file__).parent / "data" / "real"))
CONFIG = Path(__file__).parents[1] / "configs" / "biexp_af488.toml"
FILES = sorted(REAL.glob("*.dat")) if REAL.is_dir() else []

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not FILES, reason=f"no .dat files in {REAL}"),
]


@pytest.fixture(scope="module")
def decays():
    return [load_decay(f) for f in FILES]


@pytest.mark.parametrize("shift_mode", ["global", "per_dataset", "anchored"])
@pytest.mark.parametrize("irf_source", ["reference", "own"])
def test_real_data_fits(decays, irf_source, shift_mode):
    cfg = FitConfig.from_file(CONFIG).with_overrides(irf_source=irf_source, shift_mode=shift_mode)
    result = fit_global(decays, cfg)

    assert np.all(np.isfinite(result.taus))
    for spec, tau in zip(cfg.taus, result.taus):
        assert spec.min <= tau <= spec.max
    assert np.all(np.isfinite(result.summary()["chi2_red"]))
