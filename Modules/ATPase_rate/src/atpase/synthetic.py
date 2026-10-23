r"""Mock NADH-coupled ATPase plates with known rates.

For a sample well with enzyme concentration [E] and rate *k* (1/min), NADH
absorbance falls at ``v = k * eps * l * [E]`` per minute until the NADH is used up:

.. math::  N(t) = s \log\!\big(1 + e^{(N_0 - v\,\tau(t)) / s}\big)

(a straight line that bends smoothly into zero over a width *s*). An optional lag
``tau(t) = t - lag (1 - e^{-t/lag})`` makes the start slower, which is what a
config ``Starti`` skips. The control well (no enzyme) drifts down slightly, and the
sample sits on the same drift plus a small offset, so ``sample - control`` is
the NADH signal (minus ``N0``) plus noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fluostar.synthetic import write_fluostar_csv

from .settings import AtpaseSettings


def nadh_absorbance(time_s, rate_per_min, enzyme_uM, n0=1.3, smooth=0.03, lag_s=0.0,
                    settings: AtpaseSettings = AtpaseSettings()):
    t = np.asarray(time_s, float)
    v = rate_per_min * settings.extinction_coeff * settings.path_length_cm * enzyme_uM * 1e-6  # A/min
    tau = t - lag_s * (1 - np.exp(-t / lag_s)) if lag_s > 0 else t
    return smooth * np.logaddexp(0, (n0 - v * tau / 60) / smooth)


@dataclass
class MockWell:
    well: str
    rate: float           # ATP per enzyme per minute
    enzyme_uM: float = 4.0
    lag_s: float = 0.0
    starti: int | None = None   # written to the config (None -> '-')


DEFAULT_WELLS = (
    MockWell("B8", rate=0.20),
    MockWell("C8", rate=0.35),
    MockWell("D8", rate=0.45, lag_s=500, starti=150),   # slow start: config Starti skips it
    MockWell("E8", rate=0.60),
    MockWell("F8", rate=0.80, enzyme_uM=2.0),
)


@dataclass
class MockAtpasePlate:
    time_s: np.ndarray
    wells: dict
    control: str
    samples: list


def make_atpase_plate(samples=DEFAULT_WELLS, *, duration_h=2.1, interval_s=16.0, control="H8",
                      control_level=1.7, control_drift_per_h=0.03, offset=-0.1, noise=0.015,
                      seed: int | None = 0, settings: AtpaseSettings = AtpaseSettings()) -> MockAtpasePlate:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_h * 3600, interval_s)
    baseline = control_level - control_drift_per_h * t / 3600
    wells = {control: ("2 uM X; (no ATPase)", baseline + rng.normal(0, noise, t.size)),
             "H7": ("Buffer", baseline + 0.2 + rng.normal(0, noise, t.size))}
    for s in samples:
        n = nadh_absorbance(t, s.rate, s.enzyme_uM, lag_s=s.lag_s, settings=settings)
        wells[s.well] = (f"2 uM X; {s.enzyme_uM:g} uM A;", baseline + offset + n - n[0] + rng.normal(0, noise, t.size))
    return MockAtpasePlate(t, wells, control, list(samples))


def write_mock_atpase_experiment(folder: str | Path, seed: int | None = 0, **plate_kwargs):
    """Mock plate (FLUOstar CSV) plus a config pointing to it. Returns (csv, config, plate)."""
    folder = Path(folder)
    plate = make_atpase_plate(seed=seed, **plate_kwargs)
    csv_path = write_fluostar_csv(folder / "mock" / "mock ATPase plate.csv", plate.time_s, plate.wells,
                                  gains=None, measurement="Absorbance", channel="340", decimals=3,
                                  test_name="mock ATPase no injection")
    config = pd.DataFrame([{
        "folder": "mock", "filename": csv_path.name, "Well": s.well, "Control": plate.control,
        "Starti": "-" if s.starti is None else s.starti, "Endi": "-",
        "exp": "mock ATPase", "date": 20250101, "X": 2, "M": 0, "A": s.enzyme_uM, "B": 0, "C": 0,
        "Rate": np.nan, "note": f"simulated rate {s.rate:g} /min" + (f", lag {s.lag_s:g} s" if s.lag_s else ""),
    } for s in plate.samples])
    config_path = folder / "mock ATPase config.xlsx"
    config.to_excel(config_path, index=False)
    return csv_path, config_path, plate
