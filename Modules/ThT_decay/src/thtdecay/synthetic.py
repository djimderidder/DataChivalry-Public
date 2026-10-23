r"""Mock ThT disaggregation plates with known ground truth.

Normalised decay of a sample well: a power law

.. math::  n(t) = p + (1 - p)\,(1 + t/t_0)^{-\alpha}, \qquad \alpha = 0.6

(``t0`` sets how fast a well disaggregates, ``p`` an optional plateau).

Raw signals (before normalisation; t in hours for the slopes):

* base control (no fibrils):     B(t) = A_base (1 - s t)                 much lower signal
* fibril control (no ATP):       F(t) = B(t) + A_fib (1 - s t)           no decay, tiny drift
* sample:                        S(t) = B(t) + A_i (1 - s t) n(t)

Every well gets its own amplitude ``A_i`` (ThT intensity varies a lot between wells),
drawn around ``amplitude`` with a log-normal spread; ``A_fib`` is close to the
experimental amplitude. Multiplicative noise is added per point, and every gain is a
scaled copy. Because the drift ``(1 - s t)`` is shared, the noise-free normalised
sample equals ``n(t)`` exactly, so tests can compare against the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fluostar.synthetic import write_fluostar_csv

GAIN_SCALE = {1: 1 / 750, 2: 1 / 30, 3: 1.0}   # roughly the ratio between gains in real exports


def powerlaw_decay(time_s, alpha: float = 0.6, t0_s: float = 3600.0, plateau: float = 0.0) -> np.ndarray:
    t = np.asarray(time_s, float)
    return plateau + (1 - plateau) * (1 + t / t0_s) ** (-alpha)


@dataclass
class MockSample:
    well: str
    t0_s: float            # smaller = faster disaggregation
    b_uM: float = 0.0      # titrated component (written to the config's B column)
    alpha: float = 0.6
    plateau: float = 0.0

    @property
    def content(self) -> str:
        return f"2 uM X; 2 uM A; {self.b_uM:g} uM B;"


DEFAULT_SAMPLES = (  # a titration of B: more B -> faster disaggregation
    MockSample("A9", t0_s=40000, b_uM=0.02),
    MockSample("B9", t0_s=12000, b_uM=0.1),
    MockSample("C9", t0_s=4000, b_uM=0.5),
    MockSample("D9", t0_s=1500, b_uM=2),
    MockSample("E9", t0_s=600, b_uM=8),
)


@dataclass
class MockPlate:
    time_s: np.ndarray
    wells: dict            # well -> (content, values per gain), ready for write_fluostar_csv
    truth: dict            # sample well -> noise-free normalised trace n(t)
    fibril_control: str
    base_control: str
    samples: list = field(default_factory=list)


def make_tht_plate(
    samples=DEFAULT_SAMPLES,
    *,
    duration_h: float = 18.0,
    interval_s: float = 120.0,
    amplitude: float = 50000.0,
    amplitude_spread: float = 0.3,     # log-normal sigma between sample wells
    fibril_spread: float = 0.05,       # fibril control: close to the experimental amplitude
    base_fraction: float = 0.04,       # no-fiber control: much lower signal
    drift_per_hour: float = 0.002,     # tiny negative slope of the controls (and samples)
    noise: float = 0.01,               # relative noise per point
    gains=(1, 2, 3),
    fibril_control: str = "H10",
    base_control: str = "H11",
    seed: int | None = 0,
) -> MockPlate:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_h * 3600 + 1, interval_s)
    drift = 1 - drift_per_hour * t / 3600

    base = amplitude * base_fraction * drift
    raw = {base_control: ("2 uM A; 1 uM B; (no fibrils)", base),
           fibril_control: ("2 uM X; (no ATP)", base + amplitude * rng.lognormal(0, fibril_spread) * drift)}
    truth = {}
    for s in samples:
        truth[s.well] = powerlaw_decay(t, s.alpha, s.t0_s, s.plateau)
        a = amplitude * rng.lognormal(0, amplitude_spread)
        raw[s.well] = (s.content, base + a * drift * truth[s.well])

    wells = {}
    for well, (content, signal) in raw.items():
        noisy = signal * (1 + rng.normal(0, noise, (len(gains), t.size)))
        wells[well] = (content, np.array([noisy[i] * GAIN_SCALE.get(g, 1.0) for i, g in enumerate(gains)]))
    return MockPlate(t, wells, truth, fibril_control, base_control, list(samples))


def write_mock_tht_experiment(folder: str | Path, seed: int | None = 0, **plate_kwargs):
    """Write a mock plate (FLUOstar CSV) plus a config spreadsheet that points to it.

    Returns ``(csv_path, config_path, MockPlate)``. Use ``folder`` as ``raw_data_root``.
    """
    folder = Path(folder)
    plate = make_tht_plate(seed=seed, **plate_kwargs)
    csv_path = write_fluostar_csv(folder / "mock" / "mock ThT plate.csv", plate.time_s, plate.wells,
                                  gains=(1, 2, 3), test_name="mock ThT disaggregation 3 gains")
    config = pd.DataFrame([{
        "folder": "mock", "filename": csv_path.name, "Well": s.well,
        "Control F": plate.fibril_control, "Control T": plate.base_control,
        "exp": "mock titration", "date": 20250101, "X": 2, "A": 2,
        "B": s.b_uM, "C": 0,
        "comments": f"simulated: power law alpha={s.alpha}, t0={s.t0_s:g} s",
    } for s in plate.samples])
    config_path = folder / "mock ThT config.xlsx"
    config.to_excel(config_path, index=False)
    return csv_path, config_path, plate
