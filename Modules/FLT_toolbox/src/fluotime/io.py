"""Loading FluoTime text exports.

Expected formats (whitespace separated, no header):

* Lifetime decay:  ``time_ns  decay  irf``      (one row per TCSPC bin)
* Anisotropy:      ``wavelength  VV  VH  HH  HV``
* Blank:           ``wavelength  VV  VH``        (HH/HV optional)

Column positions are configurable, so other exports can be read as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ANISOTROPY_CHANNELS = ("VV", "VH", "HH", "HV")


def _read_columns(path: str | Path) -> np.ndarray:
    return np.loadtxt(path, ndmin=2)


def _check_columns(data: np.ndarray, needed: int, path) -> None:
    if data.shape[1] < needed:
        raise ValueError(
            f"{path}: expected at least {needed} columns, found {data.shape[1]}"
        )


@dataclass
class Decay:
    """One fluorescence decay histogram with (optionally) its measured IRF.

    ``metadata`` holds anything describing the sample (concentration, condition,
    ...); it is carried through to result tables and plot labels.
    """

    time: np.ndarray
    counts: np.ndarray
    irf: np.ndarray | None
    name: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.counts = np.asarray(self.counts, dtype=float)
        if self.irf is not None:
            self.irf = np.asarray(self.irf, dtype=float)

        arrays = [self.time, self.counts] + ([self.irf] if self.irf is not None else [])
        if any(a.ndim != 1 for a in arrays):
            raise ValueError("time, counts and irf must be 1-D arrays")
        if len({a.size for a in arrays}) != 1:
            raise ValueError("time, counts and irf must have the same length")
        if self.time.size < 2:
            raise ValueError("a decay needs at least two time points")

        steps = np.diff(self.time)
        if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-3, atol=1e-9):
            raise ValueError("time axis must be increasing and uniform (constant bin width)")

    @property
    def dt(self) -> float:
        return float(self.time[1] - self.time[0])


def load_decay(
    path: str | Path,
    name: str | None = None,
    metadata: dict | None = None,
    *,
    time_col: int = 0,
    counts_col: int = 1,
    irf_col: int | None = 2,
) -> Decay:
    """Read a decay file. Use ``irf_col=None`` for files without an IRF column."""
    path = Path(path)
    data = _read_columns(path)
    cols = [time_col, counts_col] + ([irf_col] if irf_col is not None else [])
    _check_columns(data, max(cols) + 1, path)

    return Decay(
        time=data[:, time_col],
        counts=data[:, counts_col],
        irf=data[:, irf_col] if irf_col is not None else None,
        name=name if name is not None else path.stem,
        metadata=dict(metadata or {}),
    )


@dataclass
class Spectrum:
    """Polarised emission spectra on a common wavelength axis."""

    wavelength: np.ndarray
    channels: dict[str, np.ndarray]
    name: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.channels = {k: np.asarray(v, dtype=float) for k, v in self.channels.items()}
        for k, v in self.channels.items():
            if v.shape != self.wavelength.shape:
                raise ValueError(f"channel {k} has a different length than the wavelength axis")

    def __getitem__(self, channel: str) -> np.ndarray:
        try:
            return self.channels[channel]
        except KeyError:
            raise KeyError(f"channel {channel!r} not in spectrum {self.name!r} "
                           f"(has {sorted(self.channels)})") from None

    def replace(self, **channels: np.ndarray) -> "Spectrum":
        """Return a copy with some channels replaced (the original is not changed)."""
        new = {k: v.copy() for k, v in self.channels.items()}
        new.update(channels)
        return Spectrum(self.wavelength.copy(), new, self.name, dict(self.metadata))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"wavelength": self.wavelength, **self.channels})


def load_spectrum(
    path: str | Path,
    channels: tuple[str, ...] = ANISOTROPY_CHANNELS,
    name: str | None = None,
    metadata: dict | None = None,
) -> Spectrum:
    """Read ``wavelength`` followed by one column per channel, in the order given."""
    path = Path(path)
    data = _read_columns(path)
    _check_columns(data, len(channels) + 1, path)
    return Spectrum(
        wavelength=data[:, 0],
        channels={ch: data[:, i + 1] for i, ch in enumerate(channels)},
        name=name if name is not None else path.stem,
        metadata=dict(metadata or {}),
    )
