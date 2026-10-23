"""Time windows and the fits/averages done over them.

Both assays reduce a trace to numbers measured over a window:

* ThT initial rate:  linear fit over ``Window(end_s=3600)`` (absolute time, after the
  equilibration cut);  ThT endpoint: mean over ``last_seconds(t, 3600)``
* ATPase rate:       linear fit over ``Window.from_indices(t, starti, endi)``

All times are in seconds. A Window includes both of its bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Window:
    start_s: float = -math.inf
    end_s: float = math.inf

    def __post_init__(self):
        if self.start_s > self.end_s:
            raise ValueError(f"window start ({self.start_s}) is after its end ({self.end_s})")

    def mask(self, time_s: np.ndarray) -> np.ndarray:
        time_s = np.asarray(time_s, dtype=float)
        return (time_s >= self.start_s) & (time_s <= self.end_s)

    @classmethod
    def from_indices(cls, time_s: np.ndarray, start: int, stop: int) -> "Window":
        """Window covering ``time_s[start:stop]`` (Python slice: ``stop`` excluded)."""
        n = len(time_s)
        if not 0 <= start < stop <= n:
            raise ValueError(f"need 0 <= start < stop <= {n}, got start={start}, stop={stop}")
        return cls(float(time_s[start]), float(time_s[stop - 1]))

    def indices(self, time_s: np.ndarray) -> tuple[int, int]:
        """(start, stop) slice indices of the points inside the window."""
        inside = np.flatnonzero(self.mask(time_s))
        if inside.size == 0:
            raise ValueError("no time points inside the window")
        return int(inside[0]), int(inside[-1]) + 1


def last_seconds(time_s: np.ndarray, duration_s: float) -> Window:
    """The final ``duration_s`` seconds of the trace."""
    return Window(float(np.max(time_s)) - duration_s, math.inf)


@dataclass(frozen=True)
class LinearFit:
    slope: float          # signal per time unit
    intercept: float      # signal at t = 0
    r_value: float
    window: Window
    n_points: int
    time_unit_s: float    # 60 = slope per minute, 3600 = per hour

    @property
    def r_squared(self) -> float:
        return self.r_value ** 2

    def line(self, time_s: np.ndarray) -> np.ndarray:
        return self.slope * np.asarray(time_s, float) / self.time_unit_s + self.intercept


def _points(time_s, signal, window):
    time_s, signal = np.asarray(time_s, float), np.asarray(signal, float)
    use = window.mask(time_s) & np.isfinite(signal)  # NaN = 'overflow' or a gap
    return time_s[use], signal[use]


def fit_linear(time_s: np.ndarray, signal: np.ndarray, window: Window,
               time_unit_s: float = 60.0) -> LinearFit:
    """Least-squares line through the points inside ``window`` (NaN points skipped)."""
    t, y = _points(time_s, signal, window)
    if t.size < 2:
        raise ValueError(f"need at least 2 points inside the window, got {t.size}")
    res = stats.linregress(t / time_unit_s, y)
    return LinearFit(float(res.slope), float(res.intercept), float(res.rvalue), window,
                     int(t.size), float(time_unit_s))


def window_mean(time_s: np.ndarray, signal: np.ndarray, window: Window) -> float:
    _, y = _points(time_s, signal, window)
    return float(np.mean(y)) if y.size else float("nan")
