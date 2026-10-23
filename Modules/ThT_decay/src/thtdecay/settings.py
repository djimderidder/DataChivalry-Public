"""All analysis settings in one validated object."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

MODES = {  # the old ``Mode`` argument -> (smooth_initial, smooth_final)
    "off": (False, False),
    "initial": (True, False),
    "final": (False, True),
    "both": (True, True),
}


@dataclass(frozen=True)
class ThtSettings:
    """
    gain              which gain of a multi-gain export to use
    equilibration_s   drop t <= this (noisy start); the initial-rate window still uses
                      absolute time, so the default 1 h window covers 600-3600 s
    max_time_s        drop t > this (None = keep everything)
    initial_window_s  initial rate: linear fit over t <= initial_window_s (absolute time)
    final_window_s    endpoint: mean over the last final_window_s seconds
    smoothing_window, smoothing_poly
                      Savitzky-Golay smoothing (window in points, odd)
    smooth_initial, smooth_final
                      use the smoothed trace for the initial rate / endpoint
                      (old Mode: 'off', 'initial', 'final', 'both' -> see from_mode)
    normalize_points  normalise each trace by the mean of its first n points (old: 1)
    """

    gain: int = 3
    equilibration_s: float = 600.0
    max_time_s: float | None = None
    initial_window_s: float = 3600.0
    final_window_s: float = 3600.0
    smoothing_window: int = 51
    smoothing_poly: int = 3
    smooth_initial: bool = False
    smooth_final: bool = False
    normalize_points: int = 1

    def __post_init__(self):
        if self.smoothing_window % 2 != 1 or self.smoothing_window < 3:
            raise ValueError(f"smoothing_window must be odd and >= 3, got {self.smoothing_window}")
        if not 0 <= self.smoothing_poly < self.smoothing_window:
            raise ValueError("smoothing_poly must be >= 0 and smaller than smoothing_window")
        if self.initial_window_s <= 0 or self.final_window_s <= 0:
            raise ValueError("initial_window_s and final_window_s must be positive")
        if self.max_time_s is not None and self.max_time_s <= self.equilibration_s:
            raise ValueError("max_time_s must be larger than equilibration_s")
        if self.normalize_points < 1:
            raise ValueError("normalize_points must be at least 1")

    @classmethod
    def from_mode(cls, mode: str, **kwargs) -> "ThtSettings":
        """Old-style ``Mode='off'|'initial'|'final'|'both'``."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
        smooth_initial, smooth_final = MODES[mode]
        return cls(smooth_initial=smooth_initial, smooth_final=smooth_final, **kwargs)

    def with_changes(self, **changes) -> "ThtSettings":
        return dataclasses.replace(self, **changes)
