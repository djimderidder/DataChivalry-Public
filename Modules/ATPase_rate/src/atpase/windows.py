"""The fitting window of an ATPase trace, as slice indices ``signal[start:stop]``.

The NADH absorbance falls linearly while ATP is regenerated and goes flat when the
NADH is used up. Fitting must stop before that point:

* automatic start: the first point (0)
* automatic stop:  the first point (from ``start`` on) where the signal reaches
                   ``auto_threshold`` (95 %) of the mean of the last ``auto_n_last`` (5) points
* from the config: the ``Starti`` / ``Endi`` columns, each one used when it is filled in

Each bound is chosen independently: a config with ``Starti = 150`` and ``Endi = '-'``
fits from point 150 to the automatic stop. (The old atpaseVisualize.py switched both
bounds to automatic when either one was missing.)
"""

from __future__ import annotations

import numpy as np

from fluostar.config import is_missing

from .settings import AtpaseSettings


def auto_stop(signal: np.ndarray, settings: AtpaseSettings = AtpaseSettings(), start: int = 0) -> int:
    """Index of the first point at/after ``start`` that reached the depletion threshold."""
    signal = np.asarray(signal, float)
    threshold = settings.auto_threshold * np.nanmean(signal[-settings.auto_n_last:])
    candidates = np.flatnonzero(signal[start:] <= threshold) + start
    candidates = candidates[candidates > start]
    return int(candidates[0]) if candidates.size else len(signal) - 1


def fit_window(signal: np.ndarray, starti=None, endi=None,
               settings: AtpaseSettings = AtpaseSettings()) -> tuple[int, int, str]:
    """``(start, stop, source)`` from the config values, automatic where missing.

    ``source`` says where each bound came from, e.g. ``"config/auto"``.
    """
    n = len(signal)
    start = 0 if is_missing(starti) else int(float(starti))
    stop = auto_stop(signal, settings, start) if is_missing(endi) else int(float(endi))
    if not 0 <= start < stop <= n:
        raise ValueError(f"invalid window start={start}, stop={stop} for a trace of {n} points")
    source = f"{'auto' if is_missing(starti) else 'config'}/{'auto' if is_missing(endi) else 'config'}"
    return start, stop, source
