"""Derived quantities from fitted amplitudes and lifetimes."""

from __future__ import annotations

from typing import Sequence

import numpy as np

KINDS = ("amplitude", "intensity")


def _weights(amplitudes, taus, kind):
    a, tau = np.asarray(amplitudes, float), np.asarray(taus, float)
    if kind == "amplitude":
        return a
    if kind == "intensity":
        return a * tau
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def fractions(amplitudes: Sequence[float], taus: Sequence[float], kind: str = "amplitude") -> np.ndarray:
    """Fractional amplitudes ``a_i / sum(a)`` or intensities ``a_i tau_i / sum(a tau)``."""
    w = _weights(amplitudes, taus, kind)
    total = w.sum()
    return np.zeros_like(w) if total <= 0 else w / total


def mean_lifetime(amplitudes: Sequence[float], taus: Sequence[float], kind: str = "amplitude") -> float:
    """Amplitude-weighted ``sum(a tau)/sum(a)`` or intensity-weighted ``sum(a tau^2)/sum(a tau)``."""
    w = _weights(amplitudes, taus, kind)
    total = w.sum()
    return float("nan") if total <= 0 else float(np.sum(w * np.asarray(taus, float)) / total)
