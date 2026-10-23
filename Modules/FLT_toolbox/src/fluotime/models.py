r"""Reconvolution model for TCSPC decays.

.. math::

    y(t) = \Delta t \,\big[\mathrm{IRF}(t - s) \otimes \sum_i A_i e^{-t/\tau_i}\big]
           + b + S\,\mathrm{IRF}(t - s)

* ``IRF`` normalised to unit sum, ``s`` the IRF shift (ns)
* ``A_i`` amplitudes in counts/ns, so ``sum(y) ~= sum(A_i * tau_i)`` (+ background)
* ``b`` constant background (counts/bin)
* ``S`` scatter: total counts of an IRF-shaped (prompt) contribution

Pure numpy/scipy: no fitting library is involved, which keeps it easy to test.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import fftconvolve

from .irf import shift_irf


def multi_exp_kernel(t: np.ndarray, amplitudes: Sequence[float], taus: Sequence[float]) -> np.ndarray:
    """``sum_i A_i * exp(-(t - t[0]) / tau_i)``."""
    if len(amplitudes) != len(taus):
        raise ValueError("amplitudes and taus must have the same length")
    t = np.asarray(t, dtype=float)
    rel = t - t[0]
    kernel = np.zeros_like(t)
    for a, tau in zip(amplitudes, taus):
        kernel += a * np.exp(-rel / tau)
    return kernel


def reconvolution_model(
    t: np.ndarray,
    irf: np.ndarray,
    amplitudes: Sequence[float],
    taus: Sequence[float],
    shift: float = 0.0,
    background: float = 0.0,
    scatter: float = 0.0,
    t_irf: np.ndarray | None = None,
) -> np.ndarray:
    """Model decay on time axis ``t`` (uniform). ``irf`` must be normalised (unit sum)."""
    t = np.asarray(t, dtype=float)
    dt = t[1] - t[0]
    irf_shifted = shift_irf(t, irf, shift, t_irf)
    kernel = multi_exp_kernel(t, amplitudes, taus)
    convolved = fftconvolve(irf_shifted, kernel, mode="full")[: t.size] * dt
    return convolved + background + scatter * irf_shifted
