"""Instrument response function (IRF) handling.

The IRF is always normalised to unit *sum* (not unit area), so that the
reconvolution amplitudes keep the units of the decay (see ``models.py``).

Preprocessing options (``preprocess_irf(method=...)``):

``"mean_threshold"``  (what the original scripts did)
    Fit a constant to the whole IRF trace (lmfit ``ConstantModel``; this is just
    the mean of all bins, peak included), set every bin below that value to zero,
    keep the rest unchanged. Note: nothing is subtracted, the mean acts as a
    noise threshold.
``"percentile_subtract"``  (used in the old time-resolved anisotropy script)
    Subtract a low percentile of the trace as baseline and clip at zero.
``"none"``
    Only clip negative values and normalise.
"""

from __future__ import annotations

import numpy as np

PREPROCESSING_METHODS = ("mean_threshold", "percentile_subtract", "none")


def irf_background(irf: np.ndarray, method: str = "mean", percentile: float = 5.0) -> float:
    """Estimate the IRF baseline: ``"mean"`` of the trace or a ``"percentile"``."""
    irf = np.asarray(irf, dtype=float)
    if method == "mean":
        return float(np.mean(irf))
    if method == "percentile":
        return float(np.percentile(irf, percentile))
    raise ValueError(f"unknown background method {method!r}; use 'mean' or 'percentile'")


def preprocess_irf(irf: np.ndarray, method: str = "mean_threshold",
                   percentile: float = 5.0) -> np.ndarray:
    """Clean a measured IRF and normalise it to unit sum. Returns a new array."""
    irf = np.asarray(irf, dtype=float)

    if method == "mean_threshold":
        threshold = irf_background(irf, "mean")
        clean = np.where(irf >= threshold, irf, 0.0)
    elif method == "percentile_subtract":
        baseline = irf_background(irf, "percentile", percentile)
        clean = np.clip(irf - baseline, 0.0, None)
    elif method == "none":
        clean = np.clip(irf, 0.0, None)
    else:
        raise ValueError(f"unknown IRF preprocessing method {method!r}; "
                         f"choose from {PREPROCESSING_METHODS}")

    total = clean.sum()
    if total <= 0:
        raise ValueError("IRF is empty after preprocessing (all bins zero)")
    return clean / total


def shift_irf(t_target: np.ndarray, irf: np.ndarray, shift: float,
              t_irf: np.ndarray | None = None) -> np.ndarray:
    """Return the IRF delayed by ``shift`` (ns), sampled on ``t_target``.

    Linear interpolation; bins shifted in from outside the IRF window are zero.
    ``t_irf`` is the time axis of ``irf`` (defaults to ``t_target``), so an IRF
    can be used on a dataset with a different time grid.
    """
    t_target = np.asarray(t_target, dtype=float)
    t_irf = t_target if t_irf is None else np.asarray(t_irf, dtype=float)
    return np.interp(t_target - shift, t_irf, np.asarray(irf, dtype=float), left=0.0, right=0.0)
