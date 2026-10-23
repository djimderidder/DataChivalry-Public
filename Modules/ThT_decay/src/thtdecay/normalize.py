r"""Normalisation of ThT traces against two control wells.

For a sample well S with fibril control F (fibrils without ATP: no disaggregation)
and base control B (no fibrils: background ThT signal):

.. math::

    n(t) = \frac{(S - B)(t) \,/\, (S - B)_0}{(F - B)(t) \,/\, (F - B)_0}

The subscript 0 is the start value: the first point (old behaviour) or the mean of
the first ``n_points``. So n starts at 1, and fibril-control drift (bleaching) is
divided out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fluostar.config import clean_well
from fluostar.io import TIME_COLUMN, FluostarData


def _start_value(trace: np.ndarray, n_points: int) -> float:
    finite = trace[np.isfinite(trace)]
    if finite.size == 0:
        raise ValueError("trace has no finite values")
    return float(np.mean(finite[:n_points]))


def normalized_trace(sample: np.ndarray, fibril: np.ndarray, base: np.ndarray,
                     n_points: int = 1) -> np.ndarray:
    signal = sample - base
    fibril_signal = fibril - base
    return (signal / _start_value(signal, n_points)) / (fibril_signal / _start_value(fibril_signal, n_points))


def normalize_tht(data: FluostarData, roi: pd.DataFrame, n_points: int = 1) -> pd.DataFrame:
    """Normalised traces for every row of ``roi`` (columns well, nFibril, nBase).

    ``data`` must have a single gain (use ``prepare`` or ``data.select_gain``).
    Returns a DataFrame with ``Time [s]`` and one column per well.
    """
    out = {TIME_COLUMN: data.time_s}
    for _, row in roi.iterrows():
        well = clean_well(row["well"])
        out[well] = normalized_trace(data.well(well), data.well(row["nFibril"]),
                                     data.well(row["nBase"]), n_points)
    return pd.DataFrame(out)
