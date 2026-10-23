"""Background subtraction against a control well."""

from __future__ import annotations

import numpy as np

from fluostar.config import is_missing
from fluostar.io import FluostarData


def background_subtracted(data: FluostarData, well: str, control=None) -> np.ndarray:
    """``well - control``; just ``well`` when no control is given (None, NaN, '-', '')."""
    if is_missing(control):
        return data.well(well)
    return data.well(well) - data.well(control)
