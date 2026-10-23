r"""From absorbance slope to ATPase rate.

In the coupled assay every ATP hydrolysed oxidises one NADH, so

.. math::

    \text{rate} = -\frac{\mathrm{d}A_{340}/\mathrm{d}t}{\varepsilon \, l \, [E]}

with the slope in absorbance per minute, :math:`\varepsilon` in M^-1 cm^-1, path
length *l* in cm and the enzyme concentration [E] in M: ATP per enzyme per minute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fluostar.windows import LinearFit, Window, fit_linear

from .settings import AtpaseSettings


@dataclass(frozen=True)
class AtpaseFit:
    rate: float          # ATP per enzyme per minute
    fit: LinearFit       # slope in absorbance per minute
    start: int           # fitted points: signal[start:stop]
    stop: int

    @property
    def slope(self) -> float:
        return self.fit.slope

    def as_dict(self) -> dict:
        return {"rate": self.rate, "slope_per_min": self.fit.slope, "r_squared": self.fit.r_squared,
                "starti": self.start, "endi": self.stop}


def atpase_rate(time_s: np.ndarray, signal: np.ndarray, start: int, stop: int, enzyme_uM: float,
                settings: AtpaseSettings = AtpaseSettings()) -> AtpaseFit:
    enzyme_uM = float(enzyme_uM)
    if not math.isfinite(enzyme_uM) or enzyme_uM <= 0:
        raise ValueError(f"enzyme concentration must be a positive number (uM), got {enzyme_uM} "
                         f"(config column {settings.enzyme_column!r})")
    fit = fit_linear(time_s, signal, Window.from_indices(time_s, start, stop), time_unit_s=60)
    rate = -fit.slope / (settings.extinction_coeff * settings.path_length_cm * enzyme_uM * 1e-6)
    return AtpaseFit(rate, fit, start, stop)
