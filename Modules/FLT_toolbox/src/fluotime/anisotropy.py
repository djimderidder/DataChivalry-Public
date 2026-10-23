r"""Steady-state anisotropy spectra (L-format, wavelength-resolved).

.. math::

    G(\lambda) = \frac{I_{HV}}{I_{HH}}, \qquad
    r(\lambda) = \frac{I_{VV} - G I_{VH}}{I_{VV} + 2 G I_{VH}}

Processing order (each step optional, defaults reproduce the original script):

1. Gaussian smoothing of the sample channels (``smooth_sigma``, in data points)
2. Blank subtraction of ``blank_channels`` (default VV and VH only); the blank is
   interpolated onto the sample wavelengths and, unless ``smooth_blank=True``,
   used unsmoothed
3. G factor and anisotropy per wavelength
4. Mean +/- std of r in a window around a target wavelength

Steps 1 and 2 are linear, so ``smooth_blank=True`` is identical to subtracting
first and smoothing afterwards. With the default the blank's own noise is
not smoothed; that adds a little noise to r but does not bias the average.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .io import ANISOTROPY_CHANNELS, Spectrum


def g_factor(hv: np.ndarray, hh: np.ndarray) -> np.ndarray:
    hv, hh = np.asarray(hv, float), np.asarray(hh, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(hh != 0, hv / np.where(hh != 0, hh, 1.0), np.nan)


def anisotropy(vv: np.ndarray, vh: np.ndarray, g) -> np.ndarray:
    vv, vh = np.asarray(vv, float), np.asarray(vh, float)
    num = vv - g * vh
    den = vv + 2 * g * vh
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, num / np.where(den != 0, den, 1.0), np.nan)


def smooth_spectrum(spectrum: Spectrum, sigma: float,
                    channels: Sequence[str] | None = None) -> Spectrum:
    """Gaussian-smooth the given channels (all by default). Returns a new Spectrum."""
    channels = list(spectrum.channels) if channels is None else channels
    return spectrum.replace(**{ch: gaussian_filter1d(spectrum[ch], sigma=sigma) for ch in channels})


def subtract_blank(spectrum: Spectrum, blank: Spectrum,
                   channels: Sequence[str] = ("VV", "VH")) -> Spectrum:
    """Subtract the blank (interpolated onto the sample wavelengths). Returns a new Spectrum."""
    out = {}
    for ch in channels:
        if ch not in blank.channels:
            raise KeyError(f"blank has no {ch} channel (has {sorted(blank.channels)})")
        out[ch] = spectrum[ch] - np.interp(spectrum.wavelength, blank.wavelength, blank[ch])
    return spectrum.replace(**out)


def anisotropy_spectrum(
    sample: Spectrum,
    blank: Spectrum | None = None,
    *,
    smooth_sigma: float | None = 2.0,
    smooth_blank: bool = False,
    blank_channels: Sequence[str] = ("VV", "VH"),
) -> pd.DataFrame:
    """r(lambda) and G(lambda) plus the processed channels, as a DataFrame."""
    processed = sample
    if smooth_sigma:
        processed = smooth_spectrum(processed, smooth_sigma)
    if blank is not None:
        if smooth_sigma and smooth_blank:
            blank = smooth_spectrum(blank, smooth_sigma)
        processed = subtract_blank(processed, blank, blank_channels)

    g = g_factor(processed["HV"], processed["HH"])
    r = anisotropy(processed["VV"], processed["VH"], g)

    df = pd.DataFrame({"wavelength": processed.wavelength, "r": r, "G": g})
    for ch in ANISOTROPY_CHANNELS:
        df[ch] = processed[ch]
    return df


def window_average(df: pd.DataFrame, center: float, half_width: float,
                   column: str = "r") -> tuple[float, float, int]:
    """Mean, std and number of points of ``column`` within center +/- half_width."""
    mask = (df["wavelength"] >= center - half_width) & (df["wavelength"] <= center + half_width)
    values = df.loc[mask, column].dropna()
    if values.empty:
        return float("nan"), float("nan"), 0
    return float(values.mean()), float(values.std()), int(values.size)


@dataclass
class AnisotropyConfig:
    target_wavelength: float
    half_width: float = 5.0
    smooth_sigma: float | None = 2.0
    smooth_blank: bool = False
    blank_channels: tuple[str, ...] = ("VV", "VH")


def analyse_anisotropy(samples: Sequence[Spectrum], blank: Spectrum | None,
                       config: AnisotropyConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Process several samples. Returns ({name: spectrum DataFrame}, summary table)."""
    spectra, rows = {}, []
    for s in samples:
        df = anisotropy_spectrum(s, blank, smooth_sigma=config.smooth_sigma,
                                 smooth_blank=config.smooth_blank,
                                 blank_channels=config.blank_channels)
        spectra[s.name] = df
        r_mean, r_std, n = window_average(df, config.target_wavelength, config.half_width)
        g_mean, _, _ = window_average(df, config.target_wavelength, config.half_width, "G")
        rows.append({"name": s.name, **s.metadata, "r_mean": r_mean, "r_std": r_std,
                     "G_mean": g_mean, "n_points": n})
    return spectra, pd.DataFrame(rows)
