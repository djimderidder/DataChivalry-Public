"""Simulated data with known ground truth.

Used by the tests (parameter recovery) and by the demo in runFluoTime.py (``analysis = "demo"``) so the
code can be tried without measured data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .io import ANISOTROPY_CHANNELS, Decay, Spectrum
from .models import reconvolution_model


def gaussian_irf(t: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Gaussian IRF normalised to unit sum."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    g = np.exp(-0.5 * ((np.asarray(t, float) - center) / sigma) ** 2)
    return g / g.sum()


def make_synthetic_decay(
    name: str,
    amplitudes: Sequence[float],
    taus: Sequence[float],
    *,
    shift: float = 0.0,
    background: float = 2.0,
    scatter: float = 0.0,
    n_points: int = 1024,
    dt: float = 0.05,
    irf_center: float = 5.0,
    irf_fwhm: float = 0.15,
    irf_peak_counts: float = 1e4,
    irf_background: float = 2.0,
    seed: int | None = None,
    metadata: dict | None = None,
) -> Decay:
    """Poisson-noise decay plus a noisy 'measured' IRF.

    The decay is generated with the true IRF delayed by ``shift``; the stored IRF
    is the undelayed one, so a fit should recover ``shift``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points) * dt
    irf_true = gaussian_irf(t, irf_center, irf_fwhm)

    expected = reconvolution_model(t, irf_true, amplitudes, taus, shift=shift,
                                   background=background, scatter=scatter)
    counts = rng.poisson(expected).astype(float)

    measured_irf = rng.poisson(irf_true / irf_true.max() * irf_peak_counts + irf_background)
    return Decay(time=t, counts=counts, irf=measured_irf.astype(float), name=name,
                 metadata=dict(metadata or {}))


def make_synthetic_spectrum(
    r: float,
    g: float,
    *,
    peak: float = 1e4,
    blank_level: float = 0.0,
    center: float = 520.0,
    width: float = 20.0,
    wavelengths: np.ndarray | None = None,
    channels: tuple[str, ...] = ANISOTROPY_CHANNELS,
    noise: bool = True,
    seed: int | None = None,
    name: str = "synthetic",
) -> Spectrum:
    """Polarised spectra with anisotropy ``r`` and G factor ``g`` (L-format).

    VV = I(1+2r)/3, VH = I(1-r)/(3G), HV = G*HH; a flat ``blank_level`` is added
    to VV and VH (as a buffer/scatter blank would be).
    """
    rng = np.random.default_rng(seed)
    wl = np.arange(490.0, 560.0, 0.5) if wavelengths is None else np.asarray(wavelengths, float)
    intensity = peak * np.exp(-0.5 * ((wl - center) / width) ** 2)

    all_channels = {
        "VV": intensity * (1 + 2 * r) / 3 + blank_level,
        "VH": intensity * (1 - r) / 3 / g + blank_level,
        "HH": intensity / 3,
        "HV": intensity / 3 * g,
    }
    out = {}
    for ch in channels:
        values = all_channels[ch]
        out[ch] = rng.normal(values, np.sqrt(np.clip(values, 1, None))) if noise else values
    return Spectrum(wavelength=wl, channels=out, name=name)
