"""Lane detection: find where each lane sits horizontally in a gel image."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from gelanalyzer.border import GelArea
from gelanalyzer.comb import find_comb


def detect_lanes(pixels: np.ndarray, n_lanes: int, area: GelArea | None = None) -> list[float]:
    """Return the x-centre (px) of each of the `n_lanes` lanes, left to right.

    Uses the comb: every lane starts at a well, and wells are evenly spaced.
    Lanes are treated as vertical for now (no fan-out / smile correction yet).
    """
    return find_comb(pixels, n_lanes=n_lanes, area=area).centers


def lanes_from_profile(
    image: np.ndarray,
    smoothing: float = 5.0,
    min_prominence: float = 0.2,
) -> list[float]:
    """Fallback without comb or lane count: peaks of the column brightness profile.

    Works on clean gels; on real gels the border and big bands fool it (US-1).

    smoothing:      Gaussian sigma in pixels (bigger = less noise sensitive).
    min_prominence: how far a peak must rise above its surroundings, as a
                    fraction of the profile's range (0..1). Filters noise bumps.
    """
    profile = image.mean(axis=0)
    profile = gaussian_filter1d(profile, sigma=smoothing)

    span = profile.max() - profile.min()
    if span == 0:
        return []  # blank image: no lanes

    peaks, _ = find_peaks(profile, prominence=min_prominence * span)
    return [float(x) for x in peaks]
