"""Find the gel area: where the gel itself is, without the background around it.

What a gel edge looks like (from our real ChemiDoc images): a THIN LINE (the
gel's rim), with background on the outside that has LESS signal than the gel.
Things that are NOT the edge but look similar:
  * a slow brightness gradient outside the gel (uneven illumination / vignetting)
  * the line of well floors, a band running across all lanes, the loading front

Method, per side:
  1. Intensity profile across the image: per column the median over all rows
     (left/right edges run along the whole height); per row the 90th
     percentile over all columns (the top edge is interrupted by the wells,
     so a median would miss it).
  2. Keep only NARROW ridges: profile minus a smoothed copy of itself.
     That removes slow gradients.
  3. Walk from the image edge inward through the strong ridges in the outer
     15% of the image; the first one with background beyond it (outside lower
     than inside) is the edge. None found -> the gel is cropped on that side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class GelArea:
    """Rectangle in pixel coordinates; right/bottom are the first pixel OUTSIDE the gel."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


def find_gel_area(
    pixels: np.ndarray,
    smoothing: float = 1.5,
    ridge_width: float = 10.0,
    min_ridge_strength: float = 6.0,
    outer_zone: float = 0.15,
    min_contrast: float = 0.1,
) -> GelArea:
    """Return the gel area in `pixels` (2D, high value = more signal).

    smoothing:          Gaussian sigma (px) against pixel noise.
    ridge_width:        features broader than this (px) are treated as gradient, not edge.
    min_ridge_strength: a ridge must be this many times the typical ridge noise (MAD).
    outer_zone:         only look for an edge in this outer fraction of the image.
    min_contrast:       background must be lower than inside by this fraction
                        of (typical gel - darkest 1%).
    """
    image = gaussian_filter(pixels.astype(float), smoothing)
    p1, typical = np.percentile(image, [1, 50])
    rule = _EdgeRule(ridge_width, min_ridge_strength, outer_zone, needed=min_contrast * (typical - p1))
    height, width = image.shape

    columns = np.median(image, axis=0)
    rows = np.percentile(image, 90, axis=1)
    return GelArea(
        left=rule.find(columns, image, from_end=False, default=0),
        top=rule.find(rows, image.T, from_end=False, default=0),
        right=rule.find(columns, image, from_end=True, default=width),
        bottom=rule.find(rows, image.T, from_end=True, default=height),
    )


@dataclass
class _EdgeRule:
    ridge_width: float
    min_strength: float
    zone: float
    needed: float

    def find(self, profile: np.ndarray, image: np.ndarray, from_end: bool, default: float) -> float:
        """Edge position along the columns of `image` (profile has one value per column)."""
        n = len(profile)
        ridge = profile - gaussian_filter1d(profile, self.ridge_width)
        noise = np.median(np.abs(ridge - np.median(ridge))) + 1e-12
        peaks, _ = find_peaks(ridge, height=self.min_strength * noise)

        limit = int(self.zone * n)
        candidates = [p for p in peaks if p >= n - limit] if from_end else [p for p in peaks if p < limit]
        for p in (reversed(candidates) if from_end else candidates):
            if from_end:
                outside, inside = image[:, p + 3:], image[:, max(p - 3 - max(n - p, 15), 0):p - 3]
            else:
                outside, inside = image[:, :max(p - 3, 0)], image[:, p + 3:p + 3 + max(p, 15)]
            if self._is_background(outside, inside):
                return float(p)
        return float(default)

    def _is_background(self, outside: np.ndarray, inside: np.ndarray) -> bool:
        if outside.size == 0 or inside.size == 0:
            return False
        return np.median(outside) < np.median(inside) - self.needed
