"""Find the comb wells at the top of the gel.

A casting comb is machined, so its wells are exactly evenly spaced. We fit a
GRID with two unknowns: the centre of the first well (x0) and the pitch (P):

    well k  at  x0 + k * P              (k = 0 .. n-1)
    tooth   at  x0 + (k - 0.5) * P      (between wells, and at both ends)

Score of a grid = | mean signal in the wells - mean signal on the teeth |.
  * Absolute value: wells may be lighter (Coomassie) or darker (stain-free).
  * Wells are WIDER than teeth, so we average a wide window (50 % of the pitch)
    at each well and a narrow one (15 %) at each tooth. A grid shifted by half
    a pitch (wells on teeth) then scores worse, instead of equally well.

That coarse search finds the pitch robustly, but thin well walls can pull it a
few px sideways. So we refine (coarse -> fine): every well is mirror-symmetric
around its centre, so we find each well's symmetry centre separately and fit a
straight line  centre = x0 + k * pitch  through them, dropping wells that don't
fit (e.g. a damaged well) and fitting again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from gelanalyzer.border import GelArea, find_gel_area

WELL_WINDOW = 0.50    # fraction of the pitch averaged around a well centre
TOOTH_WINDOW = 0.15   # fraction of the pitch averaged around a tooth centre


@dataclass
class Comb:
    centers: list[float]   # x of each well centre (px), left to right
    pitch: float           # distance between neighbouring wells (px)
    score: float           # well/tooth contrast of the best grid (for diagnostics)
    top: float = 0.0       # y of the first row crossing the wells
    bottom: float = 0.0    # y just below the well floors: where the lanes start


def find_comb(
    pixels: np.ndarray,
    n_lanes: int,
    area: GelArea | None = None,
    comb_fraction: float = 0.15,
    pitch_step: float = 0.25,
) -> Comb:
    """Fit an evenly spaced grid of `n_lanes` wells to the top of the gel.

    area:          gel area to search in (found automatically if not given).
    comb_fraction: how far down from the gel top to look, as a fraction of gel height.
    pitch_step:    resolution of the pitch search (px).
    """
    if n_lanes < 2:
        raise ValueError(f"n_lanes must be at least 2 to fit a comb, got {n_lanes}")
    area = area or find_gel_area(pixels)

    inset = 5   # stay clear of the gel's rim
    x_start, x_stop = int(area.left) + inset, int(area.right) - inset
    y_start = int(area.top)
    y_stop = int(area.top + comb_fraction * area.height)
    band = pixels[y_start:y_stop, x_start:x_stop].astype(float)
    width = band.shape[1]

    # remove slow trends (uneven illumination) but keep the well/tooth pattern
    band = band - gaussian_filter1d(band, width / n_lanes, axis=1)
    rows = _comb_rows(band)
    profile = np.median(band[rows], axis=0)
    row_index = np.nonzero(_comb_rows(band, smoothing=1.0))[0]   # sharper, for the well floor
    comb_top, comb_bottom = y_start + row_index[0], y_start + row_index[-1] + 1

    # integral of the profile -> fast window means
    integral = np.concatenate([[0.0], np.cumsum(profile)])
    positions = np.arange(width + 1, dtype=float)

    def window_means(centres, half):
        lo = np.clip(centres - half, 0, width)
        hi = np.clip(centres + half, 0, width)
        total = np.interp(hi, positions, integral) - np.interp(lo, positions, integral)
        return total / np.maximum(hi - lo, 1e-9)

    best = (-np.inf, 0.0, 0.0)
    k_wells = np.arange(n_lanes)
    k_teeth = np.arange(n_lanes + 1) - 0.5
    # the comb fills most of the gel width; the two outer blocks are at most ~1.5 pitch each
    for pitch in np.arange(width / (n_lanes + 3), width / (n_lanes - 0.5), pitch_step):
        margin = WELL_WINDOW * pitch / 2
        x0 = np.arange(margin, width - (n_lanes - 1) * pitch - margin, 1.0)
        if x0.size == 0:
            continue
        wells = window_means(x0[:, None] + pitch * k_wells, WELL_WINDOW * pitch / 2).mean(axis=1)
        teeth = window_means(x0[:, None] + pitch * k_teeth, TOOTH_WINDOW * pitch / 2).mean(axis=1)
        scores = np.abs(wells - teeth)
        i = int(np.argmax(scores))
        if scores[i] > best[0]:
            best = (float(scores[i]), float(x0[i]), float(pitch))

    score, x0, pitch = best
    x0, pitch = _refine_by_symmetry(profile, x0, pitch, n_lanes)
    centers = [x_start + x0 + k * pitch for k in range(n_lanes)]
    return Comb(centers=centers, pitch=pitch, score=score, top=float(comb_top), bottom=float(comb_bottom))


def _comb_rows(band: np.ndarray, smoothing: float = 3.0) -> np.ndarray:
    """Rows that cross the wells: where the left-right pattern is strongest.

    Wells can be deep (most of the search band) or shallow (stain-free: a few
    % of the gel height), so a fixed number of rows doesn't work for both.
    """
    texture = gaussian_filter1d(np.std(band, axis=1), smoothing)
    return texture >= 0.5 * texture.max()


def _symmetry_centre(profile: np.ndarray, guess: float, pitch: float) -> float:
    """Position near `guess` (within a quarter pitch) around which the profile is most mirror-symmetric."""
    positions = np.arange(len(profile), dtype=float)
    u = np.arange(1.0, pitch / 2)                      # distances from the centre
    best_d, best_err = 0.0, np.inf
    for d in np.arange(-pitch / 4, pitch / 4, 0.25):
        c = guess + d
        left = np.interp(c - u, positions, profile)
        right = np.interp(c + u, positions, profile)
        err = np.mean((left - right) ** 2)
        if err < best_err:
            best_d, best_err = d, err
    return guess + best_d


def _refine_by_symmetry(profile: np.ndarray, x0: float, pitch: float, n: int) -> tuple[float, float]:
    """Refine the grid: symmetry centre per well, then a robust straight-line fit."""
    smooth = gaussian_filter1d(profile, 1.0)
    k = np.arange(n, dtype=float)
    centres = np.array([_symmetry_centre(smooth, x0 + i * pitch, pitch) for i in k])

    keep = np.ones(n, dtype=bool)
    for _ in range(2):                                  # fit, drop outliers, fit again
        pitch_fit, x0_fit = np.polyfit(k[keep], centres[keep], 1)
        residual = np.abs(centres - (x0_fit + pitch_fit * k))
        keep = residual < max(3.0, 0.1 * pitch)
        if keep.sum() < 2:
            return x0, pitch                            # refinement not trustworthy: keep coarse grid
    return float(x0_fit), float(pitch_fit)
