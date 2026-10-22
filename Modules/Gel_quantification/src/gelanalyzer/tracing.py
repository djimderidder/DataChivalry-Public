"""Follow the lanes down the gel (US-12).

Lanes rarely run straight down: they shift, fan out, sometimes curve, and the
spread is uneven. But neighbouring lanes move TOGETHER and never overlap, and
where a lane is hard to see, its neighbours tell where it is.

Model -- every lane k starts at its comb well c_k and moves sideways as it runs
down; with t = 0 at the well floors and t = 1 at the gel bottom:

    x_k(t) = c_k + s_k * t + q_k * t^2
             s_k: slope     (how far lane k has moved at the bottom, px)
             q_k: curvature (extra bend, px)

How the s_k, q_k are found:
  1. Evidence: walk down each lane (around its current estimate) and find BANDS
     = peaks in the vertical profile (short in y). Streaks and smears have no
     peak and are ignored. At each band, measure its horizontal centre from its
     half-height edges; blobs that are wider than a lane are rejected.
  2. Fit: weighted least squares of the band centres, with two penalties:
       * slopes and curvatures must change smoothly from lane to lane
         (second differences across lanes) -> neighbours move together, and a
         lane without bands gets its path from its neighbours;
       * curvature costs extra -> lanes only bend when the bands demand it.
     Bands that don't fit (residual too large) are dropped and we fit again.
  3. Repeat from 1 with the new estimate, so strongly slanted lanes are found.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, minimum_filter1d
from scipy.signal import find_peaks

from gelanalyzer.border import GelArea, find_gel_area
from gelanalyzer.comb import Comb, find_comb


@dataclass
class BandPoint:
    """One piece of evidence: a band centre measured in a lane."""

    lane: int        # 0-based
    y: float
    x: float         # measured horizontal centre of the band
    width: float     # measured band width (half-height edges)
    strength: float  # peak height relative to noise


@dataclass
class Lanes:
    """Lane centre paths for a whole gel. Lanes are 0-based, left to right."""

    comb_centers: np.ndarray       # x of each well (px)
    y_start: float                 # well floors: lanes start moving here (t = 0)
    y_end: float                   # gel bottom (t = 1)
    slopes: np.ndarray             # s_k
    curves: np.ndarray             # q_k
    points: list[BandPoint] = field(default_factory=list)   # evidence used for the fit

    @classmethod
    def straight(cls, centers, y_start: float, y_end: float) -> Lanes:
        """Lanes running straight down from the comb."""
        c = np.asarray(centers, dtype=float)
        return cls(comb_centers=c, y_start=y_start, y_end=y_end,
                   slopes=np.zeros_like(c), curves=np.zeros_like(c))

    def __len__(self) -> int:
        return len(self.comb_centers)

    def x_at(self, y):
        """Lane centres at height y. Scalar y -> shape (n,); array of heights -> (len(y), n)."""
        y_arr = np.asarray(y, dtype=float)
        t = np.clip((y_arr - self.y_start) / (self.y_end - self.y_start), 0, None)[..., None]
        return self.comb_centers + self.slopes * t + self.curves * t ** 2

    def boundaries_at(self, y: float) -> np.ndarray:
        """n+1 lane edges at height y: halfway between neighbouring lanes (so lanes never overlap)."""
        x = self.x_at(y)
        mid = (x[:-1] + x[1:]) / 2
        first = x[0] - (mid[0] - x[0]) if len(x) > 1 else x[0]
        last = x[-1] + (x[-1] - mid[-1]) if len(x) > 1 else x[-1]
        return np.concatenate([[first], mid, [last]])

    def width_at(self, y: float) -> np.ndarray:
        """Width of each lane at height y (distance between its edges)."""
        return np.diff(self.boundaries_at(y))


def trace_lanes(
    pixels: np.ndarray,
    n_lanes: int,
    area: GelArea | None = None,
    comb: Comb | None = None,
    smoothness: float = 3.0,
    curvature_penalty: float = 1.0,
    min_snr: float = 5.0,
    iterations: int = 4,
) -> Lanes:
    """Find the path of every lane from its well to the bottom of the gel.

    smoothness:        how strongly neighbouring lanes are tied together.
    curvature_penalty: how much evidence it takes before a lane may bend.
    min_snr:           how clear a band must be (peak / noise) to count as evidence.
    """
    area = area or find_gel_area(pixels)
    comb = comb or find_comb(pixels, n_lanes=n_lanes, area=area)
    pitch = comb.pitch
    lanes = Lanes.straight(comb.centers, y_start=comb.bottom, y_end=area.bottom)

    signal = _band_signal(pixels, pitch)
    height = area.bottom - comb.bottom
    ys = np.arange(int(comb.bottom + 0.02 * height), int(area.bottom - 0.03 * area.height))
    if ys.size == 0:
        return lanes

    for _ in range(iterations):
        points = _find_band_points(signal, lanes, ys, pitch, min_snr)
        if len(points) < 3:
            break
        lanes, kept = _fit(lanes, points, pitch, smoothness, curvature_penalty)
        lanes.points = kept
    return lanes


# --- evidence ----------------------------------------------------------------------

def _band_signal(pixels: np.ndarray, pitch: float) -> np.ndarray:
    """Smoothed image minus the background level in the gaps between lanes (per row)."""
    image = gaussian_filter(pixels.astype(float), 1.5)
    gaps = minimum_filter1d(image, max(int(1.2 * pitch), 3), axis=1)
    return image - gaussian_filter1d(gaps, pitch / 3, axis=1)


def _find_band_points(signal, lanes: Lanes, ys, pitch, min_snr) -> list[BandPoint]:
    paths = lanes.x_at(ys)                                   # (len(ys), n)
    offsets = np.linspace(-0.25 * pitch, 0.25 * pitch, 9)    # central half of the lane
    points = []
    for k in range(len(lanes)):
        cols = np.clip(np.round(paths[:, k][:, None] + offsets).astype(int), 0, signal.shape[1] - 1)
        down_the_lane = signal[ys[:, None], cols].mean(axis=1)
        noise = 1.4826 * np.median(np.abs(np.diff(down_the_lane))) / np.sqrt(2) + 1e-12
        peaks, props = find_peaks(down_the_lane, prominence=min_snr * noise, width=(1, 0.5 * pitch))
        for i, prominence in zip(peaks, props["prominences"]):
            y = ys[i]
            across = signal[max(y - 2, 0):y + 3].mean(axis=0)
            band = _measure_band(across, paths[i, k], pitch)
            if band is not None:
                x, width = band
                points.append(BandPoint(lane=k, y=float(y), x=x, width=width, strength=prominence / noise))
    return points


def _measure_band(profile: np.ndarray, guess: float, pitch: float) -> tuple[float, float] | None:
    """Centre and width of the band at `guess`, from its half-height edges.

    Both edges must lie inside the lane window, and the width must fit a lane:
    blobs and smears that run into the neighbours are rejected.
    (A wider window finds more bands on strongly slanted lanes, but also lets in
    wrong points -- tried and rejected by the real-gel ratchet test. See US-12b.)
    """
    lo, hi = int(guess - 0.55 * pitch), int(guess + 0.55 * pitch)
    if lo < 0 or hi >= len(profile):
        return None
    window = profile[lo:hi]
    i = int(np.argmax(window))
    top = window[i]
    if top <= 0:
        return None
    half = top / 2
    left = i
    while left > 0 and window[left] > half:
        left -= 1
    right = i
    while right < len(window) - 1 and window[right] > half:
        right += 1
    if left == 0 or right == len(window) - 1:
        return None
    # sub-pixel half-height crossings
    l_edge = left + (half - window[left]) / (window[left + 1] - window[left])
    r_edge = right - 1 + (window[right - 1] - half) / (window[right - 1] - window[right])
    width = r_edge - l_edge
    if not 0.3 * pitch < width < 1.0 * pitch:
        return None
    return lo + (l_edge + r_edge) / 2, width


# --- fit ---------------------------------------------------------------------------

def _fit(lanes: Lanes, points: list[BandPoint], pitch, smoothness, curvature_penalty):
    n = len(lanes)
    k = np.array([p.lane for p in points])
    t = (np.array([p.y for p in points]) - lanes.y_start) / (lanes.y_end - lanes.y_start)
    shift = np.array([p.x for p in points]) - lanes.comb_centers[k]
    strength = np.array([p.strength for p in points])
    weight = np.sqrt(np.minimum(strength, 3 * np.median(strength)))    # strong bands count more, blobs not too much

    keep = np.ones(len(points), dtype=bool)
    slopes, curves = lanes.slopes, lanes.curves
    for _ in range(2):
        slopes, curves = _solve(n, k[keep], t[keep], shift[keep], weight[keep], smoothness, curvature_penalty)
        residual = np.abs(shift - (slopes[k] * t + curves[k] * t ** 2))
        keep = residual < max(0.12 * pitch, 3 * np.median(residual))

    fitted = Lanes(comb_centers=lanes.comb_centers, y_start=lanes.y_start, y_end=lanes.y_end,
                   slopes=slopes, curves=curves)
    return fitted, [p for p, ok in zip(points, keep) if ok]


def _solve(n, k, t, shift, weight, smoothness, curvature_penalty):
    """Penalised weighted least squares for slopes s and curvatures q (2n unknowns)."""
    rows = np.arange(len(shift))
    A = np.zeros((len(shift), 2 * n))
    A[rows, k] = t
    A[rows, n + k] = t ** 2

    second_diff = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        second_diff[i, i:i + 3] = [1, -2, 1]
    W = weight ** 2
    scale = W.sum() / n                                       # penalties relative to the evidence
    smooth = smoothness * scale * second_diff.T @ second_diff
    penalty = np.zeros((2 * n, 2 * n))
    penalty[:n, :n] = smooth + 1e-6 * scale * np.eye(n)
    penalty[n:, n:] = smooth + curvature_penalty * scale * np.eye(n)

    solution = np.linalg.solve(A.T @ (A * W[:, None]) + penalty, A.T @ (shift * W))
    return solution[:n], solution[n:]
