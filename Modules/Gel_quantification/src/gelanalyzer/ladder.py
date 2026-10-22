"""Find and trace the ladder lane (US-13).

Recipe (from the product owner):
  1. Crop from the middle of the left neighbour lane to the middle of the right
     neighbour (an edge lane: up to the gel border), from the well floor to the
     gel bottom.
  2. Background first: remove a slow top-to-bottom trend, then take the most
     common value (histogram mode) as background and measure the noise on the
     background side of it.
  3. Threshold at background + k * noise -> mask -> blobs. A blob is NOT a
     ladder band if it touches the side or top of the crop (merged with a
     neighbour's band, a line across the gel, well residue) or if it is much
     weaker than the strong ladder bands.
  4. The ladder = the blobs roughly below each other: a robust straight line
     through their centres. Their widths, fitted from top to bottom, give the
     lane's broadness -> a trapezoid.
  5. Lane end = lowest ladder band + the distance between the last two bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_opening, find_objects, gaussian_filter, gaussian_filter1d, label
from scipy.signal import find_peaks

from gelanalyzer.border import GelArea, find_gel_area
from gelanalyzer.comb import Comb, find_comb
from gelanalyzer.io import GelImage
from gelanalyzer.profiles import lane_profile, rolling_disc_background


@dataclass
class LadderBand:
    x: float          # intensity-weighted centre (image coordinates)
    y: float
    width: float      # width at half the band's height (independent of band intensity)
    height: float
    peak: float       # strongest signal above background


@dataclass
class LadderLane:
    lane: int                          # 1-based
    centre: tuple[float, float]        # x = a + b * y
    width_line: tuple[float, float]    # width = c + d * y
    y_start: float                     # well floor
    y_end: float                       # lowest band + last band spacing
    bands: list[LadderBand]            # blobs on the ladder line
    crop: tuple[int, int, int, int]    # x0, x1, y0, y1 of the searched region
    pitch: float
    rejected: list[LadderBand] = field(default_factory=list)   # blobs in the crop that are not ladder
    mask: np.ndarray | None = None     # thresholded crop, for diagnostics

    def x_at(self, y):
        a, b = self.centre
        return a + b * np.asarray(y, dtype=float)

    def width_at(self, y):
        """Lane broadness at height y (kept between 0.2 and 1.2 lane pitch)."""
        c, d = self.width_line
        return np.clip(c + d * np.asarray(y, dtype=float), 0.2 * self.pitch, 1.2 * self.pitch)

    def trapezoid(self) -> list[tuple[float, float]]:
        """Corners top-left, top-right, bottom-right, bottom-left."""
        corners = []
        for y, sides in ((self.y_start, (-1, 1)), (self.y_end, (1, -1))):
            for side in sides:
                corners.append((float(self.x_at(y) + side * self.width_at(y) / 2), float(y)))
        return corners


def find_ladder(
    gel: GelImage,
    lane: int | None = None,
    area: GelArea | None = None,
    comb: Comb | None = None,
    k_sigma: float = 5.0,
    min_relative_strength: float = 0.15,
) -> LadderLane:
    """Trace the ladder lane of `gel` (lane number from gel.config unless given).

    k_sigma:               threshold = background + k_sigma * noise.
    min_relative_strength: a ladder band must be at least this fraction of a strong band.
    """
    if gel.config is None:
        raise ValueError("find_ladder needs gel.config (number of lanes, ladder lane)")
    if lane is None:
        if not gel.config.ladder_lanes:
            raise ValueError("gel.config has no ladder_lanes; give lane=...")
        lane = gel.config.ladder_lanes[0]
    n = gel.config.n_lanes
    pixels = gel.pixels
    area = area or find_gel_area(pixels)
    comb = comb or find_comb(pixels, n_lanes=n, area=area)
    centres, pitch, k = comb.centers, comb.pitch, lane - 1

    x0 = int(round(centres[k - 1])) if k > 0 else int(area.left)
    x1 = int(round(centres[k + 1])) if k < n - 1 else int(area.right)
    y0, y1 = int(comb.bottom), int(area.bottom)
    signal = _above_background(pixels[y0:y1, x0:x1])
    mask, blobs = _blobs(signal, k_sigma, min_area=0.02 * pitch ** 2, offset=(x0, y0))
    if blobs:
        strong = np.percentile([b.peak for b in blobs], 75)
        blobs = [b for b in blobs if b.peak >= min_relative_strength * strong]

    on_line = _robust_line(blobs, pitch)
    bands = sorted((b for b, ok in zip(blobs, on_line) if ok), key=lambda b: b.y)
    rejected = [b for b, ok in zip(blobs, on_line) if not ok]
    if len(bands) < 2:
        raise ValueError(f"no ladder found in lane {lane}: {len(bands)} band(s) on a line")

    ys = np.array([b.y for b in bands])
    b_, a_ = np.polyfit(ys, [b.x for b in bands], 1)
    widths = np.array([b.width for b in bands])
    if len(bands) >= 3 and np.ptp(ys) > 0:
        d_, c_ = np.polyfit(ys, widths, 1)
    else:
        d_, c_ = 0.0, float(np.median(widths))
    y_end = min(ys[-1] + (ys[-1] - ys[-2]), area.bottom)

    return LadderLane(lane=lane, centre=(float(a_), float(b_)), width_line=(float(c_), float(d_)),
                      y_start=float(comb.bottom), y_end=float(y_end), bands=bands, crop=(x0, x1, y0, y1),
                      pitch=pitch, rejected=rejected, mask=mask)


def _above_background(crop: np.ndarray) -> np.ndarray:
    """Signal above the background, in units of background noise."""
    image = gaussian_filter(crop.astype(float), 1.5)
    trend = gaussian_filter1d(np.percentile(image, 25, axis=1), 15)    # slow top-to-bottom change
    residual = image - trend[:, None]
    lo, hi = np.percentile(residual, [0.5, 99.5])
    counts, edges = np.histogram(residual, bins=400, range=(lo, hi))
    background = (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2
    below = residual[residual <= background] - background            # background side only
    noise = 1.4826 * np.median(np.abs(below)) + 1e-12
    return (residual - background) / noise


def _blobs(signal, k_sigma, min_area, offset):
    x0, y0 = offset
    mask = binary_opening(signal > k_sigma, iterations=1)
    labels, _ = label(mask)
    height, width = mask.shape
    blobs = []
    for i, region in enumerate(find_objects(labels), start=1):
        rows, cols = region
        if cols.start == 0 or cols.stop == width or rows.start == 0:
            continue                                   # touches the crop side or top: not a ladder band
        inside = labels[region] == i
        if inside.sum() < min_area:
            continue
        yy, xx = np.nonzero(inside)
        weight = signal[region][inside]
        cy = rows.start + float((yy * weight).sum() / weight.sum())
        blobs.append(LadderBand(
            x=x0 + cols.start + float((xx * weight).sum() / weight.sum()),
            y=y0 + cy,
            width=_half_height_width(signal, int(round(cy)), cols),
            height=float(rows.stop - rows.start),
            peak=float(weight.max())))
    return mask, blobs


def _half_height_width(signal: np.ndarray, row: int, cols: slice) -> float:
    """Width of the band at half its height, on a few rows through its centre."""
    profile = signal[max(row - 2, 0):row + 3].mean(axis=0)
    span = profile[cols.start:cols.stop]
    peak = int(np.argmax(span)) + cols.start
    half = profile[peak] / 2
    left = peak
    while left > 0 and profile[left] > half:
        left -= 1
    right = peak
    while right < len(profile) - 1 and profile[right] > half:
        right += 1
    # sub-pixel crossings
    l_edge = left + (half - profile[left]) / (profile[left + 1] - profile[left] + 1e-12)
    r_edge = right - 1 + (profile[right - 1] - half) / (profile[right - 1] - profile[right] + 1e-12)
    return float(r_edge - l_edge)


def _robust_line(blobs: list[LadderBand], pitch: float, max_slope: float = 0.15) -> np.ndarray:
    """Which blobs lie on one near-vertical line (try every pair, keep the line most agree with)."""
    if len(blobs) < 2:
        return np.ones(len(blobs), dtype=bool)
    x = np.array([b.x for b in blobs])
    y = np.array([b.y for b in blobs])
    tolerance = 0.15 * pitch
    best = np.zeros(len(blobs), dtype=bool)
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            if abs(y[j] - y[i]) < 5:
                continue
            slope = (x[j] - x[i]) / (y[j] - y[i])
            if abs(slope) > max_slope:
                continue
            inliers = np.abs(x - (x[i] + slope * (y - y[i]))) < tolerance
            if inliers.sum() > best.sum():
                best = inliers
    if best.sum() >= 2:                                   # refit on the inliers and re-select once
        slope, intercept = np.polyfit(y[best], x[best], 1)
        best = np.abs(x - (intercept + slope * y)) < tolerance
    return best


# --- US-14: which blob is which ladder band ------------------------------------------

@dataclass
class LadderBandCall:
    y: float
    kda: float | None        # None when the band set is incomplete (can't tell which is which)
    strength: float          # peak height above the rolling-disc background
    thick: bool = False      # one of the ladder's reference bands (e.g. 75/50/25)


@dataclass
class LadderIdentification:
    bands: list[LadderBandCall]
    complete: bool                     # exactly as many bands as the ladder has
    thick_bands_ok: bool               # the strongest bands are the ladder's thick ones
    spillage: list[float]              # heights of peaks rejected as spill from a neighbour
    ys: np.ndarray                     # profile, for diagnostics
    profile: np.ndarray
    background: np.ndarray


def identify_ladder_bands(gel: GelImage, ladder: LadderLane, width_fraction: float = 0.8) -> LadderIdentification:
    """Find the ladder's bands in its lane profile and give each its size (kDa).

    1. profile along the ladder's centre line, `width_fraction` of the lane width
    2. rolling-disc background (radius from the band height), then peaks: lower
       the bar step by step until we have as many as the ladder has bands;
       peaks must be apart by at least 30% of the typical band spacing
    3. too many: drop peaks that are spill from a neighbour (strongest at the
       lane edge, not in the middle); still too many: drop the weakest
    4. assign sizes top (largest) to bottom; check the thick reference bands
       are among the strongest
    """
    definition = gel.config.ladder if gel.config else None
    if definition is None:
        raise ValueError("gel.config.ladder is needed to identify ladder bands (e.g. 'Precision Plus Unstained')")
    n = len(definition)

    ys, values = lane_profile(gel.pixels, ladder.x_at, ladder.width_at, ladder.y_start, ladder.y_end,
                              fraction=width_fraction)
    values = gaussian_filter1d(values, 1.0)
    blob_heights = [b.height for b in ladder.bands] or [10.0]
    radius = int(max(2.5 * np.median(blob_heights), 15))
    background = rolling_disc_background(values, radius)
    signal = values - background

    blob_ys = sorted(b.y for b in ladder.bands)
    spacing = np.median(np.diff(blob_ys)) if len(blob_ys) > 1 else (ladder.y_end - ladder.y_start) / n
    distance = max(int(0.3 * spacing), 3)
    peaks = np.array([], dtype=int)
    for bar in np.geomspace(0.5, 0.01, 40):                        # lower the bar until we have n
        peaks, _ = find_peaks(signal, prominence=bar * signal.max(), distance=distance)
        if len(peaks) >= n:
            break

    spillage = []
    if len(peaks) > n:
        keep = [p for p in peaks if not _is_spill(gel.pixels, ladder, ys[p])]
        spillage = [float(ys[p]) for p in peaks if p not in keep]
        peaks = np.array(keep, dtype=int)
    if len(peaks) > n:                                             # still too many: keep the strongest
        peaks = np.sort(peaks[np.argsort(signal[peaks])[-n:]])

    complete = len(peaks) == n
    thick_index = {definition.kda.index(k) for k in definition.thick}
    calls = []
    for i, p in enumerate(peaks):
        kda = definition.kda[i] if complete else None
        calls.append(LadderBandCall(y=float(ys[p]), kda=kda, strength=float(signal[p]),
                                    thick=complete and i in thick_index))
    strongest = set(np.argsort([c.strength for c in calls])[-len(thick_index):]) if calls else set()
    thick_ok = complete and strongest == thick_index
    return LadderIdentification(bands=calls, complete=complete, thick_bands_ok=thick_ok, spillage=spillage,
                                ys=ys, profile=values, background=background)


def _is_spill(pixels: np.ndarray, ladder: LadderLane, y: float) -> bool:
    """A real ladder band is strongest in the middle of the lane; spill is strongest at an edge."""
    row = int(round(y))
    x, half = float(ladder.x_at(y)), float(ladder.width_at(y)) / 2
    xs = np.arange(int(x - ladder.pitch), int(x + ladder.pitch) + 1)
    xs = xs[(xs >= 0) & (xs < pixels.shape[1])]
    across = pixels[max(row - 2, 0):row + 3, xs].mean(axis=0)
    across = across - across.min()
    dx = xs - x
    middle = across[np.abs(dx) < half / 2].mean()
    left = across[(dx <= -half / 2) & (dx > -half)].mean()
    right = across[(dx >= half / 2) & (dx < half)].mean()
    return max(left, right) > middle
