"""Synthetic gel images with a known ground truth (used by the tests and the mock example).

Real gel photos are messy and we never know the "right answer" for sure.
A fake gel where WE placed every band lets tests check the analysis exactly:
we know how many lanes there are, where they are, and where each band sits.

Optional realism (all off by default, so old tests keep their simple gel):
  margin      background outside the gel on each side (left, top, right, bottom)
  comb        wells at the top of every lane, like a real casting comb
  well_level  well brightness relative to the gel: < 0 like Coomassie (wells
              lighter / less signal), > 0 like stain-free (wells darker / more signal)
  shift, fan, bend
              lanes that don't run straight down: at the gel bottom every lane has
              moved `shift` px, the outermost lanes `fan` px further outward
              (inner lanes proportionally), and `bend` adds curvature (px)
  empty_lanes lanes (0-based) without any bands
  ladder_lane a lane (0-based) holding a 10-band ladder at fixed heights, with
              three thicker (more intense) reference bands, like Precision Plus
  extra_bands extra bands drawn anywhere: dicts with x, y, intensity, width --
              e.g. an overloaded band in a neighbour lane that spills over
"""

from dataclasses import dataclass, field

import numpy as np


# ladder band heights as a fraction of the band region (top = largest protein),
# spacing shrinking towards the bottom like a real gradient gel; 75/50/25-like bands thicker
LADDER_FRACTIONS = np.array([0.0, 0.14, 0.26, 0.37, 0.50, 0.60, 0.72, 0.80, 0.90, 1.0])
LADDER_THICK = (3, 4, 6)


@dataclass
class Band:
    lane: int          # which lane (0-based)
    y: float           # vertical position in pixels (migration distance)
    intensity: float   # peak brightness, 0..1


@dataclass
class SyntheticGel:
    image: np.ndarray                  # 2D float array, 0..1, bright bands on dark background
    lane_centers: list[float]          # true x-centre of each lane (= centre of its well)
    bands: list[Band] = field(default_factory=list)
    gel_area: tuple[float, float, float, float] | None = None   # true left, top, right, bottom
    lane_pitch: float = 0.0            # distance between neighbouring lanes / wells
    comb_bottom: float = 0.0           # y of the well floors (lanes start here)
    deformation: tuple[float, float, float] = (0.0, 0.0, 0.0)   # shift, fan, bend

    def lane_x_at(self, lane: int, y: float) -> float:
        """True centre of `lane` (0-based) at height y."""
        shift, fan, bend = self.deformation
        _, _, _, bottom = self.gel_area
        t = max(0.0, (y - self.comb_bottom) / (bottom - self.comb_bottom))
        n = len(self.lane_centers)
        u = (2 * lane / (n - 1) - 1) if n > 1 else 0.0          # -1 (leftmost) .. +1 (rightmost)
        return self.lane_centers[lane] + (shift + fan * u) * t + bend * u * t * t


def make_gel(
    n_lanes: int = 6,
    width: int = 600,
    height: int = 400,
    lane_width: float = 40.0,
    band_height: float = 6.0,
    bands_per_lane: int = 3,
    noise: float = 0.02,
    background: float = 0.05,
    seed: int = 0,
    margin: int | tuple[int, int, int, int] = 0,
    outside_level: float = 0.0,
    border_line: float = 0.25,
    comb: bool = False,
    well_depth: int = 50,
    well_level: float = -0.03,
    well_walls: float = 0.15,
    shift: float = 0.0,
    fan: float = 0.0,
    bend: float = 0.0,
    empty_lanes: tuple[int, ...] = (),
    ladder_lane: int | None = None,
    extra_bands: tuple[dict, ...] = (),
) -> SyntheticGel:
    """Draw a fake gel: evenly spaced lanes, Gaussian-shaped bands, plus noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    image = np.full((height, width), background)

    # --- gel area: background outside, a thin darker rim along the edge -----------
    left, top, right, bottom = (margin,) * 4 if isinstance(margin, int) else margin
    gx0, gy0, gx1, gy1 = left, top, width - right, height - bottom
    inside = (xx >= gx0) & (xx < gx1) & (yy >= gy0) & (yy < gy1)
    if any((left, top, right, bottom)):
        image[~inside] = outside_level
        rim = inside & ~((xx >= gx0 + 3) & (xx < gx1 - 3) & (yy >= gy0 + 3) & (yy < gy1 - 3))
        for side, size in zip(("l", "t", "r", "b"), (left, top, right, bottom)):
            if size == 0:   # cropped side: no rim there
                rim &= {"l": xx >= gx0 + 3, "t": yy >= gy0 + 3,
                        "r": xx < gx1 - 3, "b": yy < gy1 - 3}[side]
        image[rim] += border_line

    spacing = (gx1 - gx0) / (n_lanes + 1)
    lane_centers = [gx0 + spacing * (i + 1) for i in range(n_lanes)]

    # --- comb: one well per lane, open at the gel top ------------------------------
    depth = well_depth if comb else 0
    if comb:
        half = 0.35 * spacing                       # wells are wider than the teeth between them
        for cx in lane_centers:
            well = (np.abs(xx - cx) < half) & (yy >= gy0) & (yy < gy0 + depth)
            walls = well & ~((np.abs(xx - cx) < half - 2) & (yy < gy0 + depth - 2))
            image[well] += well_level
            image[walls] += well_walls

    truth = SyntheticGel(image=image, lane_centers=lane_centers, gel_area=(gx0, gy0, gx1, gy1),
                         lane_pitch=spacing, comb_bottom=gy0 + depth, deformation=(shift, fan, bend))

    # --- bands, below the wells, following the (possibly bent) lane paths --------
    gel_h = gy1 - gy0
    lowest, highest = gy0 + depth + 0.15 * gel_h, gy0 + 0.9 * gel_h
    bands = []
    for lane, cx in enumerate(lane_centers):
        ys = np.sort(rng.uniform(lowest, highest, bands_per_lane))
        if lane in empty_lanes:
            continue
        if lane == ladder_lane:
            ys = lowest + LADDER_FRACTIONS * (highest - lowest)
        for i, y in enumerate(ys):
            intensity = rng.uniform(0.4, 0.9)
            if lane == ladder_lane:
                intensity = 0.9 if i in LADDER_THICK else 0.5
            x = truth.lane_x_at(lane, y)
            # Band = flat-ish across the lane (super-Gaussian in x), Gaussian in y
            shape_x = np.exp(-(((xx - x) / (lane_width / 2)) ** 6))
            shape_y = np.exp(-0.5 * ((yy - y) / (band_height / 2)) ** 2)
            image += intensity * shape_x * shape_y
            bands.append(Band(lane=lane, y=float(y), intensity=float(intensity)))

    for extra in extra_bands:
        shape_x = np.exp(-(((xx - extra["x"]) / (extra["width"] / 2)) ** 6))
        shape_y = np.exp(-0.5 * ((yy - extra["y"]) / (band_height / 2)) ** 2)
        image += extra["intensity"] * shape_x * shape_y

    image += rng.normal(0, noise, image.shape)
    truth.image = np.clip(image, 0, 1)
    truth.bands = bands
    return truth
