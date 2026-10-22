"""Lane profiles: signal along a lane, and its background.

These are general building blocks: they work for any lane whose centre and
width are known as functions of height (ladder now, every lane later).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.ndimage import grey_opening, map_coordinates


def lane_profile(
    pixels: np.ndarray,
    x_at: Callable,
    width_at: Callable,
    y_start: float,
    y_end: float,
    fraction: float = 0.8,
    samples: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean signal across the lane, for every row from y_start to y_end.

    The lane may be slanted: on each row we sample `samples` points spread over
    `fraction` of the lane width around its centre (fraction < 1 keeps us away
    from the lane edges and the neighbours).
    Returns (ys, values).
    """
    ys = np.arange(int(round(y_start)), int(round(y_end)))
    spread = np.linspace(-0.5, 0.5, samples)[None, :] * fraction * np.asarray(width_at(ys))[:, None]
    xs = np.asarray(x_at(ys))[:, None] + spread
    rows = np.repeat(ys[:, None], samples, axis=1)
    values = map_coordinates(pixels.astype(float), [rows, xs], order=1, mode="nearest")
    return ys, values.mean(axis=1)


def rolling_disc_background(values: np.ndarray, radius: int) -> np.ndarray:
    """Background under a profile: roll a flat disc of `radius` along it from below.

    (Morphological opening: everything narrower than the disc -- the bands --
    is cut off; slow changes in the background are followed.) Pick the radius
    clearly larger than a band is tall, but smaller than slow background bumps.
    """
    return grey_opening(np.asarray(values, dtype=float), size=2 * int(radius) + 1)
