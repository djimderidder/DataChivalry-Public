"""All analysis settings in one validated object."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class AtpaseSettings:
    """
    extinction_coeff   NADH at 340 nm, M^-1 cm^-1
    path_length_cm     1.0 for exports that say "Absorbance values are pathlength correct"
    enzyme_column      config column with the ATPase concentration (uM) used to normalise
                       the rate (old spreadsheets: "H")
    auto_threshold     automatic end: first point where the signal reaches this fraction
    auto_n_last        ... of the mean of the last n points (NADH depleted)
    label_columns      config columns shown under the bars of the rate plot
    """

    extinction_coeff: float = 6220.0
    path_length_cm: float = 1.0
    enzyme_column: str = "A"
    auto_threshold: float = 0.95
    auto_n_last: int = 5
    label_columns: tuple[str, ...] = ("X", "M", "A", "B", "C", "D", "E")

    def __post_init__(self):
        if self.extinction_coeff <= 0 or self.path_length_cm <= 0:
            raise ValueError("extinction_coeff and path_length_cm must be positive")
        if not 0 < self.auto_threshold <= 1:
            raise ValueError("auto_threshold must be in (0, 1]")
        if self.auto_n_last < 1:
            raise ValueError("auto_n_last must be at least 1")

    def with_changes(self, **changes) -> "AtpaseSettings":
        return dataclasses.replace(self, **changes)
