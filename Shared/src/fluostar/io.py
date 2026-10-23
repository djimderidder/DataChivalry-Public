"""Loading BMG FLUOstar (Omega/MARS) CSV exports.

Export layout (one row per well, one column per time point and gain)::

    User: ..., Path: ..., Test run no.: ...          meta data ("key: value" cells)
    Test name: ..., Date: ..., Time: ...
    ID1: ..., ID2: ..., ID3: ...
    Fluorescence (FI), multichromatic                measurement type
    Well, Content, Raw Data (448-10/482-10 1), ...   one column per time point; the
                                                     last number is the gain (if any)
    ,     Time [s], 0, 82, 164, ...
     A9,  2 uM X; ..., 54, 55, ...                   one row per well ("overflow" -> NaN)

``FluostarData.data`` is the transposed table: columns ``Gain``, ``Time [s]`` and
one column per well (names without surrounding spaces).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import clean_well

TIME_COLUMN = "Time [s]"
GAIN_COLUMN = "Gain"
_PARENTHESES = re.compile(r"\(([^()]*)\)\s*$")


def _gain_from_header(cell: str) -> int:
    """'Raw Data (448-10/482-10 3)' -> 3; 'Raw Data (340)' (no gain number) -> 1."""
    match = _PARENTHESES.search(cell.strip())
    parts = match.group(1).split() if match else []
    return int(parts[-1]) if len(parts) >= 2 and parts[-1].isdigit() else 1


def _to_float(value: str, where: str) -> float:
    value = value.strip()
    if value == "" or value.lower() == "overflow":
        return np.nan
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"cannot read value {value!r} ({where})") from None


@dataclass
class FluostarData:
    """One plate-reader export. Methods return new objects; the original never changes."""

    data: pd.DataFrame
    layout: dict[str, str]                       # well -> content description
    meta: dict[str, str] = field(default_factory=dict)
    source: Path | None = None

    # --- description ----------------------------------------------------------
    @property
    def gains(self) -> list[int]:
        return sorted(int(g) for g in self.data[GAIN_COLUMN].unique())

    @property
    def wells(self) -> list[str]:
        return [c for c in self.data.columns if c not in (GAIN_COLUMN, TIME_COLUMN)]

    @property
    def file_folder(self) -> Path | None:
        return self.source.parent if self.source else None

    @property
    def file_name(self) -> str | None:
        return self.source.name if self.source else None

    def content(self, well: str) -> str:
        return self.layout.get(clean_well(well), "")

    # --- values (need a single gain) ----------------------------------------------
    def _require_single_gain(self) -> None:
        if len(self.gains) > 1:
            raise ValueError(f"data has gains {self.gains}; call select_gain(...) first")

    @property
    def time_s(self) -> np.ndarray:
        self._require_single_gain()
        return self.data[TIME_COLUMN].to_numpy(dtype=float)

    def well(self, name: str) -> np.ndarray:
        self._require_single_gain()
        name = clean_well(name)
        if name not in self.data.columns:
            raise KeyError(f"well {name!r} not in {self.file_name or 'data'} (has {self.wells})")
        return self.data[name].to_numpy(dtype=float)

    # --- selections (return copies) ---------------------------------------------
    def _with_data(self, data: pd.DataFrame) -> "FluostarData":
        return FluostarData(data.reset_index(drop=True), dict(self.layout), dict(self.meta), self.source)

    def copy(self) -> "FluostarData":
        return self._with_data(self.data.copy())

    def select_gain(self, gain: int) -> "FluostarData":
        if gain not in self.gains:
            raise ValueError(f"gain {gain} not in data (has {self.gains})")
        return self._with_data(self.data[self.data[GAIN_COLUMN] == gain].copy())

    def trim(self, start_s: float | None = None, end_s: float | None = None) -> "FluostarData":
        """Keep ``start_s < t <= end_s`` (either bound optional).

        Same convention as the old scripts: ``t > equilibration_time`` and ``t <= max_time``.
        """
        t = self.data[TIME_COLUMN]
        keep = pd.Series(True, index=self.data.index)
        if start_s is not None:
            keep &= t > start_s
        if end_s is not None:
            keep &= t <= end_s
        return self._with_data(self.data[keep].copy())


def load_fluostar(path: str | Path, file_name: str | None = None) -> FluostarData:
    """Load an export: ``load_fluostar(path)`` or ``load_fluostar(folder, file_name)``."""
    path = Path(path) / file_name if file_name is not None else Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, newline="", encoding="latin-1") as fh:
        rows = list(csv.reader(fh))

    header = next((i for i, r in enumerate(rows) if r and r[0].strip() == "Well"), None)
    if header is None:
        raise ValueError(f"{path.name}: no row starting with 'Well' (is this a FLUOstar CSV export?)")

    meta = {}
    for row in rows[:header]:
        for cell in row:
            if ":" in cell:
                key, value = cell.split(":", 1)
                meta[key.strip()] = value.strip()
        if row and row[0].strip() and ":" not in row[0] and "measurement" not in meta:
            meta["measurement"] = row[0].strip()

    head = rows[header]
    time_row = rows[header + 1] if header + 1 < len(rows) else []
    if len(time_row) < 2 or not time_row[1].strip().startswith("Time"):
        raise ValueError(f"{path.name}: expected a 'Time [s]' row below the 'Well' row")

    columns = [j for j in range(2, len(head)) if head[j].strip()]
    gains = [_gain_from_header(head[j]) for j in columns]
    times = [_to_float(time_row[j] if j < len(time_row) else "", f"time, column {j}") for j in columns]

    table = {GAIN_COLUMN: gains, TIME_COLUMN: times}
    layout = {}
    for r, row in enumerate(rows[header + 2:], start=header + 2):
        if not row or not row[0].strip():
            continue
        well = clean_well(row[0])
        layout[well] = row[1].strip() if len(row) > 1 else ""
        table[well] = [_to_float(row[j] if j < len(row) else "", f"well {well}, line {r + 1}")
                       for j in columns]

    data = pd.DataFrame(table)
    data[GAIN_COLUMN] = data[GAIN_COLUMN].astype(int)
    return FluostarData(data=data, layout=layout, meta=meta, source=path)
