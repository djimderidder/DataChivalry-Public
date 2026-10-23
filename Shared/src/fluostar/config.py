"""The per-experiment config spreadsheet (one row per well to analyse).

Common columns: ``folder`` and ``filename`` (relative to a raw-data root), ``Well``,
the control well(s), and free columns describing the sample (``X`` fibrils,
``M`` monomers, ``A``, ``B``, ``C``, ... other components, ``exp``, ``date``, ...).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

MISSING_TEXT = {"", "-", "nan", "none", "n/a"}


def is_missing(value) -> bool:
    """True for empty cells: None, NaN, '', '-', 'nan' (spaces ignored)."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in MISSING_TEXT


def clean_well(value) -> str:
    """' A9' -> 'A9' (FLUOstar exports and configs often pad well names)."""
    return str(value).strip()


def read_config(path: str | Path, sheet=0, required: Iterable[str] = ()) -> pd.DataFrame:
    """Read a config from .xlsx/.xls or .csv and check the required columns exist."""
    path = Path(path)
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path, sheet_name=sheet)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}; found {list(df.columns)}")
    return df


def select_rows(config: pd.DataFrame, rows: Sequence[int] | None = None) -> list[int]:
    """Row indices to process: all rows for ``None``, otherwise the given ones (checked)."""
    if rows is None:
        return list(range(len(config)))
    rows = list(rows)
    bad = [r for r in rows if not 0 <= r < len(config)]
    if bad:
        raise IndexError(f"row(s) {bad} not in config (has {len(config)} rows)")
    return rows


def raw_file_path(row, raw_data_root: str | Path) -> Path:
    """Path of the raw export of a config row. Windows ``\\`` in ``folder`` also work on macOS/Linux."""
    folder = str(row["folder"]).replace("\\", "/").strip()
    return Path(raw_data_root) / folder / str(row["filename"]).strip()
