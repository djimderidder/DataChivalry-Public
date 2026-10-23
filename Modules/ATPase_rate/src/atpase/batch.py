"""All rows of a config spreadsheet, and writing picked windows back."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from fluostar.config import clean_well, raw_file_path, select_rows
from fluostar.io import FluostarData, load_fluostar

from .rate import AtpaseFit, atpase_rate
from .settings import AtpaseSettings
from .signal import background_subtracted
from .windows import fit_window

REQUIRED_COLUMNS = ("folder", "filename", "Well", "Control", "Starti", "Endi")


@dataclass
class AtpaseTrace:
    row: int
    well: str
    time_s: np.ndarray
    signal: np.ndarray
    result: AtpaseFit
    window_source: str


def load_trace(config_row, raw_data_root, cache: dict | None = None):
    """``(time_s, background-subtracted signal)`` of one config row."""
    path = raw_file_path(config_row, raw_data_root)
    cache = {} if cache is None else cache
    if path not in cache:
        cache[path] = load_fluostar(path)
    data: FluostarData = cache[path]
    return data.time_s, background_subtracted(data, config_row["Well"], config_row["Control"])


def analyse_config(config: pd.DataFrame, raw_data_root: str | Path, settings: AtpaseSettings,
                   rows: Sequence[int] | None = None):
    """Returns ``(list of AtpaseTrace, summary DataFrame)``; each raw file is loaded once."""
    if settings.enzyme_column not in config.columns:
        raise ValueError(f"config has no enzyme column {settings.enzyme_column!r} "
                         f"(set AtpaseSettings(enzyme_column=...))")
    cache, traces, summary = {}, [], []
    for k in select_rows(config, rows):
        row = config.iloc[k]
        time_s, signal = load_trace(row, raw_data_root, cache)
        start, stop, source = fit_window(signal, row["Starti"], row["Endi"], settings)
        result = atpase_rate(time_s, signal, start, stop, row[settings.enzyme_column], settings)
        traces.append(AtpaseTrace(k, clean_well(row["Well"]), time_s, signal, result, source))
        labels = {c: row[c] for c in settings.label_columns if c in config.columns}
        summary.append({"row": k, "well": clean_well(row["Well"]), **labels, **result.as_dict(),
                        "window": source})
    return traces, pd.DataFrame(summary)


def update_config(config: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``config`` with Starti/Endi/Rate from ``picks`` (columns row, starti, endi, rate)."""
    updated = config.copy()
    for column in ("Starti", "Endi", "Rate"):
        if column not in updated.columns:
            updated[column] = np.nan
        updated[column] = updated[column].astype(object)
    for _, pick in picks.iterrows():
        k = int(pick["row"])
        updated.loc[k, "Starti"] = int(pick["starti"])
        updated.loc[k, "Endi"] = int(pick["endi"])
        updated.loc[k, "Rate"] = float(pick["rate"])
    return updated
