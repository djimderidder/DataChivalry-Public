"""Initial disaggregation rate and endpoint of normalised ThT traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from fluostar.config import clean_well, is_missing, raw_file_path, select_rows
from fluostar.io import TIME_COLUMN, FluostarData, load_fluostar
from fluostar.windows import LinearFit, Window, fit_linear, last_seconds, window_mean

from .normalize import normalize_tht
from .settings import ThtSettings

# config spreadsheet columns
WELL, FIBRIL_CONTROL, BASE_CONTROL = "Well", "Control F", "Control T"
DESCRIPTION_COLUMNS = ("exp", "date", "X", "M", "A", "B", "C", "D", "E")


def prepare(data: FluostarData, settings: ThtSettings) -> FluostarData:
    """Select the gain and trim the time axis. Returns a new object (safe to re-run)."""
    if data.gains != [settings.gain]:
        data = data.select_gain(settings.gain)
    return data.trim(start_s=settings.equilibration_s, end_s=settings.max_time_s)


def smooth(signal: np.ndarray, settings: ThtSettings) -> np.ndarray:
    return savgol_filter(signal, settings.smoothing_window, settings.smoothing_poly)


@dataclass(frozen=True)
class ThtKinetics:
    rate_percent_per_hour: float   # initial slope of the normalised trace, %/h
    initial_fit: LinearFit         # slope per hour of the normalised trace (fraction)
    endpoint_percent: float        # mean of the normalised trace over the final window, %
    drift: float                   # first - last value inside the final window (fraction)
    final_window: Window
    smoothed: np.ndarray
    end_time_h: float              # last time point

    def as_dict(self) -> dict:
        return {"rate_percent_per_hour": self.rate_percent_per_hour,
                "endpoint_percent": self.endpoint_percent, "drift": self.drift,
                "end_time_h": self.end_time_h, "r_squared": self.initial_fit.r_squared}


def tht_kinetics(time_s: np.ndarray, signal: np.ndarray, settings: ThtSettings) -> ThtKinetics:
    """Initial rate (linear fit, absolute-time window) and endpoint (mean of the last part)."""
    time_s, signal = np.asarray(time_s, float), np.asarray(signal, float)
    smoothed = smooth(signal, settings)

    initial_data = smoothed if settings.smooth_initial else signal
    fit = fit_linear(time_s, initial_data, Window(end_s=settings.initial_window_s), time_unit_s=3600)

    final_data = smoothed if settings.smooth_final else signal
    final_window = last_seconds(time_s, settings.final_window_s)
    final_values = final_data[final_window.mask(time_s) & np.isfinite(final_data)]

    return ThtKinetics(
        rate_percent_per_hour=fit.slope * 100,
        initial_fit=fit,
        endpoint_percent=window_mean(time_s, final_data, final_window) * 100,
        drift=float(final_values[0] - final_values[-1]),
        final_window=final_window,
        smoothed=smoothed,
        end_time_h=float(time_s.max() / 3600),
    )


def process_well(prepared: FluostarData, roi_row, settings: ThtSettings):
    """Normalise one well and compute its kinetics.

    ``prepared`` comes from ``prepare(data, settings)``; it is not changed.
    ``roi_row``: mapping or one-row DataFrame with ``well``, ``nFibril``, ``nBase``.
    Returns ``(normalised DataFrame, ThtKinetics)``.
    """
    if isinstance(roi_row, pd.DataFrame):
        if len(roi_row) != 1:
            raise ValueError("process_well needs a single roi row; loop over the roi for more wells")
        roi = roi_row
    else:
        roi = pd.DataFrame([dict(roi_row)])
    norm = normalize_tht(prepared, roi, settings.normalize_points)
    well = clean_well(roi["well"].iloc[0])
    return norm, tht_kinetics(norm[TIME_COLUMN].to_numpy(), norm[well].to_numpy(), settings)


def process_plate(prepared: FluostarData, roi: pd.DataFrame, settings: ThtSettings):
    """All wells of a roi: ``(normalised DataFrame, {well: ThtKinetics}, summary DataFrame)``."""
    norm = normalize_tht(prepared, roi, settings.normalize_points)
    kinetics, rows = {}, []
    for well in (clean_well(w) for w in roi["well"]):
        kinetics[well] = tht_kinetics(norm[TIME_COLUMN].to_numpy(), norm[well].to_numpy(), settings)
        rows.append({"well": well, "content": prepared.content(well), **kinetics[well].as_dict()})
    return norm, kinetics, pd.DataFrame(rows)


def roi_from_config_row(row) -> pd.DataFrame:
    for column in (WELL, FIBRIL_CONTROL, BASE_CONTROL):
        if is_missing(row[column]):
            raise ValueError(f"config row has no {column!r}")
    return pd.DataFrame({"well": [clean_well(row[WELL])], "nFibril": [clean_well(row[FIBRIL_CONTROL])],
                         "nBase": [clean_well(row[BASE_CONTROL])]})


def analyse_config(config: pd.DataFrame, raw_data_root: str | Path, settings: ThtSettings,
                   rows: Sequence[int] | None = None):
    """Batch over config rows. Returns ``({row: (norm, kinetics)}, summary DataFrame)``.

    Each raw file is loaded once, even when several rows use it.
    """
    cache: dict[Path, FluostarData] = {}
    results, summary = {}, []
    for k in select_rows(config, rows):
        row = config.iloc[k]
        path = raw_file_path(row, raw_data_root)
        if path not in cache:
            cache[path] = prepare(load_fluostar(path), settings)
        norm, kin = process_well(cache[path], roi_from_config_row(row), settings)
        results[k] = (norm, kin)
        description = {c: row[c] for c in DESCRIPTION_COLUMNS if c in config.columns}
        summary.append({"row": k, "well": clean_well(row[WELL]), **description, **kin.as_dict()})
    return results, pd.DataFrame(summary)
