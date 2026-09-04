# -*- coding: utf-8 -*-
"""
Processing functions for FLUOstar ThT disaggregation data.

@author: ridderdde
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter


def NormalizeThT(Data, Roi):
    """
    Baseline-subtract and normalize ThT traces against fibril/base controls.

    Data: FLUOstarData instance
    Roi: DataFrame with columns ['well', 'nFibril', 'nBase']
    """
    roi = Roi['well']
    output = pd.DataFrame(index=Data.data.index, columns=Roi['well'])

    for j in range(len(Roi)):
        # baseline subtraction
        output[roi.iloc[j]] = Data.data[roi.iloc[j]] - Data.data[Roi['nBase'].iloc[j]]
        fiber = Data.data[Roi['nFibril'].iloc[j]] - Data.data[Roi['nBase'].iloc[j]]
        fiber = fiber / fiber.iloc[0]
        # relative change over time
        output[roi.iloc[j]] = output[roi.iloc[j]] / output[roi.iloc[j]].iloc[0]
        output[roi.iloc[j]] = output[roi.iloc[j]] / fiber

    output['Time [s]'] = Data.data['Time [s]']
    return output


def Smooth_tht(Data, Window_size=51, Poly_order=3):
    """
    Smooth ThT trace with Savitzky-Golay.
    """
    return savgol_filter(Data, Window_size, Poly_order)


def Compute_initial_rate(Time_s, Data, Window_s=3600):
    """
    Gets initial rate by linear fit to the first `window_s` seconds.
    Returns the rate in fraction/h.
    """
    mask = Time_s <= Window_s
    result = stats.linregress(Time_s[mask] / 60, Data[mask])
    return {
        'rate_percent_per_hour': result.slope * 60 * 100,
        'slope': result.slope, # units: fraction per minute (in case normalized)
        'intercept': result.intercept,
        't_start': Time_s[mask].min(),
        't_end': Time_s[mask].max(),
        'mask': mask,
    }


def Compute_final_level(Time_s, Data, Window_s=3600):
    """
    Get final tht signal (as fraction) by taking the mean over the last `window_s` seconds
    """
    mask = Time_s >= (Time_s.max() - Window_s)
    data_final = Data[mask]
    return {
        'endpoint_percent': np.mean(data_final) * 100,
        'drift': data_final[0] - data_final[-1],
        't_start': Time_s[mask].min(),
        't_end': Time_s[mask].max(),
        'mask': mask,
    }


def Compute_tht_kinetics(Time_s, Data, Initial_window_s=3600, Final_window_s=3600,
        Window_size=51, Poly_order=3, Mode='off'):
    """
    mode : {'off', 'initial', 'final', 'both'}, default='off'
        Controls whether smoothed data are used for metric calculation.

        - 'off'     : use raw data for all calculations
        - 'initial' : use smoothed data only for the initial rate
        - 'final'   : use smoothed data only for the final level
        - 'both'    : use smoothed data for both metrics
    """

    smoothed = Smooth_tht(Data, Window_size, Poly_order)

    initial_data = smoothed if Mode in ['initial', 'both'] else Data
    final_data = smoothed if Mode in ['final', 'both'] else Data

    return {
        'smoothed': smoothed,
        'initial': Compute_initial_rate(Time_s, initial_data, Initial_window_s),
        'final': Compute_final_level(Time_s, final_data, Final_window_s),
    }


def Process_well(Raw, Roi, Max_time_s=None, Initial_window_s=3600, Final_window_s=3600,
                  Window_size=51, Poly_order=3, Mode='off'):
    """
    Normalize a single well and compute its ThT kinetics summary.

    raw: FLUOstarData instance, already gain-selected and equilibration-trimmed.
    roi: single-row DataFrame with ['well', 'nFibril', 'nBase'].

    Returns (norm, kinetics) where norm is the normalized DataFrame
    (useful for plotting) and kinetics is the dict from compute_tht_kinetics.
    """
    if Max_time_s is not None:
        Raw.data = Raw.data[Raw.data['Time [s]'] <= Max_time_s]

    norm = NormalizeThT(Raw, Roi)
    well = Roi['well'].iloc[0]
    data = norm[well].values.astype(float)
    time_s = norm['Time [s]'].values.astype(float)

    kinetics = Compute_tht_kinetics(
        time_s, data,
        Initial_window_s=Initial_window_s,
        Final_window_s=Final_window_s,
        Window_size=Window_size,
        Poly_order=Poly_order,
        Mode = Mode
    )
    return norm, kinetics