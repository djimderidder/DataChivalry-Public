# -*- coding: utf-8 -*-
"""
Processing functions for FLUOstar ATPase (NADH-coupled) assay data.
@author: ridderdde
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter


def Get_signal(Raw, Well, Control=None):
    """
    Return the background-subtracted signal (well - control) if a control
    well is given, otherwise just the raw well signal.
    code: is cleaned by Claude Sonnet 5
    """
    no_control = (
        Control is None
        or Control == '-'
        or (isinstance(Control, float) and np.isnan(Control))
    )
    if no_control:
        return Raw.data[Well].values.astype(float)
    return (Raw.data[Well] - Raw.data[Control]).values.astype(float)


def Auto_select_window(Time, Data):
    """
    Automatic starti/endi for an ATPase run.
 
    starti = start of the trace (index 0)
    endi = first point where Data drops below 95% of the mean of the

    last 5 datapoints.

    """
    starti = 0
    
    mean_last5 = np.mean(Data[-5:])
    threshold = 0.95 * mean_last5
    
    candidates = np.where(Data <= threshold)[0]
    
    endi = candidates[0] if len(candidates) else len(Data) - 1
    
    return starti, int(endi)


def Fit_rate(Time, Data, Starti, Endi, Conc_uM, Extinction_coeff=6220):
    """
    Linear fit of `sig` vs time (minutes) over [starti:endi], converted to a
    concentration-normalized rate using the NADH extinction coefficient.

    conc_uM: enzyme/substrate concentration in uM used for normalization
             (config column 'H').
    """
    res = stats.linregress(Time[Starti:Endi] / 60, Data[Starti:Endi])
    rate = -1 / (Extinction_coeff * Conc_uM * 1e-6) * res.slope
    return {
        'rate': rate,
        'slope': res.slope,
        'intercept': res.intercept,
        'r_value': res.rvalue,
        'starti': Starti,
        'endi': Endi,
    }