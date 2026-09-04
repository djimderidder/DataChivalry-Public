# -*- coding: utf-8 -*-
"""
Plotting functions for FLUOstar ATPase (NADH-coupled) assay data.
@author: ridderdde
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors
import palettable.cartocolors.sequential as colose
import palettable.cartocolors.qualitative as coloqu
import palettable.cartocolors.diverging as colodi


def Plot_rate_fit(Time, Trace, Fit, Title=''):
    """
    Plot a single (background-subtracted) trace with its fitted linear
    regression over [starti:endi], and markers at starti/endi.
    Fit: dict returned by processData.fit_rate
    code: is cleaned by Claude Sonnet 5
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Time / 3600, Trace, color='gray', linewidth=1.5, label='data')

    starti, endi = Fit['starti'], Fit['endi']
    x_fit = Time[starti:endi] / 3600
    y_fit = (Time[starti:endi] / 60) * Fit['slope'] + Fit['intercept']
    ax.plot(x_fit, y_fit, '--', color='crimson', linewidth=2, label='fit')

    ax.axvline(Time[starti] / 3600, color='#7EC8E3', linestyle=':', label='start')
    ax.axvline(Time[endi] / 3600, color='#00008B', linestyle=':', label='end')

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Absorbance NADH (bg-subtracted)')
    ax.set_title(f"{Title}\nrate = {Fit['rate']:.4f}, R\u00b2 = {Fit['r_value']**2:.3f}")
    ax.legend()
    plt.show()


def Plot_atpase_overview(Traces, Xlim=None, Ylim=None):
    """
    Overlay multiple ATPase traces + their linear fits in one figure.

    traces: list of dicts, each with:
        'time', 'sig'   - arrays
        'fit'           - dict from processData.fit_rate (needs starti/endi/slope/intercept)
        'label'         - legend label
        'style'         - optional dict: {'color': '#RRGGBB', 'linestyle': '--'}
    code: is cleaned by Claude Sonnet 5
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = [matplotlib.colors.rgb2hex(c) for c in coloqu.Safe_10.mpl_colors]

    for i, tr in enumerate(Traces):
        style = tr.get('style', {}) or {}
        color = style.get('color', palette[i % len(palette)])
        linestyle = style.get('linestyle', '-')

        time, sig, fit = tr['time'], tr['sig'], tr['fit']
        starti, endi = fit['starti'], fit['endi']

        x_fit = time[starti:endi] / 3600
        y_fit = (time[starti:endi] / 60) * fit['slope'] + fit['intercept']

        ax.plot(time / 3600, sig, linewidth=2, linestyle=linestyle, color=color, label=tr.get('label'))
        ax.plot(x_fit, y_fit, '--', linewidth=1.5, color=color)
        ax.axvline(time[starti] / 3600, linestyle=':', color=color, alpha=0.5)
        ax.axvline(time[endi] / 3600, linestyle=':', color=color, alpha=0.5)

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_ylabel('Absorbance NADH (bg-subtracted)')
    ax.set_xlabel('Time [hours]')
    if Xlim:
        ax.set_xlim(Xlim)
    if Ylim:
        ax.set_ylim(Ylim)
    ax.legend()
    plt.show()

def Plot_rates(Traces, Rows_to_plot, Config):
    """
    Plot ATPase rates for selected rows.
 
    Parameters
    ----------
    traces : list
    List containing fit results.
    rows_to_plot : list
    Row numbers from config corresponding to traces.
    config : pandas.DataFrame
        ATPase config table.
    """
    palette = [matplotlib.colors.rgb2hex(c) for c in coloqu.Safe_10.mpl_colors]
    rates = [Traces[i]['fit']['rate'] for i in range(len(Rows_to_plot))]

    labels = ['X\nM\nH\nD\nD dRH\nB\nA']
    for row in Rows_to_plot:
        labels.append(
            f"{Config.iloc[row]['X']}\n"
            f"{Config.iloc[row]['M']}\n"
            f"{Config.iloc[row]['H']}\n"
            f"{Config.iloc[row]['D']}\n"
            f"{Config.iloc[row]['D dRH']}\n"
            f"{Config.iloc[row]['B']}\n"
            f"{Config.iloc[row]['A']}"
        )

    fig, ax = plt.subplots(figsize=(max(8, len(rates)*1.2), 6))
    x = np.arange(len(rates))
    ax.bar(x, rates,color=palette)
    ax.set_xticks(np.append(np.array([-1]),np.arange(len(rates))))
    ax.set_xticklabels(labels)
    ax.set_ylabel('ATPase rate')
    ax.set_title('ATPase rates')
    
    plt.tight_layout()
    plt.show()
    return fig, ax