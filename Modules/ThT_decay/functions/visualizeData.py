# -*- coding: utf-8 -*-
"""
Lets map out some data
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

#def PlotWellData:
    

def PlotThT(Raw, Norm, Roi=None):
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(12,6) #width,height
    gs3 = gridspec.GridSpec (1, 2,width_ratios=[1,1],height_ratios=[1])
    ax1 = fig.add_subplot(gs3[0,0])
    ax2 = fig.add_subplot(gs3[0,1])
    colori = np.array([])
    
    wells = Norm.columns[Norm.columns != 'Time [s]']
    y1 = Raw.data.loc[:,Raw.data.columns != 'Time [s]']
    x1 = Raw.data['Time [s]']
    y2 = Norm.loc[:,Norm.columns != 'Time [s]']
    x2 = Norm['Time [s]']

    for count, i in enumerate(wells):
        if len(Norm.columns)<11:
            colorOI = matplotlib.colors.rgb2hex(coloqu.Safe_10.mpl_colors[count])  # Use count to index colors
        elif count<7:
            colorOI = matplotlib.colors.rgb2hex(colose.Emrld_7.mpl_colors[count])
        elif count<14:
            colorOI = matplotlib.colors.rgb2hex(colose.Burg_7.mpl_colors[count-7]) 
        elif count<21:
            colorOI = matplotlib.colors.rgb2hex(colose.Mint_7.mpl_colors[count-14]) 
        else:
            colorOI = matplotlib.colors.rgb2hex(coloqu.Safe_10.mpl_colors[0])
        colori = np.append(colori, colorOI)
    
        ax1.plot(x1 / 60, y1[i], linewidth=2, color=colori[count])  # Use count for colori index
        ax2.plot(x2 / 60, y2[i], linewidth=2, color=colori[count])

    
    if Roi is not None:
        for count, i in enumerate(wells):
            nBase = Roi.loc[Roi['well'] == i, 'nBase']
            nFibril = Roi.loc[Roi['well'] == i, 'nFibril']
            ax1.plot(x1/60,Raw.data[nBase],':',color=colori[count])
            ax1.plot(x1/60,Raw.data[nFibril],':',color=colori[count])


    ax1.set_title('Raw data')
    ax1.ticklabel_format(style = 'sci', axis='y', scilimits=(0,0))
    ax1.set_ylabel('ThT Fluorescence')
    ax1.set_xlabel('time [min]')
    ax1.set_ylim([0,8e4])
    ax2.set_title('Normalised data')
    ax2.set_xlabel('time [min]')
    ax2.legend([Raw.dict.get(coord) for coord in wells])
    ax2.set_ylim([0,1.2])
    plt.show()
    #ax2.axvline(x=12075/60,ymin=-2,ymax=2,color='k')
def PlotThTDiagnostics(Time_s, Data, Kinetics, Title=''):
    """
    3-panel diagnostic plot for a single well's kinetics:
      1) raw + smoothed trace
      2) initial slope fit
      3) final endpoint window
 
    time_s, Data: arrays for the normalized trace
    kinetics: dict returned by processData.compute_tht_kinetics
    """
    yhat = Kinetics['smoothed']
    initial = Kinetics['initial']
    final = Kinetics['final']
 
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(18, 5)
    gs = gridspec.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
 
    # --- Panel 1: raw + smoothed ---
    ax1.plot(Time_s / 3600, Data, label='raw', color='gray')
    ax1.plot(Time_s / 3600, yhat, label='smoothed', color='blue')
    ax1.set_title(f"{Title}: raw + smoothed".strip(': '))
    ax1.set_xlabel("time [h]")
    ax1.set_ylabel("normalized ThT")
    ax1.legend()
 
    # --- Panel 2: initial slope fit ---
    mask_init = initial['mask']
    t_lo = Time_s[mask_init].min() / 3600
    t_hi = Time_s[mask_init].max() / 3600
    pad = max((t_hi - t_lo), 0.1)
    xfit = np.linspace(t_lo - pad, t_hi + pad, 200)
    yfit = xfit * 60 * initial['slope'] + initial['intercept']
 
    ax2.plot(Time_s[mask_init] / 3600, Data[mask_init], 'o', color='black')
    ax2.plot(xfit, yfit, '--', color='red')
    ax2.set_title(f"Initial slope\n{initial['rate_percent_per_hour']:.2f} %/h")
    ax2.set_xlabel("time [h]")
    ax2.set_ylabel("normalized ThT")
 
    # --- Panel 3: final endpoint window ---
    mask_final = final['mask']
    ax3.plot(Time_s / 3600, Data, color='gray')
    ax3.plot(Time_s[mask_final] / 3600, Data[mask_final], 'o', color='green')
    ax3.set_title(f"Final endpoint\n{final['endpoint_percent']:.1f} %")
    ax3.set_xlabel("time [h]")
    ax3.set_ylabel("normalized ThT")
 
    plt.show()
 
    print(f"Initial slope window: {initial['t_start']:.0f}s -> {initial['t_end']:.0f}s")
    print(f"Final endpoint window: {final['t_start']:.0f}s -> {final['t_end']:.0f}s")
    print(f"Rate: {initial['rate_percent_per_hour']:.3f} %/h")
    print(f"Endpoint: {final['endpoint_percent']:.3f} %")
