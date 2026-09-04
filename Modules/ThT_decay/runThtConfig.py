# -*- coding: utf-8 -*-
"""
Load a FLUOstar ThT disaggregation dataset by reading folder and filename from a config file.
This code batch-process a set of this data. And gets initial rate and final tht signal.

Config spreadsheet is expected to have (at least) the columns:
    folder, filename, Well, Control F, Control T
 
@author: ridderdde
"""
import sys, os
from pathlib import Path

cwdPath = Path.cwd().parent

for functionDir in [
    cwdPath / ".." / "Shared",
    cwdPath / "ThT_decay" / "functions"
]:
    sys.path.append(str(functionDir.resolve()))

import numpy as np
import pandas as pd

from loadData import FLUOstarData
from processData import Process_well
from visualizeData import PlotThTDiagnostics

# ---------- USER SETTINGS ----------
raw_data_root = r"D:\phd_leiden\experiments\fluostar" #path were all raw data is stored

inputPath = cwdPath / "ThT_decay" / "input"
outputPath = cwdPath / "ThT_decay" / "output"
fileName = "Checked disag assay space.xlsx"

gain = 3 #often multiple gain are used to measure tht signal pick if you want the first second or third...
equilibration_time_s = 600 #cut off first 10 minutes
max_time_s = 14*3600 #cut of data after 14 hours
window_size = 51 # Savitzky-Golay smoothing window (must be odd)
poly_order = 3             
initial_window_s = 60 * 60 # initial-rate fit window
final_window_s = 60 * 60 # endpoint averaging window
 
rows_to_run = None # None = all rows, or e.g. range(0, 10) for a subset
diagnotic_rows = [] # row indices to also show the 3-panel diagnostic plot for
save_results = False # set True to write the summary csv to output folder

# -------------------------------------------------
 
if not raw_data_root:
    raise ValueError(
        "Set the raw_data_root environment variable to the folder containing your raw FLUOstar exports."
    )

config = pd.read_excel(inputPath / fileName) #config = pd.read_excel(os.path.join(inputPath,fileName))
rows = range(len(config)) if rows_to_run is None else rows_to_run
 
results = []
for k in rows:
    row = config.iloc[k]
    roi = pd.DataFrame({
        'well': [row['Well']],
        'nFibril': [row['Control F']],
        'nBase': [row['Control T']],
    })
 
    raw = FLUOstarData(os.path.join(raw_data_root, row['folder']), row['filename'])
    raw.set_gain(Gain=gain)
    raw.data = raw.data[raw.data['Time [s]'] > equilibration_time_s]
 
    norm, kinetics = Process_well(
        raw, roi,
        Max_time_s=max_time_s,
        Initial_window_s=initial_window_s,
        Final_window_s=final_window_s,
        Window_size=window_size,
        Poly_order=poly_order,
        Mode='off'
    )
 
    results.append({
        'row': k,
        'well': row['Well'],
        'rate_percent_per_hour': kinetics['initial']['rate_percent_per_hour'],
        'endpoint_percent': kinetics['final']['endpoint_percent'],
        'drift': kinetics['final']['drift'],
    })
 
    if k in diagnotic_rows:
        dataNorm = norm[row['Well']].values.astype(float)
        time_s = norm['Time [s]'].values.astype(float)
        PlotThTDiagnostics(time_s, dataNorm, kinetics, title=f"Run {k} ({row['Well']})")
 
results = pd.DataFrame(results)
print(results)
 
if save_results:
    outputPath.mkdir(exist_ok=True)
    out_file =outputPath / "tht_summary.csv"
    results.to_csv(out_file, index=False)
    print(f"Saved summary to {out_file}")
 



