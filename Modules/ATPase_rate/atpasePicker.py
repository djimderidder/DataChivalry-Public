# -*- coding: utf-8 -*-
"""
Determine Starti/Endi (and the resulting rate) for each ATPase run listed in
the config spreadsheet, either automatically or by manual line-picking.

For every row that still needs Starti/Endi (or every row in ROWS_TO_RUN):
  1. Load the raw trace and background-subtract it against the Control well
     (if no Control is set, just uses the raw well trace).
  2. Propose starti/endi automatically: starti = start of trace, endi = first
     point where the trace's smoothed slope goes flat (i.e. has caught up to
     the background rate).
  3. Plot the fit and ask if it's acceptable.
  4. If not: open an interactive window (Shared/selectData.SelectTraceWindow)
     to manually drag start (light blue) / end (dark blue) lines, then refit.
  5. Ask whether to save the chosen Starti/Endi/Rate back into the config file.

Requires an interactive matplotlib backend (e.g. Qt5Agg, TkAgg) for the
manual picker to work outside of a notebook.

code: is cleaned by Claude Sonnet 5
@author: ridderdde
"""
import sys, os
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

cwdPath = Path.cwd().parent
for functionDir in [cwdPath / ".." / "Shared", cwdPath / "ATPase_rate" / "functions"]:
    sys.path.append(str(functionDir.resolve()))

from loadData import FLUOstarData
from selectData import SelectTraceWindow
from processData import Get_signal, Auto_select_window, Fit_rate
from visualizeData import Plot_rate_fit

# ---------------- USER SETTINGS ----------------
# Folder containing the raw FLUOstar exports referenced by the config's
# 'folder'/'filename' columns, e.g.:
#   export ATPASE_RAW_DATA_ROOT=/path/to/experiments/fluostar
raw_data_root = os.environ.get("ATPASE_RAW_DATA_ROOT", "")

input_path = #cwdPath / "ATPase_rate" / "input"
config_file = "Checked atpase assay space.xlsx"

extinction_coeff = 6220     # NADH extinction coefficient, M^-1 cm^-1

rows_to_run = [13,3]          # None = only rows missing Starti/Endi, or e.g. [0, 3, 7]
# -------------------------------------------------

if not raw_data_root:
    raise ValueError(
        "RAW_DATA_ROOT is not set. Set the ATPASE_RAW_DATA_ROOT environment variable "
        "to the folder containing your raw FLUOstar exports."
    )

#config = pd.read_excel(input_path / config_file)
config = pd.read_excel(os.path.join(input_path,config_file))

if rows_to_run is None:
    def _missing(v):
        return pd.isna(v) or v == '-'
    rows = [i for i in range(len(config))
            if _missing(config['Starti'].iloc[i]) or _missing(config['Endi'].iloc[i])]
else:
    rows = rows_to_run

plt.close('all')
for k in rows:
    row = config.iloc[k]
    raw = FLUOstarData(os.path.join(raw_data_root, row['folder']), row['filename'])
    time = raw.data['Time [s]'].values.astype(float)
    sig = Get_signal(raw, row['Well'], row['Control'])

    # ---- automatic proposal ----
    starti, endi = Auto_select_window(
        time, sig
    )
    fit = Fit_rate(time, sig, starti, endi, row['H'], Extinction_coeff=extinction_coeff)
    Plot_rate_fit(time, sig, fit, Title=f"Row {k} ({row['Well']}) - automatic")
    print(f"Row {k}: automatic starti={starti}, endi={endi}, rate={fit['rate']:.4f}")
    plt.show(block=False)
    plt.pause(0.1)

    happy = input("Are you satisfied with starti/endi? [y/n]: ").strip().lower()

    # ---- manual override ----
    if happy != 'y':
        plt.close('all')
        print("Opening manual picker: drag start (light blue) / end (dark blue), then click Finish.")
        starti, endi = SelectTraceWindow(
            time, sig,
            Start_guess=time[starti], End_guess=time[endi],
            Xlabel='Time [s]', Ylabel='Signal (bg-subtracted)',
            Title=f"Row {k} ({row['Well']})"
        )
        fit = Fit_rate(time, sig, starti, endi, row['H'], Extinction_coeff=extinction_coeff)
        Plot_rate_fit(time, sig, fit, Title=f"Row {k} ({row['Well']}) - manual")
        print(f"Row {k}: manual starti={starti}, endi={endi}, rate={fit['rate']:.4f}")
        plt.pause(3)

    # ---- save back to config ----
    #plt.show(block=False)
    #plt.pause(0.1)
    #update = input("Update config with these Starti/Endi/Rate? [y/n]: ").strip().lower()
    #if update == 'y':
    #    config.loc[k, 'Starti'] = starti
    #    config.loc[k, 'Endi'] = endi
    #    config.loc[k, 'Rate'] = fit['rate']
    #    config.to_excel(input_path / config_file, index=False)
    #    print(f"Row {k}: config updated and saved.")
    #else:
    #    print(f"Row {k}: not saved.")

print("Done.")
