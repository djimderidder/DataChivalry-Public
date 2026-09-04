# -*- coding: utf-8 -*-
"""
Visualize the ATPase runs listed in the config spreadsheet, using their
Starti/Endi if set, otherwise a standard fallback window. Does NOT run the
automatic or manual start/end determination - use atpasePicker.py for that.

code: is cleaned by Claude Sonnet 5
@author: ridderdde
"""
import sys, os
from pathlib import Path

import numpy as np
import pandas as pd

cwdPath = Path.cwd().parent
for functionDir in [cwdPath / ".." / "Shared", cwdPath / "ATPase_rate" / "functions"]:
    sys.path.append(str(functionDir.resolve()))

from loadData import FLUOstarData
from processData import Get_signal, Auto_select_window, Fit_rate
from visualizeData import Plot_atpase_overview, Plot_rates

# ---------------- USER SETTINGS ----------------
raw_data_root = os.environ.get("ATPASE_RAW_DATA_ROOT", "")
input_path = #cwdPath / "ATPase_rate" / "input"
config_file = "Checked atpase assay space.xlsx"

extinction_coeff = 6220
rows_to_plot = [5,6]    # None = all rows, or e.g. [0, 1, 4]

# Optional per-row style overrides, keyed by row index:
#   ROW_STYLES = {0: {'color': '#AA4466', 'linestyle': '--'}}
row_styles = {}
# -------------------------------------------------

if not raw_data_root:
    raise ValueError(
        "RAW_DATA_ROOT is not set. Set the ATPASE_RAW_DATA_ROOT environment variable "
        "to the folder containing your raw FLUOstar exports."
    )

config = pd.read_excel(input_path / config_file)
rows = range(len(config)) if rows_to_plot is None else rows_to_plot

traces = []
for k in rows:
    row = config.iloc[k]
    raw = FLUOstarData(os.path.join(raw_data_root, row['folder']), row['filename'])
    time = raw.data['Time [s]'].values.astype(float)
    sig = Get_signal(raw, row['Well'], row['Control'])

    starti, endi = row['Starti'], row['Endi']
    if pd.isna(starti) or pd.isna(endi) or starti == '-' or endi == '-':
        starti, endi = Auto_select_window(time, sig)
    else:
        starti, endi = int(starti), int(endi)

    fit = Fit_rate(time, sig, starti, endi, row['H'], Extinction_coeff=extinction_coeff)

    traces.append({
        'label': row['Well'],
        'time': time,
        'sig': sig,
        'fit': fit,
        'style': row_styles.get(k, {}),
    })

Plot_atpase_overview(traces)
Plot_rates(traces, rows_to_plot, config)
