# -*- coding: utf-8 -*-
"""
Load one FLUOstar ThT disaggregation plate, normalise the selected wells against
their controls and compute the initial rate and endpoint of every well.

Fill in the USER SETTINGS below and run this file (Spyder: "Run file", F5).
Leave fileName = None to run on a simulated plate first. Afterwards `data`,
`prepared`, `norm`, `kinetics` and `summary` are in the Variable Explorer.

From a terminal the same settings are used; flags override them, e.g.
    python runTht.py --folder D:/fluostar/20250509 --file "plate.csv" --wells A9,B9 --save output
@author: ridderdde
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for folder in (HERE / "src", HERE.parents[1] / "Shared" / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import argparse
import tempfile

import pandas as pd

from fluostar.io import load_fluostar
from fluostar.save import save_and_show
from thtdecay import ThtSettings, prepare
from thtdecay.kinetics import process_plate
from thtdecay.plotting import plot_diagnostics, plot_tht
from thtdecay.synthetic import write_mock_tht_experiment

# ---------- USER SETTINGS ----------
fileFolder = HERE / "data" / "202505 tht disag" / "20250509" 
fileName = None # "ThT disaggregation 3 gains.csv" or None to switches to simulated data

wells = None                # wells to analyse, e.g. ["A9", "B9", "C9"]; None = first 6 wells
fibril_control = "H10"      # fibrils without ATP (no disaggregation)   -> nFibril
base_control = "H11"        # no fibrils (background ThT)               -> nBase

settings = ThtSettings(
    gain=3,                 # multiple gains are measured: pick one
    equilibration_s=600,    # cut off the first 10 minutes (noisy)
    max_time_s=None,        # e.g. 14 * 3600 to cut off after 14 hours
    initial_window_s=3600,  # initial rate: linear fit over t <= 1 h (absolute time)
    final_window_s=3600,    # endpoint: mean over the last hour
    smoothing_window=51,    # Savitzky-Golay window (odd number of points)
    smoothing_poly=3,
    smooth_initial=False,   # use the smoothed trace for the initial rate
    smooth_final=False,     # use the smoothed trace for the endpoint
)

diagnostic_wells = []       # wells to also show the 3-panel diagnostic plot for, e.g. ["A9"]
show_figures = True
save_results = False        # also write summary (CSV) and figures (PNG) to output/
# -------------------------------------------------


def command_line(argv):
    """Optional terminal flags; Spyder's Run file passes none, so the settings above are used."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default=fileFolder)
    p.add_argument("--file", default=fileName)
    p.add_argument("--wells", default=None, help="comma separated, e.g. A9,B9")
    p.add_argument("--diagnostics", default=None, help="comma separated wells")
    p.add_argument("--save", type=Path, default=HERE / "output" if save_results else None, metavar="DIR")
    p.add_argument("--no-show", dest="show", action="store_false", default=show_figures)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = command_line(sys.argv[1:])
    wells_to_run = args.wells.split(",") if args.wells else wells
    diagnostics_for = args.diagnostics.split(",") if args.diagnostics else diagnostic_wells

    # 1 - load the plate (or a simulated one)
    if args.file is None:
        mock_folder = HERE / "data" / "simulated"
        csv_path, _, plate = write_mock_tht_experiment(mock_folder)
        print(f"No fileName given: running on a simulated plate ({csv_path})")
        data = load_fluostar(csv_path)
        wells_to_run = wells_to_run or list(plate.truth)
        fibril, base = plate.fibril_control, plate.base_control
        diagnostics_for = diagnostics_for or wells_to_run[:1]
    else:
        data = load_fluostar(args.folder, args.file)
        fibril, base = fibril_control, base_control
        wells_to_run = wells_to_run or data.wells[:6]

    # 2 - gain + trimming (returns a new object; `data` keeps everything)
    prepared = prepare(data, settings)

    # 3 - normalise and compute the kinetics of every well
    roi = pd.DataFrame({"well": wells_to_run, "nFibril": fibril, "nBase": base})
    norm, kinetics, summary = process_plate(prepared, roi, settings)

    # 4 - figures
    figures = {"tht_overview": plot_tht(prepared, norm, roi)}
    for well in diagnostics_for:
        figures[f"diagnostics_{well}"] = plot_diagnostics(norm["Time [s]"], norm[well], kinetics[well], title=well)
    save_and_show(figures, {"tht_summary": summary}, args.save, args.show)
