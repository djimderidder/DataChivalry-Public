# -*- coding: utf-8 -*-
"""
Batch-process ThT disaggregation runs listed in a config spreadsheet: for every row
the initial rate and the final ThT signal.

The config has (at least) the columns  folder, filename, Well, Control F, Control T
(Control F = fibrils without ATP, Control T = no fibrils), plus free description
columns (exp, date, X = fibrils, M = monomers, A, B, C, ... = other components).
See input/ThT config template.xlsx.

Fill in the USER SETTINGS below and run this file (Spyder: "Run file", F5).
Leave raw_data_root = None to run on a simulated experiment first. Afterwards
`config`, `results` and `summary` are in the Variable Explorer.

From a terminal the same settings are used; flags override them, e.g.
    python runThtConfig.py --root D:/fluostar --config input/my_config.xlsx --rows 0,1,2 --save output
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

from fluostar.config import read_config
from fluostar.save import save_and_show
from thtdecay import ThtSettings, analyse_config
from thtdecay.plotting import plot_diagnostics

# ---------- USER SETTINGS ----------
raw_data_root = None #HERE / "data" or None for simulated experiment
config_file = HERE / "data" / "my config.xlsx"   # was: HERE / "input" / "ThT config template.xlsx"


settings = ThtSettings(
    gain=3,                 # multiple gains are measured: pick one
    equilibration_s=600,    # cut off the first 10 minutes (noisy)
    max_time_s=14 * 3600,   # cut off after 14 hours (None = keep everything)
    initial_window_s=3600,  # initial rate: linear fit over t <= 1 h (absolute time)
    final_window_s=3600,    # endpoint: mean over the last hour
    smoothing_window=51,    # Savitzky-Golay window (odd number of points)
    smoothing_poly=3,
    smooth_initial=False,
    smooth_final=False,
)

rows_to_run = None          # None = all rows, or e.g. [0, 1, 4]
diagnostic_rows = []        # rows to also show the 3-panel diagnostic plot for
show_figures = True
save_results = False        # also write the summary (CSV) and figures (PNG) to output/
# -------------------------------------------------


def command_line(argv):
    """Optional terminal flags; Spyder's Run file passes none, so the settings above are used."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=raw_data_root)
    p.add_argument("--config", type=Path, default=config_file)
    p.add_argument("--rows", default=None, help="comma separated row indices, e.g. 0,1,4")
    p.add_argument("--diagnostics", default=None, help="comma separated row indices")
    p.add_argument("--save", type=Path, default=HERE / "output" if save_results else None, metavar="DIR")
    p.add_argument("--no-show", dest="show", action="store_false", default=show_figures)
    return p.parse_args(argv)


def _int_list(text, default):
    return [int(x) for x in text.split(",")] if text else default


if __name__ == "__main__":
    args = command_line(sys.argv[1:])
    rows = _int_list(args.rows, rows_to_run)
    diagnostics_for = _int_list(args.diagnostics, diagnostic_rows)

    if args.root is None:
        from thtdecay.synthetic import write_mock_tht_experiment

        root = HERE / "data" / "simulated"
        _, config_path, _ = write_mock_tht_experiment(root)
        print(f"No raw_data_root given: running on a simulated experiment ({root})")
        diagnostics_for = diagnostics_for or [0]
    else:
        root, config_path = Path(args.root), args.config

    config = read_config(config_path, required=("folder", "filename", "Well", "Control F", "Control T"))
    results, summary = analyse_config(config, root, settings, rows)

    figures = {}
    for k in diagnostics_for:
        norm, kin = results[k]
        well = summary.loc[summary["row"] == k, "well"].iloc[0]
        figures[f"diagnostics_row{k}_{well}"] = plot_diagnostics(norm["Time [s]"], norm[well], kin,
                                                                 title=f"Run {k} ({well})")
    save_and_show(figures, {"tht_summary": summary}, args.save, args.show)
