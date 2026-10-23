# -*- coding: utf-8 -*-
"""
Visualise the ATPase runs listed in the config spreadsheet: every trace with its fit,
and a bar plot of the rates.

The fitting window of every row uses Starti/Endi from the config where they are filled
in, and the automatic choice where they are missing ('-' or empty). Each bound is
chosen separately. Use atpasePicker.py to find or check windows.

Fill in the USER SETTINGS below and run this file (Spyder: "Run file", F5).
Leave raw_data_root = None to run on a simulated experiment first. Afterwards
`config`, `traces` and `summary` are in the Variable Explorer.

From a terminal the same settings are used; flags override them, e.g.
    python atpaseVisualize.py --root D:/fluostar --config input/my_config.xlsx --rows 5,6 --save output
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

from atpase import AtpaseSettings, analyse_config
from atpase.plotting import plot_overview, plot_rates
from fluostar.config import read_config
from fluostar.save import save_and_show

# ---------------- USER SETTINGS ----------------
raw_data_root = None #HERE / "data" or None for simulated data
config_file = HERE / "data" / "my ATPase config.xlsx"

settings = AtpaseSettings(
    extinction_coeff=6220,  # NADH at 340 nm, M^-1 cm^-1
    path_length_cm=1.0,     # exports are path-length corrected ("Absorbance values are pathlength correct")
    enzyme_column="A",      # config column with the ATPase concentration in uM (old spreadsheets: "H")
    label_columns=("X", "M", "A", "B", "C"),   # composition shown under the bars
)

rows_to_plot = None         # None = all rows, or e.g. [0, 1, 4]
row_styles = {}             # optional per-row styles, e.g. {0: {'color': '#AA4466', 'linestyle': '--'}}
show_figures = True
save_results = False        # also write the summary (CSV) and figures (PNG) to output/
# -------------------------------------------------


def command_line(argv):
    """Optional terminal flags; Spyder's Run file passes none, so the settings above are used."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=raw_data_root)
    p.add_argument("--config", type=Path, default=config_file)
    p.add_argument("--rows", default=None, help="comma separated row indices, e.g. 5,6")
    p.add_argument("--save", type=Path, default=HERE / "output" if save_results else None, metavar="DIR")
    p.add_argument("--no-show", dest="show", action="store_false", default=show_figures)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = command_line(sys.argv[1:])
    rows = [int(r) for r in args.rows.split(",")] if args.rows else rows_to_plot

    if args.root is None:
        from atpase.synthetic import write_mock_atpase_experiment

        root = HERE / "data" / "simulated"
        _, config_path, _ = write_mock_atpase_experiment(root)
        print(f"No raw_data_root given: running on a simulated experiment ({root})")
    else:
        root, config_path = Path(args.root), args.config

    config = read_config(config_path, required=("folder", "filename", "Well", "Control", "Starti", "Endi"))
    traces, summary = analyse_config(config, root, settings, rows)

    figures = {"atpase_overview": plot_overview(traces, row_styles),
               "atpase_rates": plot_rates(summary, settings.label_columns)}
    save_and_show(figures, {"atpase_summary": summary}, args.save, args.show)
