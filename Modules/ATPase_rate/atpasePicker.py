# -*- coding: utf-8 -*-
"""
Find the fitting window (Starti/Endi) and the resulting rate of ATPase runs listed in
the config spreadsheet, automatically or by dragging lines.

For every row that still needs Starti/Endi (or every row in rows_to_run):
  1. Load the trace and subtract the Control well (no Control: the raw well trace).
  2. Propose a window automatically: start of the trace up to the point where the
     NADH is (95 %) used up.
  3. Plot the fit and ask whether it is acceptable.
  4. If not: open a window to drag start (light blue) / end (dark blue), then refit.
  5. Optionally write Starti/Endi/Rate to a COPY of the config (output/... (picked).xlsx).

Fill in the USER SETTINGS below and run this file (Spyder: "Run file", F5).
Leave raw_data_root = None to try it on a simulated experiment. Dragging needs an
interactive matplotlib backend (Spyder: Preferences > IPython console > Graphics >
Backend: Qt5; from a terminal this is usually the default). Afterwards `picks` holds
the chosen windows.

From a terminal the same settings are used; flags override them, e.g.
    python atpasePicker.py --root D:/fluostar --config input/my_config.xlsx --rows 13,3 --write-back
    python atpasePicker.py --accept-all        # no questions: keep every automatic window
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

import matplotlib.pyplot as plt
import pandas as pd

from atpase import AtpaseSettings, atpase_rate, auto_stop, update_config
from atpase.batch import load_trace
from atpase.plotting import plot_rate_fit
from fluostar.config import is_missing, read_config
from fluostar.selection import select_trace_window

# ---------------- USER SETTINGS ----------------
raw_data_root = None #HERE / "data" or None for simulated data
config_file = HERE / "data" / "my ATPase config.xlsx"

settings = AtpaseSettings(
    extinction_coeff=6220,  # NADH at 340 nm, M^-1 cm^-1
    path_length_cm=1.0,
    enzyme_column="A",      # config column with the ATPase concentration in uM (old spreadsheets: "H")
)

rows_to_run = None          # None = only rows missing Starti/Endi, or e.g. [13, 3]
accept_all = False          # True = no questions, keep every automatic window
write_back = False          # True = write Starti/Endi/Rate to a copy of the config in output/
# -------------------------------------------------


def command_line(argv):
    """Optional terminal flags; Spyder's Run file passes none, so the settings above are used."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=raw_data_root)
    p.add_argument("--config", type=Path, default=config_file)
    p.add_argument("--rows", default=None, help="comma separated row indices, e.g. 13,3")
    p.add_argument("--accept-all", action="store_true", default=accept_all)
    p.add_argument("--write-back", action="store_true", default=write_back)
    p.add_argument("--output", type=Path, default=HERE / "output", help="folder for the updated config copy")
    return p.parse_args(argv)


def rows_needing_a_window(config):
    return [k for k in range(len(config))
            if is_missing(config["Starti"].iloc[k]) or is_missing(config["Endi"].iloc[k])]


if __name__ == "__main__":
    args = command_line(sys.argv[1:])

    if args.root is None:
        from atpase.synthetic import write_mock_atpase_experiment

        root = HERE / "data" / "simulated"
        _, config_path, _ = write_mock_atpase_experiment(root)
        print(f"No raw_data_root given: running on a simulated experiment ({root})")
    else:
        root, config_path = Path(args.root), args.config

    config = read_config(config_path, required=("folder", "filename", "Well", "Control", "Starti", "Endi"))
    rows = [int(r) for r in args.rows.split(",")] if args.rows else (rows_to_run or rows_needing_a_window(config))

    plt.close("all")
    picked, cache = [], {}
    for k in rows:
        row = config.iloc[k]
        name = f"Row {k} ({str(row['Well']).strip()})"
        time_s, signal = load_trace(row, root, cache)
        enzyme = row[settings.enzyme_column]

        # ---- automatic proposal ----
        start, stop = 0, auto_stop(signal, settings)
        result = atpase_rate(time_s, signal, start, stop, enzyme, settings)
        print(f"{name}: automatic starti={start}, endi={stop}, rate={result.rate:.4f}")

        happy = "y"
        if not args.accept_all:
            plot_rate_fit(time_s, signal, result, title=f"{name} - automatic")
            plt.show(block=False)
            plt.pause(0.1)
            happy = input("Are you satisfied with starti/endi? [y/n]: ").strip().lower()

        # ---- manual override ----
        if happy != "y":
            plt.close("all")
            print("Opening manual picker: drag start (light blue) / end (dark blue), then click Finish.")
            # as before: Endi = index under the dark-blue line (used as slice stop, so excluded)
            start, stop = select_trace_window(time_s, signal, start_guess=time_s[start],
                                              end_guess=time_s[min(stop, len(time_s) - 1)],
                                              xlabel="Time [s]", ylabel="Signal (bg-subtracted)", title=name)
            result = atpase_rate(time_s, signal, start, stop, enzyme, settings)
            plot_rate_fit(time_s, signal, result, title=f"{name} - manual")
            print(f"{name}: manual starti={start}, endi={stop}, rate={result.rate:.4f}")
            plt.pause(3)

        picked.append({"row": k, "well": str(row["Well"]).strip(), "starti": start, "endi": stop,
                       "rate": result.rate})

    picks = pd.DataFrame(picked, columns=["row", "well", "starti", "endi", "rate"])
    print(f"\n{picks.to_string(index=False)}")

    # ---- save to a copy of the config (the original stays untouched) ----
    if args.write_back and len(picks):
        args.output.mkdir(parents=True, exist_ok=True)
        out_file = args.output / f"{Path(config_path).stem} (picked).xlsx"
        update_config(config, picks).to_excel(out_file, index=False)
        print(f"Wrote Starti/Endi/Rate to {out_file}")
    print("Done.")
