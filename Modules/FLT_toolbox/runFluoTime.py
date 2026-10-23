# -*- coding: utf-8 -*-
"""
Analyse FluoTime data: global lifetime fits and steady-state anisotropy spectra.

Fill in the USER SETTINGS below and run this file (Spyder: "Run file", F5).
Leave analysis = "demo" to run on simulated data first, so you can see what the
analysis does without any data. Afterwards the results (`result`, `summary`,
`spectra`, ...) are in the Variable Explorer.

From a terminal the same settings are used; flags override them without editing
this file, e.g.
    python runFluoTime.py --analysis lifetime --data D:/flt/run2 --save output

Measured data is not part of the repository (.dat files are git-ignored).
@author: ridderdde
"""
import sys
from pathlib import Path

# Make the package importable without installing it. Based on this file's location
# (not the working directory), so it works from Spyder, a terminal or anywhere else.
HERE = Path(__file__).resolve().parent
SRC = str(HERE / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ---------- USER SETTINGS ----------
analysis = "demo"           # "demo" (simulated data), "lifetime" or "anisotropy"
fileFolder = HERE / "data"  # folder with the .dat files, e.g. r"D:\experiments\fluotime\run2"

show_figures = True         # show the figures (Spyder: Plots pane)
save_results = False        # also write figures (PNG) and tables (CSV) to output/

# --- lifetime ---
# Model + constraints live in a config file (configs/), so other models (1-4
# exponentials, fixed/bounded lifetimes) only need another file.
fit_config = HERE / "configs" / "biexp_af488.toml"
fit_overrides = {                 # optional: override fields of the config file
    "irf_source": "reference",    # "reference" | "own"
    "shift_mode": "global",       # "global" | "per_dataset" | "anchored"
}
lifetime_files = [                # extra keys become metadata (summary table + labels)
    {"file": "sample_0uM.dat", "concentration": 0.00, "condition": "buffer"},
    {"file": "sample_0.08uM.dat", "concentration": 0.08, "condition": "buffer"},
    {"file": "sample_0.40uM.dat", "concentration": 0.40, "condition": "buffer"},
    {"file": "sample_2uM.dat", "concentration": 2.00, "condition": "buffer"},
]
lifetime_label = "{concentration:g} µM {condition}"   # plot labels from metadata
lifetime_columns = {"time_col": 0, "counts_col": 1, "irf_col": 2}  # 0-based; irf_col=None if no IRF

# --- anisotropy ---
blank_file = "blank.dat"          # wavelength VV VH; None = no blank subtraction
blank_columns = ("VV", "VH")      # channels in the blank file, in order
anisotropy_files = [              # wavelength VV VH HH HV
    {"file": "ani_0uM.dat", "concentration": 0.00},
    {"file": "ani_0.08uM.dat", "concentration": 0.08},
    {"file": "ani_0.40uM.dat", "concentration": 0.40},
    {"file": "ani_2uM.dat", "concentration": 2.00},
]
anisotropy_label = "{concentration:g} µM"
target_wavelength = 516           # nm, centre of the averaging window
half_width = 5                    # nm
smooth_sigma = 2                  # Gaussian smoothing in data points; None = off
smooth_blank = False              # original behaviour: blank not smoothed
subtract_from = ("VV", "VH")      # channels that get the blank subtracted
# -----------------------------------

import argparse  # noqa: E402  (imports after the settings on purpose: settings first)

import matplotlib.pyplot as plt  # noqa: E402

from fluotime.anisotropy import AnisotropyConfig, analyse_anisotropy  # noqa: E402
from fluotime.io import load_decay, load_spectrum  # noqa: E402
from fluotime.lifetime import FitConfig, fit_global  # noqa: E402
from fluotime.plotting import (  # noqa: E402
    format_label, plot_components, plot_fractions, plot_lifetime_fits, plot_spectra)
from fluotime.synthetic import make_synthetic_decay, make_synthetic_spectrum  # noqa: E402


# --- helpers ------------------------------------------------------------------------

def resolve(folder, filename):
    path = Path(folder) / filename
    if not path.exists():
        raise FileNotFoundError(f"{path}\nCheck `fileFolder` and the file lists in the USER SETTINGS.")
    return path


def split_entry(entry):
    """{"file": ..., **metadata} -> (filename, metadata)"""
    return entry["file"], {k: v for k, v in entry.items() if k != "file"}


def output(figures, tables, save_to=None, show=True):
    """Print the tables; optionally save figures + tables to a folder; show the figures."""
    for name, df in tables.items():
        print(f"\n=== {name} ===\n{df.to_string(index=False)}")
    if save_to is not None:
        save_to = Path(save_to)
        save_to.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(save_to / f"{name}.csv", index=False)
        for name, fig in figures.items():
            fig.savefig(save_to / f"{name}.png", dpi=150)
        print(f"\nWrote {len(tables)} table(s) and {len(figures)} figure(s) to {save_to}")
    if show:
        plt.show()
    else:
        for fig in figures.values():
            plt.close(fig)


# --- analyses -----------------------------------------------------------------------

def run_lifetime(datasets, cfg, label_fmt, save_to=None, show=True):
    result = fit_global(datasets, cfg)
    print(result.report())

    figures = {f"fit_{k:02d}_{d.name}": fig
               for k, (d, fig) in enumerate(zip(datasets, plot_lifetime_fits(result, label_fmt)))}
    figures["fractions_amplitude"] = plot_fractions(result, "amplitude", label_fmt)
    figures["fractions_intensity"] = plot_fractions(result, "intensity", label_fmt)
    output(figures, {"lifetime_summary": result.summary()}, save_to, show)
    return result


def run_anisotropy(samples, blank, cfg, label_fmt, save_to=None, show=True):
    spectra, summary = analyse_anisotropy(samples, blank, cfg)
    labels = {s.name: format_label(s.name, s.metadata, label_fmt) for s in samples}
    window = dict(center=cfg.target_wavelength, half_width=cfg.half_width)

    figures = {
        "anisotropy": plot_spectra(spectra, "r", labels, ylim=(0, 0.4), **window),
        "g_factor": plot_spectra(spectra, "G", labels, **window),
    }
    for k, (name, df) in enumerate(spectra.items()):
        figures[f"components_{k:02d}_{name}"] = plot_components(df, labels[name], **window)
    output(figures, {"anisotropy_summary": summary}, save_to, show)
    return spectra, summary


def load_lifetime_data(folder):
    datasets = []
    for entry in lifetime_files:
        filename, meta = split_entry(entry)
        datasets.append(load_decay(resolve(folder, filename), metadata=meta, **lifetime_columns))
    return datasets


def load_anisotropy_data(folder):
    blank = load_spectrum(resolve(folder, blank_file), channels=blank_columns) if blank_file else None
    samples = []
    for entry in anisotropy_files:
        filename, meta = split_entry(entry)
        samples.append(load_spectrum(resolve(folder, filename), metadata=meta))
    return samples, blank


def anisotropy_settings(target=None):
    return AnisotropyConfig(target_wavelength=target or target_wavelength, half_width=half_width,
                            smooth_sigma=smooth_sigma, smooth_blank=smooth_blank,
                            blank_channels=tuple(subtract_from))


def demo_lifetime_data():
    """Simulated titration: the 1.5 ns component grows with concentration."""
    return [
        make_synthetic_decay(f"demo_{c:g}", amplitudes=(3e5 * (1 - b), 3e5 * b), taus=(4.1, 1.5),
                             shift=0.1, background=2.0, scatter=500.0, seed=k,
                             metadata={"concentration": c})
        for k, (c, b) in enumerate(zip([0.0, 0.1, 0.5, 2.0], [0.0, 0.2, 0.45, 0.7]))
    ]


def demo_anisotropy_data():
    samples = []
    for k, (c, r) in enumerate(zip([0.0, 0.1, 0.5, 2.0], [0.05, 0.10, 0.18, 0.25])):
        s = make_synthetic_spectrum(r=r, g=0.9, blank_level=50.0, seed=k, name=f"demo_{c:g}")
        s.metadata["concentration"] = c
        samples.append(s)
    blank = make_synthetic_spectrum(r=0, g=0.9, peak=0, blank_level=50.0,
                                    channels=("VV", "VH"), noise=False, name="blank")
    return samples, blank


def command_line_overrides(argv):
    """Optional terminal flags. Spyder's Run file passes none, so the settings above are used."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", choices=["demo", "lifetime", "anisotropy"], default=analysis)
    parser.add_argument("--data", type=Path, default=Path(fileFolder), help="folder with the .dat files")
    parser.add_argument("--config", type=Path, default=fit_config, help="lifetime fit config file")
    parser.add_argument("--save", type=Path, default=HERE / "output" if save_results else None,
                        metavar="DIR", help="write figures (PNG) and tables (CSV) to DIR")
    parser.add_argument("--no-show", dest="show", action="store_false", default=show_figures,
                        help="do not show the figures")
    return parser.parse_args(argv)


# --- run ----------------------------------------------------------------------------
# Top-level variables on purpose: after "Run file" they show up in Spyder's Variable Explorer.

if __name__ == "__main__":
    args = command_line_overrides(sys.argv[1:])

    if args.analysis == "lifetime":
        cfg = FitConfig.from_file(args.config).with_overrides(**fit_overrides)
        datasets = load_lifetime_data(args.data)
        result = run_lifetime(datasets, cfg, lifetime_label, args.save, args.show)
        summary = result.summary()

    elif args.analysis == "anisotropy":
        samples, blank = load_anisotropy_data(args.data)
        spectra, summary = run_anisotropy(samples, blank, anisotropy_settings(),
                                          anisotropy_label, args.save, args.show)

    else:  # demo
        cfg = FitConfig.from_file(HERE / "configs" / "biexp_af488.toml").with_overrides(
            taus=[{"value": 4.1, "min": 3.7, "max": 4.2}, {"value": 2.0, "min": 0.05, "max": 3.8}])
        datasets = demo_lifetime_data()
        result = run_lifetime(datasets, cfg, "{concentration:g} µM", args.save, args.show)
        summary = result.summary()

        samples, blank = demo_anisotropy_data()
        spectra, anisotropy_summary = run_anisotropy(
            samples, blank, anisotropy_settings(target=520), "{concentration:g} µM", args.save, args.show)
