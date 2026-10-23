"""Figures. Functions return the figure; the run scripts decide to show or save."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fluostar.config import clean_well
from fluostar.io import TIME_COLUMN, FluostarData

from .kinetics import ThtKinetics


def _palette(n: int) -> list:
    """CARTOColors (palettable) as in the original plots, matplotlib colours otherwise."""
    try:
        import palettable.cartocolors.qualitative as qualitative
        import palettable.cartocolors.sequential as sequential
    except ImportError:
        return [plt.get_cmap("tab10" if n <= 10 else "viridis")(i / max(n - 1, 1) if n > 10 else i)
                for i in range(n)]
    if n < 11:
        return qualitative.Safe_10.mpl_colors[:n]
    blocks = sequential.Emrld_7.mpl_colors + sequential.Burg_7.mpl_colors + sequential.Mint_7.mpl_colors
    return [blocks[i] if i < len(blocks) else qualitative.Safe_10.mpl_colors[0] for i in range(n)]


def plot_tht(raw: FluostarData, norm: pd.DataFrame, roi: pd.DataFrame | None = None,
             raw_ylim=None, norm_ylim=(0, 1.2)):
    """Raw traces (controls dotted) next to the normalised traces."""
    wells = [c for c in norm.columns if c != TIME_COLUMN]
    colours = _palette(len(wells))
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)

    t_raw = raw.time_s / 60
    for colour, well in zip(colours, wells):
        ax_raw.plot(t_raw, raw.well(well), linewidth=2, color=colour)
        ax_norm.plot(norm[TIME_COLUMN] / 60, norm[well], linewidth=2, color=colour,
                     label=raw.content(well) or well)
        if roi is not None:
            match = roi[[clean_well(w) == well for w in roi["well"]]]
            for control in ("nBase", "nFibril"):
                if len(match):
                    ax_raw.plot(t_raw, raw.well(match[control].iloc[0]), ":", color=colour)

    ax_raw.set_title("Raw data")
    ax_raw.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_raw.set_ylabel("ThT fluorescence")
    ax_raw.set_xlabel("time [min]")
    if raw_ylim:
        ax_raw.set_ylim(raw_ylim)
    ax_norm.set_title("Normalised data")
    ax_norm.set_xlabel("time [min]")
    ax_norm.set_ylim(norm_ylim)
    ax_norm.legend(frameon=False, fontsize=8)
    return fig


def plot_diagnostics(time_s: np.ndarray, signal: np.ndarray, kinetics: ThtKinetics, title: str = ""):
    """Three panels: raw + smoothed trace, initial-rate fit, endpoint window."""
    time_s, signal = np.asarray(time_s, float), np.asarray(signal, float)
    hours = time_s / 3600
    fit = kinetics.initial_fit
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)

    ax1.plot(hours, signal, color="gray", label="raw")
    ax1.plot(hours, kinetics.smoothed, color="blue", label="smoothed")
    ax1.set_title(f"{title}: raw + smoothed".strip(": "))
    ax1.legend()

    init = fit.window.mask(time_s)
    ax2.plot(hours[init], signal[init], "o", color="black")
    span = np.linspace(hours[init].min(), hours[init].max(), 100)
    pad = max(span[-1] - span[0], 0.1)
    xfit = np.linspace(span[0] - pad, span[-1] + pad, 200)
    ax2.plot(xfit, fit.slope * xfit + fit.intercept, "--", color="red")
    ax2.set_title(f"Initial slope\n{kinetics.rate_percent_per_hour:.2f} %/h")

    final = kinetics.final_window.mask(time_s)
    ax3.plot(hours, signal, color="gray")
    ax3.plot(hours[final], signal[final], "o", color="green")
    ax3.set_title(f"Final endpoint\n{kinetics.endpoint_percent:.1f} %")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("time [h]")
        ax.set_ylabel("normalised ThT")
    return fig
