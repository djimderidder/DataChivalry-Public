"""Figures for lifetime fits and anisotropy spectra.

Every function returns matplotlib Figure(s) and never calls ``plt.show()``, so
the caller decides whether to show or save (see runFluoTime.py).

Labels: pass ``label_fmt`` as a format string using the dataset name and its
metadata, e.g. ``"{concentration:g} µM"`` or ``"{name}"`` (default).
"""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .lifetime.fit import LifetimeResult
from .lifetime.results import fractions


def format_label(name: str, metadata: Mapping, label_fmt: str | None) -> str:
    if not label_fmt:
        return name
    try:
        return label_fmt.format(name=name, **metadata)
    except (KeyError, ValueError, IndexError):
        return name


# --- lifetime --------------------------------------------------------------------

def plot_decay_fit(result: LifetimeResult, index: int, label_fmt: str | None = None,
                   ylim: tuple[float, float] | None = (0.8, None)) -> Figure:
    """Data, fit and scaled IRF (log scale) with weighted residuals underneath."""
    decay = result.datasets[index]
    label = format_label(decay.name, decay.metadata, label_fmt)
    t, counts = decay.time, decay.counts
    irf = result.irf_on_grid(index)

    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(t, counts, lw=0.8, color="0.4", label=f"data ({label})")
    ax.plot(t, result.curve(index), lw=1.8, color="C1", label="fit")
    ax.plot(t, irf / irf.max() * counts.max(), ls=":", color="C0", label="IRF (scaled)")
    ax.set_yscale("log")
    if ylim:
        ax.set_ylim(ylim[0], ylim[1] or counts.max() * 2)
    ax.set_ylabel("Counts")
    taus = ", ".join(f"τ{i + 1}={tau:.2f}" for i, tau in enumerate(result.taus))
    ax.set_title(f"{label}   ({taus} ns,  χ²ᵣ={result.reduced_chi2(index):.2f})")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    ax_res.plot(t, result.weighted_residuals(index), lw=0.6, color="C3")
    ax_res.axhline(0, color="k", lw=0.8)
    ax_res.set_xlabel("Time (ns)")
    ax_res.set_ylabel("Weighted\nresiduals")
    ax_res.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_lifetime_fits(result: LifetimeResult, label_fmt: str | None = None) -> list[Figure]:
    return [plot_decay_fit(result, k, label_fmt) for k in range(len(result.datasets))]


def plot_fractions(result: LifetimeResult, kind: str = "amplitude",
                   label_fmt: str | None = None) -> Figure:
    """Stacked bars of fractional amplitudes or intensities per dataset."""
    taus = result.taus
    table = np.array([fractions(result.amplitudes(k), taus, kind) for k in range(len(result.datasets))])
    labels = [format_label(d.name, d.metadata, label_fmt) for d in result.datasets]

    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(labels) + 2), 4.5))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for i in range(result.config.n_exp):
        ax.bar(x, table[:, i], bottom=bottom, label=f"τ{i + 1} = {taus[i]:.2f} ns")
        bottom += table[:, i]
    ax.set_xticks(x, labels, rotation=30 if len(labels) > 5 else 0, ha="right" if len(labels) > 5 else "center")
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Fractional {kind}")
    ax.legend(frameon=False, ncols=min(result.config.n_exp, 4),
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    return fig


# --- anisotropy ----------------------------------------------------------------

def _window_lines(ax, center, half_width):
    if center is None:
        return
    ax.axvline(center, color="k", ls="--", alpha=0.5)
    for edge in (center - half_width, center + half_width):
        ax.axvline(edge, color="0.7", ls="--", alpha=0.7)


def plot_spectra(spectra: Mapping[str, pd.DataFrame], column: str = "r",
                 labels: Mapping[str, str] | None = None, center: float | None = None,
                 half_width: float = 5.0, ylim: tuple[float, float] | None = None) -> Figure:
    """Overlay one column (``"r"`` or ``"G"``) of several anisotropy spectra."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, df in spectra.items():
        ax.plot(df["wavelength"], df[column], label=(labels or {}).get(name, name))
    _window_lines(ax, center, half_width)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel({"r": "Anisotropy r", "G": "G factor (HV/HH)"}.get(column, column))
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_components(df: pd.DataFrame, title: str = "", center: float | None = None,
                    half_width: float = 5.0) -> Figure:
    """The four processed polarisation channels of one sample."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, ch in zip(axes.flat, ("VV", "VH", "HV", "HH")):
        ax.plot(df["wavelength"], df[ch])
        ax.set_title(ch)
        _window_lines(ax, center, half_width)
        ax.set_ylabel("Intensity (a.u.)")
    for ax in axes[1]:
        ax.set_xlabel("Wavelength (nm)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
