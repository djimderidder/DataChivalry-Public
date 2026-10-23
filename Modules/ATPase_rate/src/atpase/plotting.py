"""Figures. Functions return the figure; the run scripts decide to show or save."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .batch import AtpaseTrace
from .rate import AtpaseFit


def _palette(n: int) -> list:
    try:
        import palettable.cartocolors.qualitative as qualitative
        colours = qualitative.Safe_10.mpl_colors
    except ImportError:
        colours = [plt.get_cmap("tab10")(i) for i in range(10)]
    return [colours[i % len(colours)] for i in range(n)]


def _draw_fit(ax, time_s, result: AtpaseFit, colour, alpha=1.0):
    t = time_s[result.start:result.stop]
    ax.plot(t / 3600, result.fit.line(t), "--", color=colour, linewidth=2, alpha=alpha)


def plot_rate_fit(time_s, signal, result: AtpaseFit, title: str = ""):
    """One trace with its fit and the start (light blue) / end (dark blue) markers."""
    time_s = np.asarray(time_s, float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(time_s / 3600, signal, color="gray", linewidth=1.5, label="data")
    _draw_fit(ax, time_s, result, "crimson")
    ax.axvline(time_s[result.start] / 3600, color="#7EC8E3", linestyle=":", label="start")
    ax.axvline(time_s[result.stop - 1] / 3600, color="#00008B", linestyle=":", label="end")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Absorbance NADH (bg-subtracted)")
    ax.set_title(f"{title}\nrate = {result.rate:.4f} /min, R² = {result.fit.r_squared:.3f}".strip())
    ax.legend()
    return fig


def plot_overview(traces: Sequence[AtpaseTrace], styles: dict | None = None, xlim=None, ylim=None):
    """All traces with their fits; ``styles = {row: {'color': ..., 'linestyle': ...}}``."""
    styles = styles or {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for colour, tr in zip(_palette(len(traces)), traces):
        style = styles.get(tr.row, {})
        colour = style.get("color", colour)
        ax.plot(tr.time_s / 3600, tr.signal, linewidth=2, linestyle=style.get("linestyle", "-"),
                color=colour, label=f"{tr.well} (row {tr.row})")
        _draw_fit(ax, tr.time_s, tr.result, colour)
        for i in (tr.result.start, tr.result.stop - 1):
            ax.axvline(tr.time_s[i] / 3600, linestyle=":", color=colour, alpha=0.5)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_ylabel("Absorbance NADH (bg-subtracted)")
    ax.set_xlabel("Time [hours]")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(frameon=False, fontsize=8)
    return fig


def plot_rates(summary: pd.DataFrame, label_columns: Sequence[str] = ("X", "M", "A", "B", "C")):
    """Bar per config row; the composition (label columns) is written under each bar."""
    columns = [c for c in label_columns if c in summary.columns]
    fig, ax = plt.subplots(figsize=(max(8, len(summary) * 1.2), 6))
    x = np.arange(len(summary))
    ax.bar(x, summary["rate"], color=_palette(len(summary)))
    labels = ["\n".join(f"{row[c]:g}" if isinstance(row[c], (int, float)) else str(row[c]) for c in columns)
              for _, row in summary.iterrows()]
    ax.set_xticks(np.append(-1, x), ["\n".join(columns)] + labels)
    ax.set_xlim(-1.5, len(summary) - 0.5)
    ax.set_ylabel("ATPase rate [1/min]")
    ax.set_title("ATPase rates")
    fig.tight_layout()
    return fig
