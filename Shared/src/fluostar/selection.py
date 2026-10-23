"""Interactive selection: wells on a 96-well plate, and a time window on a trace.

* ``WellSelection``      the click logic (testable, no GUI)
* ``WellPlateSelector``  the notebook widget around it (needs ``%matplotlib widget``,
                         ipywidgets and ipympl)
* ``select_trace_window`` drag start/end lines in a matplotlib window (scripts;
                         needs an interactive backend such as Qt5Agg)

Clicking a well cycles: off -> selected -> base control (light blue) ->
fibril control (dark blue) -> off. There is one base and one fibril control; when
one is taken, that step is skipped.
"""

from __future__ import annotations

import re
import time
from typing import Sequence

import numpy as np
import pandas as pd

COLOURS = {"selected": "royalblue", "base": "lightblue", "fibril": "darkblue", "off": "gray"}


class WellSelection:
    def __init__(self, wells: Sequence[str]):
        self.wells = list(wells)
        self.selected: list[str] = []
        self.base: str | None = None      # e.g. no-fiber well   -> roi column nBase
        self.fibril: str | None = None    # e.g. no-ATP well     -> roi column nFibril

    def state(self, well: str) -> str:
        if well in self.selected:
            return "selected"
        if well == self.base:
            return "base"
        if well == self.fibril:
            return "fibril"
        return "off"

    def click(self, well: str) -> str:
        if well not in self.wells:
            raise KeyError(f"well {well!r} not on this plate")
        current = self.state(well)
        if current == "selected":
            self.selected.remove(well)
            if self.base is None:
                self.base = well
            elif self.fibril is None:
                self.fibril = well
        elif current == "base":
            self.base = None
            if self.fibril is None:
                self.fibril = well
        elif current == "fibril":
            self.fibril = None
        else:
            self.selected.append(well)
        return self.state(well)

    def colours(self) -> list[str]:
        return [COLOURS[self.state(w)] for w in self.wells]

    def roi(self) -> pd.DataFrame:
        """Selected wells in plate order, each with the chosen controls."""
        if self.base is None or self.fibril is None:
            raise ValueError("select a base control (light blue) and a fibril control "
                             "(dark blue) before asking for the roi")
        wells = sorted(self.selected, key=self.wells.index)
        return pd.DataFrame({"well": wells, "nFibril": [self.fibril] * len(wells),
                             "nBase": [self.base] * len(wells)})


def _plate_position(well: str) -> tuple[int, int]:
    """'B10' -> (column 10, row 2)."""
    return int(re.findall(r"\d+", well)[0]), ord(re.findall(r"[A-Z]", well.upper())[0]) - ord("A") + 1


class WellPlateSelector:
    """Notebook widget: click wells on a drawn 96-well plate, see their traces.

    Usage (in a notebook cell with ``%matplotlib widget``)::

        selector = WellPlateSelector(data)
        ...click wells...
        roi = selector.roi()
    """

    def __init__(self, data):
        import ipywidgets as widgets
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        from IPython.display import display

        self.data = data
        self.selection = WellSelection(list(data.layout))
        self.fig = plt.figure(tight_layout=True, figsize=(9.6, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[5.6, 4])
        self.ax_plate = self.fig.add_subplot(gs[0, 0])
        self.ax_trace = self.fig.add_subplot(gs[0, 1])
        self.text = None
        self._draw_plate()
        self.cid = self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        plt.show()

        self.button = widgets.Button(description="Sort Selection")
        self.button.on_click(lambda _: print(self.summary()))
        display(self.button)

    def _draw_plate(self):
        import matplotlib.patches as patches

        ax = self.ax_plate
        xy = [_plate_position(w) for w in self.selection.wells]
        self.scatter = ax.scatter([p[0] for p in xy], [p[1] for p in xy], picker=True)
        ax.set_xticks(range(1, 13), range(1, 13))
        ax.set_yticks(range(1, 9), [chr(ord("A") + i) for i in range(8)])
        ax.set_aspect("equal")
        # plate outline (SBS footprint, units of the 9 mm well pitch) with cut corner
        xl, xr = 1 - 14.38 / 9, 12 + 14.38 / 9
        yt, yb = 1 - (4.5 + 7.01) / 9, 8 + (4.5 + 7.01) / 9
        ax.set_xlim(xl - 1, xr + 1)
        ax.set_ylim(yb + 1, yt - 1)
        for i in range(1, 13):
            for j in range(1, 9):
                ax.add_patch(patches.Circle((i, j), 4 / 9, edgecolor="black", facecolor="none"))
        cut = np.sqrt(2 * 7.85) / 9
        for xs, ys in (([xl + cut, xr], [yt, yt]), ([xl, xl], [yt + cut, yb]),
                       ([xl + cut, xl], [yt, yt + cut]), ([xr, xr], [yt, yb]), ([xl, xr], [yb, yb])):
            ax.plot(xs, ys, color="black", linewidth=1)
        self._yt = yt

    def _on_pick(self, event):
        self.fig.canvas.mpl_disconnect(self.cid)
        well = self.selection.wells[event.ind[0]]
        self.selection.click(well)
        self.scatter.set_facecolor(self.selection.colours())

        if self.text:
            self.text.remove()
        self.text = self.ax_plate.text(6.5, self._yt - 0.5, f"Well: {well}\nContent: {self.data.content(well)}",
                                       ha="center", va="top", fontsize=10,
                                       bbox=dict(facecolor="white", alpha=0.5))
        self.ax_trace.cla()
        for gain in self.data.gains:
            g = self.data.select_gain(gain)
            self.ax_trace.plot(g.time_s, g.well(well), label=f"gain {gain}")
        self.ax_trace.set_xlabel("Time [s]")
        self.ax_trace.legend(frameon=False, fontsize=8)
        self.fig.canvas.draw()
        time.sleep(0.3)
        self.cid = self.fig.canvas.mpl_connect("pick_event", self._on_pick)

    def summary(self) -> str:
        s = self.selection
        return f"Selected wells: {sorted(s.selected, key=s.wells.index)}, base: {s.base}, fibril: {s.fibril}"

    def roi(self) -> pd.DataFrame:
        return self.selection.roi()


def select_trace_window(x, y, start_guess=None, end_guess=None, xlabel="Time [s]",
                        ylabel="Signal", title="") -> tuple[int, int]:
    """Drag two vertical lines (light blue = start, dark blue = end), click Finish.

    Returns ``(start, end)`` indices nearest to the line positions; use them as a
    slice, ``x[start:end]``. With a non-interactive backend the window closes at
    once and the indices of the guesses are returned.
    """
    import matplotlib.pyplot as plt
    import matplotlib.widgets as mwidgets

    x, y = np.asarray(x, float), np.asarray(y, float)
    start_guess = x[0] if start_guess is None else start_guess
    end_guess = x[-1] if end_guess is None else end_guess

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(bottom=0.2)
    ax.plot(x, y, color="gray", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Drag start (light blue) / end (dark blue), then click Finish")
    line_start = ax.axvline(start_guess, color="#7EC8E3", linewidth=2)
    line_end = ax.axvline(end_guess, color="#00008B", linewidth=2)
    state = {"dragging": None}

    def on_press(event):
        if event.inaxes != ax or event.xdata is None:
            return
        d_start = abs(event.xdata - line_start.get_xdata()[0])
        d_end = abs(event.xdata - line_end.get_xdata()[0])
        state["dragging"] = "start" if d_start <= d_end else "end"

    def on_motion(event):
        if state["dragging"] is None or event.inaxes != ax or event.xdata is None:
            return
        if state["dragging"] == "start":
            pos = min(event.xdata, line_end.get_xdata()[0])
            line_start.set_xdata([pos, pos])
        else:
            pos = max(event.xdata, line_start.get_xdata()[0])
            line_end.set_xdata([pos, pos])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", lambda e: state.update(dragging=None))
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    finish = mwidgets.Button(fig.add_axes([0.82, 0.02, 0.12, 0.06]), "Finish")
    finish.on_clicked(lambda e: plt.close(fig))
    plt.show(block=True)  # returns when Finish is clicked or the window is closed

    start = int(np.argmin(np.abs(x - line_start.get_xdata()[0])))
    end = int(np.argmin(np.abs(x - line_end.get_xdata()[0])))
    plt.close(fig)
    return start, end
