"""Write mock data as FLUOstar CSV exports, so mock data goes through the real loader."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def write_fluostar_csv(
    path: str | Path,
    time_s: Sequence[float],
    wells: Mapping[str, tuple[str, np.ndarray]],
    *,
    gains: Sequence[int] | None = (1,),
    measurement: str = "Fluorescence (FI)",
    channel: str = "448-10/482-10",
    decimals: int = 0,
    overflow: float | None = None,
    test_name: str = "mock",
) -> Path:
    """Write ``wells = {well: (content, values)}`` in the FLUOstar export layout.

    ``values`` has shape ``(len(gains), len(time_s))`` or ``(len(time_s),)``.
    ``gains=None`` writes a single channel without gain number ("Raw Data (340)").
    Values above ``overflow`` are written as ``overflow``, as the reader does.
    """
    time_s = np.asarray(time_s, float)
    gain_list = [None] if gains is None else list(gains)
    headers = [f" Raw Data ({channel})" if g is None else f" Raw Data ({channel} {g})"
               for g in gain_list for _ in time_s]
    width = 2 + len(headers)

    def fmt(v: float) -> str:
        if not np.isfinite(v) or (overflow is not None and v > overflow):
            return "overflow"
        return f"{v:.{decimals}f}"

    lines = [
        ["User: mock user", "Path: -", "Test run no.: 0"],
        [f"Test name: {test_name}", "Date: 01/01/2025", "Time: 12:00:00"],
        [],
        ["ID1: simulated data", "ID2: -", "ID3: -"],
        [measurement],
        [],
        ["Well", "Content", *headers],
        ["", "Time [s]", *[f"{t:g}" for _ in gain_list for t in time_s]],
    ]
    for well, (content, values) in wells.items():
        values = np.atleast_2d(np.asarray(values, float))
        if values.shape != (len(gain_list), time_s.size):
            raise ValueError(f"well {well}: values shape {values.shape}, expected "
                             f"({len(gain_list)}, {time_s.size})")
        lines.append([f"{well:>3}", content, *[fmt(v) for v in values.ravel()]])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="latin-1", newline="") as fh:
        for line in lines:
            cells = [c if ("," not in c and '"' not in c) else '"' + c.replace('"', '""') + '"'
                     for c in line]
            fh.write(",".join(cells + [""] * (width - len(cells))) + "\r\n")
    return path
