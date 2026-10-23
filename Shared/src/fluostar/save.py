"""Keep a master CSV of analysed wells, and save/show the output of a run script."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .io import FluostarData


def append_plate_to_csv(data: FluostarData, roi: pd.DataFrame, master_file: str | Path) -> Path:
    """Append the wells in ``roi`` (columns well, nFibril, nBase) to ``master_file``."""
    master_file = Path(master_file)
    rows = pd.DataFrame({
        "fileFolder": str(data.file_folder or ""),
        "fileName": data.file_name or "",
        "well": roi["well"].to_list(),
        "condition": [data.content(w) for w in roi["well"]],
        "nFibril": roi["nFibril"].to_list(),
        "nBase": roi["nBase"].to_list(),
        "date": datetime.now().strftime("%Y-%m-%d"),
    })
    if master_file.exists():
        rows = pd.concat([pd.read_csv(master_file), rows], ignore_index=True)
    master_file.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(master_file, index=False)
    print(f"Appended {len(roi)} well(s) to {master_file}")
    return master_file


def save_and_show(figures: dict, tables: dict, save_to: str | Path | None = None,
                  show: bool = True) -> None:
    """End of a run script: print the tables, optionally write figures (PNG) and
    tables (CSV) to ``save_to``, then show the figures (or close them)."""
    import matplotlib.pyplot as plt

    for name, table in tables.items():
        print(f"\n=== {name} ===\n{table.to_string(index=False)}")
    if save_to is not None:
        save_to = Path(save_to)
        save_to.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(save_to / f"{name}.csv", index=False)
        for name, fig in figures.items():
            fig.savefig(save_to / f"{name}.png", dpi=150, bbox_inches="tight")
        print(f"\nWrote {len(tables)} table(s) and {len(figures)} figure(s) to {save_to}")
    if show:
        plt.show()
    else:
        for fig in figures.values():
            plt.close(fig)
