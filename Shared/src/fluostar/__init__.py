"""fluostar: shared helpers for BMG FLUOstar plate-reader exports.

Modules
-------
io         load_fluostar() -> FluostarData (select_gain, trim, well, time_s, layout)
windows    Window, last_seconds, fit_linear, window_mean: "fit or average over a time window"
config     reading the per-experiment config spreadsheet (is_missing, raw_file_path, ...)
selection  interactive well-plate selector (notebook) and trace-window picker (script)
save       append selected wells to a master CSV
synthetic  write mock FLUOstar CSVs in the real export format (tests and examples)
"""

from .io import FluostarData, load_fluostar
from .windows import LinearFit, Window, fit_linear, last_seconds, window_mean

__all__ = ["FluostarData", "load_fluostar", "Window", "LinearFit", "fit_linear",
           "last_seconds", "window_mean"]
