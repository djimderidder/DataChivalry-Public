# Shared (`fluostar`)

Helpers for BMG FLUOstar plate-reader data, used by `Modules/ThT_decay` and
`Modules/ATPase_rate`. The run scripts of those modules add `Shared/src` to the
path themselves, so nothing needs to be installed.

| module | what |
|---|---|
| `io` | `load_fluostar(path)` or `load_fluostar(folder, file_name)` returns a `FluostarData` |
| `windows` | `Window`, `last_seconds`, `fit_linear`, `window_mean`: a fit or average over a time window |
| `config` | the per-experiment config spreadsheet: `read_config`, `is_missing`, `raw_file_path`, `clean_well` |
| `selection` | `WellPlateSelector` (click wells, notebook) and `select_trace_window` (drag start/end lines, scripts) |
| `save` | `append_plate_to_csv` (master CSV of analysed wells) and `save_and_show` (end of a run script) |
| `synthetic` | `write_fluostar_csv`: mock data in the real export format |

## FluostarData

```python
data = load_fluostar(r"D:\fluostar\20250509", "plate.csv")
data.gains, data.wells, data.layout["A9"], data.meta["Test name"]

g3 = data.select_gain(3)                 # returns a new object
trimmed = g3.trim(start_s=600, end_s=14 * 3600)   # keeps 600 < t <= 14 h
trimmed.time_s, trimmed.well("A9")
```

`select_gain` and `trim` return a new object, so running a Spyder file or notebook
cell twice can't trim twice, and the original data stays available. Other details:
- Well names are stripped of spaces everywhere: `" A9"` in an export or config is `"A9"`.
- `overflow` values become NaN, and `fit_linear` / `window_mean` skip NaN points.
- `time_s` and `well()` need a single gain; with several gains they ask you to call
  `select_gain` first.

## Windows

Both assays reduce a trace to numbers over a time window (all times in seconds):

| | window |
|---|---|
| ThT initial rate | `Window(end_s=3600)`: absolute time, applied after the equilibration cut |
| ThT endpoint | `last_seconds(t, 3600)` |
| ATPase rate | `Window.from_indices(t, starti, endi)`: Python slice, `endi` excluded |

## Tests

`pip install pytest`, then `pytest` from this folder. The tests use mock exports written
by `synthetic.write_fluostar_csv`, and cover:
- parsing: gains, spaces, overflow, meta data, bad files
- trimming conventions
- windows and fits (compared with `scipy.stats.linregress`)
- the click logic of the plate selector
