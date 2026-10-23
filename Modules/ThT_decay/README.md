# ThT_decay

Disaggregation kinetics from FLUOstar Thioflavin T (ThT) plate-reader measurements.
Each sample well is normalised against a fibril control (fibrils without ATP) and a base
control (no fibrils). From the normalised trace come the **initial rate** (%/h, linear fit
over the first hour) and the **endpoint** (mean over the last hour, %).

## Run

1. Install the requirements: `pip install numpy scipy pandas matplotlib openpyxl palettable`
   (`ipywidgets ipympl` for the notebook).
2. Pick a way in:
   * **`runTht.py`**: one plate. Fill in `fileFolder`, `fileName`, the `wells` and the two
     controls in the USER SETTINGS, then run it (Spyder: *Run file*, F5).
   * **`runThtConfig.py`**: batch over a config spreadsheet (see
     `input/ThT config template.xlsx`). Set `raw_data_root` and `config_file`, then run it.
   * **`FLUOstar Fluorescence Assay Disaggregation.ipynb`**: the same steps as `runTht.py`,
     but you click the wells and controls on a drawn plate.
3. Leave `fileName = None` / `raw_data_root = None` to run on a simulated plate first.
   Afterwards `summary`, `norm`, `kinetics`, ... are in the Variable Explorer. Set
   `save_results = True` to write the summary (CSV) and figures (PNG) to `output/`.

From a terminal the same settings are used; flags override them without editing the file:

```bash
python runTht.py --folder D:/fluostar/20250509 --file "plate.csv" --wells A9,B9 --diagnostics A9
python runThtConfig.py --root D:/fluostar --config input/my_config.xlsx --rows 0,1,2 --save output --no-show
```

### Config spreadsheet

| column | meaning |
|---|---|
| `folder`, `filename` | raw export, relative to `raw_data_root` (Windows `\` also works on macOS/Linux) |
| `Well` | sample well |
| `Control F` | fibril control: fibrils without ATP (no disaggregation) |
| `Control T` | base control: no fibrils (background ThT) |
| `exp`, `date`, `X`, `M`, `A`, `B`, `C`, ... | description, copied into the summary: `X` fibrils, `M` monomers, `A`, `B`, `C`, ... other components (µM) |

### Settings (`ThtSettings`)

| setting | default | meaning |
|---|---|---|
| `gain` | 3 | which gain of a multi-gain export |
| `equilibration_s` | 600 | drop t ≤ this (noisy start) |
| `max_time_s` | None | drop t > this (`runThtConfig.py` uses 14 h) |
| `initial_window_s` | 3600 | initial rate: fit over t ≤ this, in absolute time |
| `final_window_s` | 3600 | endpoint: mean over the last part |
| `smoothing_window`, `smoothing_poly` | 51, 3 | Savitzky-Golay smoothing |
| `smooth_initial`, `smooth_final` | False | use the smoothed trace for the rate / endpoint |
| `normalize_points` | 1 | normalise by the mean of the first *n* points (1 = first point, as before) |

The initial-rate window is on absolute time, as intended. Because the equilibration cut
removes the first 10 minutes, the "first hour" rate is fitted over 600–3600 s (50 min).

The old `Mode` argument maps onto the two smoothing switches. `ThtSettings.from_mode("both")` gives:

| old `Mode` | `smooth_initial` | `smooth_final` |
|---|---|---|
| `'off'` | False | False |
| `'initial'` | True | False |
| `'final'` | False | True |
| `'both'` | True | True |

### Normalisation

```
n(t) = [(S − B)(t) / (S − B)₀] / [(F − B)(t) / (F − B)₀]
```

S is the sample, F the fibril control, B the base control, and ₀ the start value. The
normalised trace therefore starts at 1, and drift of the fibril control (bleaching) is
divided out.

## Tests

`pip install pytest`, then `pytest`. The tests use simulated plates:
- **Sample wells** decay as a power law, `n(t) = (1 + t/t₀)^−0.6`.
- **Amplitude** is random per well, because raw ThT intensity varies a lot.
- **Controls:** the fibril control is close to the experimental amplitude with a tiny
  negative slope; the base control is much lower. Noise is added, and there are three gains.

They check that the code recovers the simulated kinetics and reproduces the formulas of
the old scripts. They also run the two run scripts and the notebook. Tests on measured
data run only when `tests/data/real/` (git-ignored) holds raw exports plus a `config.xlsx`
with your previously found `end` / `end_time`; run only those with `pytest -m real_data`.

## Limits

* The normalisation needs both controls for every well. There is no fallback without a fibril control.
* Initial rate and endpoint are model-free (a linear fit and a mean), not a fit of a
  kinetic model.
* The clickable plate needs a notebook with `%matplotlib widget`.
