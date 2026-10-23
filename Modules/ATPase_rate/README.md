# ATPase_rate

ATP hydrolysis rates from an enzyme-coupled ATPase assay on a FLUOstar plate reader:
every ATP hydrolysed oxidises one NADH, so the rate follows from the decrease of the NADH
absorbance at 340 nm:

```
rate = − (dA₃₄₀/dt) / (ε · l · [E])        ATP per enzyme per minute
```

The slope is in absorbance per minute, ε = 6220 M⁻¹cm⁻¹ (NADH), l the path length
(1 cm: FLUOstar exports are path-length corrected) and [E] the ATPase concentration.

## Run

1. Install the requirements: `pip install numpy scipy pandas matplotlib openpyxl palettable`.
2. Fill in a config spreadsheet (see `input/ATPase config template.xlsx`), then run one of:
   * **`atpaseVisualize.py`**: every row's trace with its fit, plus a bar plot of the rates.
   * **`atpasePicker.py`**: find or check the fitting windows. For each row it shows the
     automatic window and asks `y/n`. With `n`, a window opens where you drag start (light
     blue) and end (dark blue) and click *Finish*. With `write_back = True` the chosen
     Starti/Endi/Rate go to a **copy** of the config in `output/`; the original is never
     overwritten.
3. Set `raw_data_root` and `config_file` in the USER SETTINGS and run the file (Spyder:
   *Run file*, F5). Leave `raw_data_root = None` to run on a simulated experiment first.
   Afterwards `summary`, `traces` (visualize) or `picks` (picker) are in the Variable Explorer.

Dragging needs an interactive matplotlib backend. In Spyder that is *Preferences →
IPython console → Graphics → Backend: Qt5*; the *Inline* backend can't drag.

From a terminal the same settings are used; flags override them without editing the file:

```bash
python atpaseVisualize.py --root D:/fluostar --config input/my_config.xlsx --rows 5,6 --save output --no-show
python atpasePicker.py --root D:/fluostar --config input/my_config.xlsx --rows 13,3 --write-back
python atpasePicker.py --accept-all --write-back      # no questions: keep every automatic window
```

### Config spreadsheet

| column | meaning |
|---|---|
| `folder`, `filename` | raw export, relative to `raw_data_root` (Windows `\` also works on macOS/Linux) |
| `Well`, `Control` | sample well, and the control well subtracted from it (`-` = none) |
| `Starti`, `Endi` | fitting window as point indices, `signal[Starti:Endi]`; `-` = automatic |
| `A` | ATPase concentration in µM (`AtpaseSettings(enzyme_column=...)`; old spreadsheets: `H`) |
| `exp`, `date`, `X`, `M`, `B`, `C`, ... | description: `X` fibrils, `M` monomers, others (µM); shown under the rate bars via `label_columns` |
| `Rate` | written by the picker (`write_back`) |

### Fitting window

* **Automatic start:** the first point.
* **Automatic end:** the first point (after the start) where the signal reaches 95 % of
  the mean of the last 5 points, i.e. where the NADH is used up. Set this with
  `auto_threshold` and `auto_n_last`.
* **From the config:** each bound is used when it's filled in, and each bound is chosen
  separately. `Starti = 150, Endi = '-'` fits from point 150 to the automatic end.

The old `atpaseVisualize.py` switched *both* bounds to automatic when either was
missing, so a filled-in `Starti` was ignored whenever `Endi` was empty.

## Tests

`pip install pytest`, then `pytest`. The tests use simulated plates: NADH absorbance
falls linearly at a known rate until it's used up, with a slightly drifting control well,
noise, and one well with a slow start that its config `Starti` skips.

They check that the code recovers the simulated rates and reproduces the formulas of the
old scripts. They also run both run scripts, including the `y`/`n` branch of the picker.
Tests on measured data run only when `tests/data/real/` (git-ignored) holds a raw export
plus a `config.xlsx` with your previously found `Rate`; run only those with
`pytest -m real_data`.

## Limits

* The rate is a linear fit over the chosen window. There is no correction for a lag
  phase other than skipping it with `Starti`.
* The automatic end assumes the trace ends flat (NADH used up). Without depletion it
  stops near the end of the trace; check those runs with the picker.
