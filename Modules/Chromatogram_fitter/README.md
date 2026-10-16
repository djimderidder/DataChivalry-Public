# AktaNgcAnalyser
Load, process and fit chromatogram traces (UV, conductivity, %B gradient) exported from ÄKTA and Bio-Rad NGC chromatography systems.

## main_AktaNgcAnalyser.py
`main_AktaNgcAnalyser.py` is the main script. It loads the config spreadsheet, loads and parses the raw exported traces, runs them through the preprocessing pipeline (dead-volume correction, baseline correction, gradient-range extraction, elutant/blank subtraction, volume reset) and fits a user-defined peak model to each run. The model in the script is an example (a Gaussian peak); replace it with your own.

## functions/LoadingAktaNgcAnalyser.py
Reads the config spreadsheet and the raw instrument export files into pandas DataFrames.
- `loadConfig` reads the `configFiles` and `configParameters` sheets and returns only the rows marked to run.
- `loadAktaNgc` parses each raw export (ÄKTA `SystemA`/`SystemB` CSV format, or Bio-Rad NGC CSV format) into a common set of columns (`Volume [mL]`, `UV [mAU]`, `B [%B]`, `Cond [mS/cm]`, etc.).
- `loadBlank` loads each run's assigned blank trace the same way.
- `checkBlankGradientProfile` warns (without stopping the pipeline) if a run's raw %B gradient doesn't match its assigned blank's.
- `saveConfig` writes fitted parameters and reduced chi-square back into the config spreadsheet.

## functions/ProcessingAktaNgcAnalyser.py
The preprocessing steps, applied to both sample and blank runs:
- `correctDeadvolume` shifts %B and elutant concentration (whichever are present) by the system's dead volume.
- `correctBaseline` zeroes the UV signal using the mean of each run's own tail.
- `getGradientRange` crops each run down to the gradient elution range, optionally extended by a number of column volumes (`ExtraCV`).
- `correctElutant` subtracts the elutant/blank UV contribution from each sample run: a flat or linear (drift-correcting) baseline for ÄKTA runs, a scaled blank for NGC runs with a blank, and a %B-shaped profile for NGC runs without one.
- `restartVolume` zeroes each run's volume axis to its own start.

Every step takes `Plot=True/False`; when `True` it shows a before/after plot of the affected signal(s) for each run. `correctElutant` additionally overlays the subtracted baseline and scaled blank.

And the fitting step:
- `fitData` fits any model function `Func(x, p1, p2, ...)` to each run with lmfit. Initial guesses come from an `Estimator(mu, var, amp)` you supply, which maps the profile's centroid, variance and area onto your model's parameters. Optional weighting (`Sigma`), including a per-run noise estimate (`Sigma='auto'`), optional parameter bounds, and optional derived quantities. With `Plot=True` each fit is shown with a side panel of fit metrics; otherwise they are printed.

## functions/PlottingAktaNgcAnalyser.py
Plotting helpers: a 3-axis overview plot of a single run (`linePlot3axis`), the before/after overlay used by the processing steps (`linePlotProcess`), the fit plot with metrics panel (`linePlotFit`), and fraction-annotated plots for ÄKTA runs (`linePlotFractions`, `addFractions`).

### Setup
Before running the script:
1. `pip install -r requirements.txt`
2. Set up a config spreadsheet (`config_MAIN.xlsx` by default) with two sheets:
   - **configFiles**: one row per run, with at least `Filename`, `Filefolder`, `System` (`SystemA`, `SystemB`, or anything else, which is treated as an NGC run), `Max Elutant [mM]` (may be left empty), `BlankFilename`, `BlankFoldername` (may be left empty for runs without a blank), and `RunBool` (set to `1` for rows you want to include). Optionally `CV [mL]` (column volume), needed for `getGradientRange(ExtraCV=...)`.
   - **configParameters**: `name` / `value` / `units` rows defining shared parameters, in particular `deadVolume`, `deadVolumeSystemA`, and `deadVolumeSystemB`.
3. Place the spreadsheet next to `main_AktaNgcAnalyser.py`, or update `configName` at the top of the script.

### Workflow
Running `main_AktaNgcAnalyser.py` will:
1. Load the config and the raw traces for every run marked to run.
2. Load each run's assigned blank and warn if its gradient profile doesn't match.
3. Apply dead-volume correction, baseline correction, gradient-range extraction, elutant/blank subtraction and a volume reset, to both the sample runs and their blanks.
4. Fit the model defined in the script to each run and show the fit with its metrics.
5. Optionally (commented out by default) save the fit results back to the config spreadsheet.

### Notes
- Set `Plot=True` on any processing step to visually check that step's effect before trusting the output.
- The pipeline expects each run's `System` value to be `SystemA`, `SystemB`, or something else (interpreted as an NGC run) — update these to match your own instrument naming.
- Each run's `Name` (used in plot titles and messages) is the first 8 characters of its filename plus its row index.
