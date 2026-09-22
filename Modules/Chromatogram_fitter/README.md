# AktaNgcAnalyser
Load and process chromatogram traces (UV, conductivity, %B gradient) exported from ÄKTA and Bio-Rad NGC chromatography systems, as a preprocessing pipeline ahead of elution-peak fitting.

## main_AktaNgcAnalyser.py
`main_AktaNgcAnalyser.py` is the main script. It loads the config spreadsheet, loads and parses the raw exported traces, and runs them through the preprocessing pipeline (dead-volume correction, baseline correction, gradient-range extraction, elutant/blank subtraction).

## functionsAktaNgcAnalyser/LoadingAktaNgcAnalyser.py
Reads the config spreadsheet and the raw instrument export files into pandas DataFrames.
- `loadConfig` reads the `configFiles` and `configParameters` sheets and returns only the rows marked to run.
- `loadAktaNgc` parses each raw export (ÄKTA `SystemA`/`SystemB` CSV format, or Bio-Rad NGC CSV format) into a common set of columns (`Volume [mL]`, `UV [mAU]`, `B [%B]`, etc.).
- `loadBlank` loads each run's assigned blank trace the same way.
- `checkBlankGradientProfile` is a sanity check that warns (without stopping the pipeline) if a run's raw %B gradient doesn't match its assigned blank's.

## functionsAktaNgcAnalyser/ProcessingAktaNgcAnalyser.py
The preprocessing steps, applied to both sample and blank runs:
- `correctDeadvolume` shifts %B and elutant concentration by the system's dead volume.
- `getGradientRange` crops each run down to the gradient elution range.
- `correctBaseline` zeroes the UV signal using the mean of each run's own tail.
- `correctElutant` subtracts the elutant/blank UV contribution from each sample run.
- `restartVolume` zeroes each run's volume axis to its own start (optional, not run by default).

Every step takes `Plot=True/False`; when `True` it shows a before/after plot of the affected signal(s) for each run, using `linePlotProcess` from the plotting module.

Peak fitting (fitting an elution profile to a model function) is not yet included in this release.

## functionsAktaNgcAnalyser/PlottingAktaNgcAnalyser.py
Plotting helpers used throughout the pipeline: a 3-axis overview plot of a single run (`linePlot3axis`), the before/after overlay used by the processing steps (`linePlotProcess`), and fraction-annotated plots for ÄKTA runs (`linePlotFractions`, `addFractions`).

### Setup
Before running the script:
1. `pip install -r requirements.txt`
2. Set up a config spreadsheet (`config_MAIN.xlsx` by default) with two sheets:
   - **configFiles**: one row per run, with at least `Filename`, `Filefolder`, `System` (`SystemA`, `SystemB`, or anything else, which is treated as an NGC run), `Max Elutant [mM]`, `BlankFilename`, `BlankFoldername`, and `RunBool` (set to `1` for rows you want to include).
   - **configParameters**: `name` / `value` / `units` rows defining shared parameters, in particular `deadVolume`, `deadVolumeSystemA`, and `deadVolumeSystemB`.
3. Place the spreadsheet next to `main_AktaNgcAnalyser.py`, or update `configName` at the top of the script.

### Workflow
Running `main_AktaNgcAnalyser.py` will:
1. Load the config and the raw traces for every run marked to run.
2. Load each run's assigned blank and warn if its gradient profile doesn't match.
3. Apply dead-volume correction, baseline correction, gradient-range extraction, and elutant/blank subtraction, to both the sample runs and their blanks.
4. Leave the result in `elution`, ready for plotting or (in a future release) peak fitting.

### Notes
- Set `Plot=True` on any processing step to visually check that step's effect before trusting the output.
- The pipeline currently expects each sample's `System` value in the config to be `SystemA`, `SystemB`, or something else (interpreted as an NGC run) — update these to match your own instrument naming.
