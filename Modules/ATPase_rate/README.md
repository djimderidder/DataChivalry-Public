# ATPase_rate
Analyse ATP hydrolysis by HSPA8 using an enzyme-coupled ATPase assay in which ATP consumption is stoichiometrically linked to NADH oxidation.

## atpaseVisualize.py
`atpaseVisualize.py` main code to read ATPase data from config and display ATPase rates once config is set up.
## atpasePicker.py: 
`atpasePicker.py` identifies fitting windows (`Starti` and `Endi`) for the ATPase rate calculations. Since after NADH is depleted you will not measure ATPase rate anymore. The script loads raw FLUOstar traces, proposes an initial fitting region automatically, and allows the user to manually adjust the fitting window when needed.
 
### Setup
Before running the script, update the **User Settings** section:
 
- `raw_data_root`: Path to the directory containing the raw FLUOstar export files.
- `input_path`: Path to the folder containing the config spreadsheet.
- `config_file`: Name of the config spreadsheet.
- `rows_to_run`: List of row indices to process. Set to `None` to process all rows that do not yet have a `Starti` or `Endi` value.
 
### Workflow
For each selected row:
 
1. The raw trace is loaded and background-subtracted using the specified control well.
2. An initial fitting window is proposed automatically.
3. The trace, fit, and calculated rate are displayed in a plot.
4. If the automatically selected window is acceptable, the displayed values can be used directly.
5. If the fit is not satisfactory (type 'n'), an interactive window opens where the start (light blue) and end (dark blue) boundaries can be dragged to a more suitable position.
6. After clicking **Finish**, the trace is refitted using the manually selected window and the updated rate is displayed for 3 second after which the next plot will show.
 
### Notes
- The script is intended as a fitting aid and does not automatically determine the optimal fitting region in all cases.
- Always visually inspect the proposed fit before accepting it.
- The script currently **does not automatically update the configuration spreadsheet**. Selected `Starti` and `Endi` values must be copied manually into the config file.
- An interactive Matplotlib backend (e.g. `Qt5Agg`) is required for the manual picker to function correctly.
- Running the script from Spyder is recommended with the graphics backend set to **Qt5** rather than **Inline**.