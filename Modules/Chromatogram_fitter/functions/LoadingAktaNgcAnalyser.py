import numpy as np
import pandas as pd
import os

class Parameter:
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

    def __float__(self):
        # float() so integer values read from Excel (numpy int64) also work
        return float(self.value)

def loadConfig(Name):
    config = pd.read_excel(Name, sheet_name='configFiles')
    dfPara = pd.read_excel(Name, sheet_name='configParameters')

    para = {row['name']: Parameter(row['value'], row['units']) for _, row in dfPara.iterrows()}

    config = (config.loc[config['RunBool'] == 1]).reset_index(drop=True)  # only continue with files for which RunBool equals 1

    return config, para


def saveConfig(Config, FitPara, ChiSquares, Name, Ask=True):
    """Write each run's fitted parameters and reduced chi-square back into
    the 'configFiles' sheet of the config spreadsheet, matched on 'Filename'.

    Config     : the config the fit was run on (one row per run, same order
                 as FitPara).
    FitPara    : fitted parameters per run, e.g. the first output of fitData.
    ChiSquares : (chi2, reduced chi2) per run, e.g. the third output of fitData.
    Name       : path to the config spreadsheet. Excel must be closed.
    Ask        : if True, ask for confirmation on the console before writing.

    Only the 'configFiles' and 'configParameters' sheets are written back;
    any other sheets or cell formatting in the workbook are lost.

    Returns a copy of Config with 'FitPara' and 'Reduced Chi-square' added.
    """
    out = Config.copy()
    out['FitPara'] = [str([float(f'{v:.6g}') for v in p]) for p in FitPara]
    out['Reduced Chi-square'] = [c[1] for c in ChiSquares]

    if Ask:
        answer = input(f"Save the fit results to '{os.path.basename(Name)}'? "
                       "Excel must be closed (y/n): ").strip().lower()
        if answer not in ('y', 'yes'):
            return out

    sheetFiles = pd.read_excel(Name, sheet_name='configFiles')
    sheetPara = pd.read_excel(Name, sheet_name='configParameters')
    newCols = ['FitPara', 'Reduced Chi-square']
    for col in newCols:
        if col not in sheetFiles.columns:
            sheetFiles[col] = None
        sheetFiles[col] = sheetFiles[col].astype(object)
    colOrder = list(sheetFiles.columns)

    updates = out.drop_duplicates('Filename', keep='last').set_index('Filename')[newCols]
    sheetFiles = sheetFiles.set_index('Filename')
    sheetFiles.update(updates)
    sheetFiles = sheetFiles.reset_index()[colOrder]

    with pd.ExcelWriter(Name, engine='openpyxl') as writer:
        sheetFiles.to_excel(writer, sheet_name='configFiles', index=False)
        sheetPara.to_excel(writer, sheet_name='configParameters', index=False)
    print(f"Fit results saved to '{os.path.basename(Name)}'.")
    return out


def loadAktaNgc(Config, Dir):
    data_list = []  # Store all dataframes

    # Pre-fetch Config values into numpy arrays for faster access
    filenames = Config["Filename"].values
    folder = Config["Filefolder"].values
    systems = Config["System"].values
    max_elutants = Config["Max Elutant [mM]"].values

    for i in range(len(filenames)):
        file_path = os.path.join(Dir, folder[i], filenames[i])
        name = filenames[i][:8] + '_' + str(i)
        system = systems[i]
        max_elutant = max_elutants[i]

        # Handle AKTA systems
        if system in {'SystemA', 'SystemB'}:
            # Attempt to read with both separators only once
            data = None
            for sep in [';', ',']:
                try:
                    data = pd.read_csv(file_path, header=1, sep=sep)
                    if data.shape[1] > 1:
                        break  # Valid read
                except Exception:
                    continue

            if data is None:
                raise ValueError(
                    f"loadAktaNgc: unable to read '{file_path}' with ';' or ',' as separator. "
                    "Check the file exists and is an unmodified instrument export, or set "
                    "RunBool to 0 for this row in the config."
                )

            try:
                # Column extraction
                v_col = data.columns[(data.iloc[0] == 'ml')][0]
                u_col = data.columns[(data.iloc[0] == 'mAU')][0]
                v = data[v_col][1:].astype(float)
                u = data[u_col][1:].astype(float)

                datai = pd.DataFrame({'Volume [mL]': v, 'UV [mAU]': u})

                if 'Gradient Concentration' in data.columns:
                    b_col = data.columns[(data.iloc[0] == '%B')][0]
                    b = data[b_col][1:].astype(float)
                    datai['B [%B]'] = b
                    if not np.isnan(max_elutant):
                        datai['[Elutant] [mM]'] = max_elutant * b / 100

                if 'Conductivity' in data.columns:
                    c_col = data.columns[(data.iloc[0] == 'mS/cm')][0]
                    datai['Cond [mS/cm]'] = data[c_col][1:].astype(float)

                if 'Fraction' in data.columns:
                    f_col = data.columns[(data.iloc[0] == 'Fraction')][0]
                    startF = data['Fraction'][1:].dropna().astype(float)
                    nameF = data[f_col].dropna()
                    datai['Fraction'] = 'T0'
                    for j in range(1, len(startF)):
                        datai.loc[datai['Volume [mL]'] >= startF[j], 'Fraction'] = nameF.iloc[j]

                datai['Name'] = name
                data_list.append(datai)

            except Exception as e:
                raise RuntimeError(
                    f"loadAktaNgc: error parsing '{file_path}' (System '{system}'): {e}"
                ) from e

        else:  # Assume NGC data
            try:
                df = pd.read_csv(file_path, header=1, sep=",")
                # NGC records one shared volume axis per run; take it from the
                # same channel as the UV signal, falling back to UV1 if needed.
                volName = 'UV2 (280 nm)_column_volume'
                if volName not in df.columns:
                    volName = 'UV1 (215 nm)_column_volume'
                volume_col = df[volName]
                uv_col = df['UV2 (280 nm)_mAU']
                cond_col = df['Conductivity_mS/cm']
                percent_b = df['%B_column_volume']
                b_percent = df['%B_%']

                # Precompute B values using vectorized operation instead of apply
                b_lookup = pd.Series(b_percent.values, index=percent_b).sort_index()
                b_interp = np.interp(volume_col, b_lookup.index, b_lookup.values)

                datai = pd.DataFrame({
                    'Volume [mL]': volume_col,
                    'UV [mAU]': uv_col,
                    'Cond [mS/cm]': cond_col,  # same column name as the AKTA branch
                    'B [%B]': b_interp,
                    'Name': name
                })

                if not np.isnan(max_elutant):
                    datai['[Elutant] [mM]'] = max_elutant * b_interp / 100

                data_list.append(datai)

            except Exception as e:
                raise RuntimeError(
                    f"loadAktaNgc: error parsing NGC file '{file_path}' (System '{system}' "
                    f"is treated as NGC): {e}"
                ) from e

    return data_list


def loadBlank(Config, Dir):
    blankRows = []
    for u in Config['BlankFilename'].unique():
        if pd.notna(u):
            first_row = Config.loc[Config['BlankFilename'] == u].iloc[0].copy()
            first_row['Filename'] = first_row['BlankFilename']
            first_row['Filefolder'] = first_row['BlankFoldername']
            blankRows.append(first_row)

    if not blankRows:
        print('No blank runs assigned in the config - continuing without blanks.')
        return [], Config, pd.DataFrame(columns=Config.columns)

    blankConfig = pd.DataFrame(blankRows).reset_index(drop=True)
    blank = loadAktaNgc(blankConfig, Dir)

    # Create a mapping from 'Filename' to the index in 'blankConfig'
    filename_to_index = blankConfig.reset_index().set_index('Filename')['index']

    # Map the filenames to indices, preserving NaNs
    Config['BlankFilename'] = Config['BlankFilename'].map(filename_to_index)
    Config['BlankFilename'] = Config['BlankFilename'].apply(
        lambda x: int(x) if pd.notna(x) else x
    )

    # blankConfig is returned too: it carries the 'System' (and other config)
    # for each blank run, which the processing steps need in order to apply
    # exactly the same corrections (dead volume, etc.) to the blank as to the
    # sample data.
    return blank, Config, blankConfig


def checkBlankGradientProfile(Data, Blank, Config, ToleranceB=2.0):
    """Sanity-check that each run's raw %B gradient matches its assigned blank's.

    For every run in Data that has a blank assigned (Config['BlankFilename']
    not NaN), this compares 'B [%B]' vs 'Volume [mL]' against the blank over
    their overlapping volume range. If they differ by more than ToleranceB
    (%B) anywhere in that range, a warning is printed naming the run and its
    blank - this never raises or stops the pipeline, it's just a heads-up
    that the wrong blank might be assigned, or that the two runs used a
    different gradient program.

    Call this right after loadBlank(), before any further processing
    (correctDeadvolume, getGradientRange, ...), since it's meant to check the
    raw, as-recorded profiles.

    Returns a list of dicts (one per mismatch found), empty if everything
    matches within tolerance.
    """
    mismatches = []
    for i in range(len(Data)):
        blankIdx = Config['BlankFilename'].iloc[i]
        if pd.isna(blankIdx):
            continue  # no blank assigned to this run, nothing to check

        datai = Data[i]
        blanki = Blank[int(blankIdx)]
        name = datai['Name'].iloc[0] if 'Name' in datai.columns else f'index {i}'
        blankName = blanki['Name'].iloc[0] if 'Name' in blanki.columns else f'blank {int(blankIdx)}'

        if 'B [%B]' not in datai.columns or 'B [%B]' not in blanki.columns:
            print(f"Warning: cannot check %B profile for '{name}' vs '{blankName}' "
                  "- 'B [%B]' column is missing from one of them.")
            continue

        vmin = max(datai['Volume [mL]'].min(), blanki['Volume [mL]'].min())
        vmax = min(datai['Volume [mL]'].max(), blanki['Volume [mL]'].max())
        if vmin >= vmax:
            print(f"Warning: '{name}' and its blank '{blankName}' have no overlapping "
                  "volume range - cannot check %B profiles.")
            continue

        mask = (datai['Volume [mL]'] >= vmin) & (datai['Volume [mL]'] <= vmax)
        volCommon = datai['Volume [mL]'][mask]
        bData = datai['B [%B]'][mask]
        bBlankInterp = np.interp(volCommon, blanki['Volume [mL]'], blanki['B [%B]'])

        diff = (bData - bBlankInterp).abs()
        maxDiff = diff.max()
        if maxDiff > ToleranceB:
            volAtMax = volCommon.loc[diff.idxmax()]
            print(f"Warning: %B profile mismatch between '{name}' and its blank "
                  f"'{blankName}' - up to {maxDiff:.1f} %B difference "
                  f"(at ~{volAtMax:.2f} mL). Double-check the correct blank was assigned.")
            mismatches.append({
                'Sample': name,
                'Blank': blankName,
                'MaxDiffB': maxDiff,
                'VolumeAtMaxDiff': volAtMax,
            })

    return mismatches


def loadFractions(Filename, Folder):
    out = pd.read_excel(os.path.join(Folder, Filename), header=1)
    return out