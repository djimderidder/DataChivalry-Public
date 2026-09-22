import numpy as np
from scipy.interpolate import interp1d

from PlottingAktaNgcAnalyser import linePlotProcess


# ---------------------------------------------------------------------------
# Shared helper: show a "before vs after" plot for every sample in a list,
# for whichever of `columns` are actually present in the data. Used by every
# processing step below when Plot=True, so we only had to write the plotting
# logic once.
# ---------------------------------------------------------------------------
def _plotStepChange(before, after, columns, stepName):
    for col in columns:
        if col in before.columns and col in after.columns:
            title = f"{after['Name'].iloc[0]} \u2014 {stepName}"
            linePlotProcess(after, before, col, title=title)


# ---------------------------------------------------------------------------
# correctDeadvolume
# ---------------------------------------------------------------------------
def _shiftDeadvolume(dataset, cfg, Para):
    """Apply the dead-volume shift to every dataframe in `dataset`, using the
    system type listed in `cfg` (same length/order as `dataset`)."""
    out = []
    deadV = None
    for i in range(len(dataset)):
        if cfg['System'].iloc[i] == 'SystemA':
            deadV = float(Para['deadVolumeSystemA'])
        elif cfg['System'].iloc[i] == 'SystemB':
            deadV = float(Para['deadVolumeSystemB'])
        else:
            deadV = float(Para['deadVolume'])
            print(f'Assuming data of index {i} is a NGC run')

        datai = dataset[i].copy()
        vol_original = datai['Volume [mL]']
        vol_shifted = vol_original + deadV

        # Create interpolators for the signals you want to shift
        interp_B = interp1d(vol_shifted, datai['B [%B]'], bounds_error=False, fill_value='extrapolate')
        interp_Elutant = interp1d(vol_shifted, datai['[Elutant] [mM]'], bounds_error=False, fill_value='extrapolate')

        # Evaluate interpolated values at original volume points
        datai['B [%B]'] = interp_B(vol_original)
        datai['[Elutant] [mM]'] = interp_Elutant(vol_original)

        out.append(datai)
    return out, deadV


def correctDeadvolume(Data, Config, Para, LogShift=None, Blank=None, BlankConfig=None, Plot=False):
    """Shift %B and elutant concentration by the system dead volume.

    Data / Config : the sample runs and their config, as before.
    Blank         : optionally, the list of blank runs (e.g. from loadBlank).
                    When given, exactly the same correction is applied to
                    them, so data and blank stay on a consistent volume
                    basis for later steps (e.g. correctElutant).
    BlankConfig   : the config belonging to Blank (e.g. from loadBlank),
                    used to look up each blank's 'System'. Defaults to
                    Config if not given.
    Plot          : if True, show a before/after plot of 'B [%B]' and
                    '[Elutant] [mM]' for every sample (and every blank, if
                    given).

    Returns Data (and Blank, if given) with the correction applied; if
    LogShift is a number, the (last) dead volume + LogShift is appended,
    matching the original behaviour.
    """
    elution, deadV = _shiftDeadvolume(Data, Config, Para)
    if Plot:
        for before, after in zip(Data, elution):
            _plotStepChange(before, after, ['B [%B]', '[Elutant] [mM]'], 'Dead volume correction')

    if Blank is not None:
        cfgBlank = BlankConfig if BlankConfig is not None else Config
        blankOut, _ = _shiftDeadvolume(Blank, cfgBlank, Para)
        if Plot:
            for before, after in zip(Blank, blankOut):
                _plotStepChange(before, after, ['B [%B]', '[Elutant] [mM]'], 'Dead volume correction (blank)')

        if isinstance(LogShift, (int, float)):
            return elution, blankOut, deadV + LogShift
        return elution, blankOut

    if isinstance(LogShift, (int, float)):
        return elution, deadV + LogShift
    return elution


# ---------------------------------------------------------------------------
# getGradientRange
# ---------------------------------------------------------------------------
def _cropGradientRange(dataset, ExtraV):
    out = []
    for i in range(len(dataset)):
        dataiStart = dataset[i]['B [%B]'].ne(0).idxmax()
        dataiEnd = dataset[i].loc[dataiStart:]['B [%B]'].eq(0).idxmax() + ExtraV
        out.append(dataset[i].loc[dataiStart:dataiEnd].copy())
    return out


def getGradientRange(Data, ExtraV=0, Blank=None, Plot=False):
    """Crop each run down to the gradient elution range.

    Blank : optional list of blank runs to crop the same way (only sensible
            if the blank runs followed the same gradient program).
    Plot  : if True, show a before/after plot of 'UV [mAU]' and 'B [%B]'
            for every sample (and every blank, if given).
    """
    elution = _cropGradientRange(Data, ExtraV)
    if Plot:
        for before, after in zip(Data, elution):
            _plotStepChange(before, after, ['UV [mAU]', 'B [%B]'], 'Gradient range extraction')

    if Blank is not None:
        blankOut = _cropGradientRange(Blank, ExtraV)
        if Plot:
            for before, after in zip(Blank, blankOut):
                _plotStepChange(before, after, ['UV [mAU]', 'B [%B]'], 'Gradient range extraction (blank)')
        return elution, blankOut

    return elution


# ---------------------------------------------------------------------------
# restartVolume
# ---------------------------------------------------------------------------
def _restartVolume(dataset):
    out = []
    startV = None
    for i in range(len(dataset)):
        datai = dataset[i].copy()  # copy: avoid mutating the caller's dataframe
        startV = datai['Volume [mL]'].iloc[0]
        datai.loc[:, 'Volume [mL]'] = datai.loc[:, 'Volume [mL]'].subtract(startV)
        out.append(datai)
    return out, startV


def restartVolume(Data, LogShift=None, Blank=None, Plot=False):
    """Zero each run's volume axis to its own start.

    Blank : optional list of blank runs, zeroed the same way (each to its
            own start volume).
    Plot  : if True, show a before/after plot of 'UV [mAU]' for every
            sample (and every blank, if given) so you can see the shift.
    """
    elution, startV = _restartVolume(Data)
    if Plot:
        for before, after in zip(Data, elution):
            _plotStepChange(before, after, ['UV [mAU]'], 'Restart volume')

    if Blank is not None:
        blankOut, _ = _restartVolume(Blank)
        if Plot:
            for before, after in zip(Blank, blankOut):
                _plotStepChange(before, after, ['UV [mAU]'], 'Restart volume (blank)')

        if isinstance(LogShift, (int, float)):
            return elution, blankOut, startV + LogShift
        return elution, blankOut

    if isinstance(LogShift, (int, float)):
        return elution, startV + LogShift
    return elution


# ---------------------------------------------------------------------------
# correctBaseline
# ---------------------------------------------------------------------------
def _correctBaseline(dataset, TailV):
    out = []
    for i in range(len(dataset)):
        datai = dataset[i].copy()
        endV = datai['Volume [mL]'].iloc[-1]
        mask = datai['Volume [mL]'] >= (endV - TailV)

        if not mask.any():
            # Same failure mode as before (empty selection -> NaN mean -> NaN
            # everywhere), but now it's much harder to hit by accident since
            # the window is always anchored to the run's own end point.
            name = datai['Name'].iloc[0] if 'Name' in datai.columns else f'index {i}'
            raise ValueError(
                f"correctBaseline: TailV={TailV} mL is larger than the run "
                f"'{name}', which only spans {endV:.2f} mL - there's no data "
                "in the requested window. Use a smaller TailV."
            )

        mean_UV = datai['UV [mAU]'][mask].mean()
        datai['UV [mAU]'] = datai['UV [mAU]'] - mean_UV
        out.append(datai)
    return out


def correctBaseline(Data, TailV, Blank=None, Plot=False):
    """Zero the UV signal using its mean over the last TailV mL of each run.

    TailV : how many mL, counted back from the end of the run, to average
            over (e.g. TailV=3 uses each run's last 3 mL). Anchoring to the
            end of the run (rather than a fixed start/end window) means this
            keeps working regardless of how earlier steps shifted or cropped
            the volume axis (correctDeadvolume, getGradientRange,
            restartVolume).
    Blank : optional list of blank runs, baseline-corrected the same way
            (each over its own last TailV mL).
    Plot  : if True, show a before/after plot of 'UV [mAU]' for every
            sample (and every blank, if given).
    """
    elution = _correctBaseline(Data, TailV)
    if Plot:
        for before, after in zip(Data, elution):
            _plotStepChange(before, after, ['UV [mAU]'], 'Baseline correction')

    if Blank is not None:
        blankOut = _correctBaseline(Blank, TailV)
        if Plot:
            for before, after in zip(Blank, blankOut):
                _plotStepChange(before, after, ['UV [mAU]'], 'Baseline correction (blank)')
        return elution, blankOut

    return elution


# ---------------------------------------------------------------------------
# correctElutant
# ---------------------------------------------------------------------------
def correctElutant(Data, Config, Blank, Plot=False):
    """Subtract the elutant/blank UV signal from each sample run.

    Plot : if True, show a before/after plot of 'UV [mAU]' for every sample.
    (Blank itself isn't modified by this step, so there's nothing separate
    to apply to it here - make sure Blank has already been through the same
    correctDeadvolume/restartVolume steps as Data before calling this.)
    """
    data_list = []
    DeltaV = 0 #this used to be a parameter, for the akta run (that is why it is capitalized)
    for i in range(len(Data)):
        before = Data[i]
        datai = Data[i].copy()
        if Config['System'].iloc[i]=='SystemA' or Config['System'].iloc[i]=='SystemB':
            endV = datai['Volume [mL]'].iloc[-1]
            mean_UV_i = datai['UV [mAU]'][(datai['Volume [mL]'] >= 0) & (datai['Volume [mL]'] <= DeltaV)].mean()
            mean_UV_f = datai['UV [mAU]'][(datai['Volume [mL]'] >= endV-DeltaV) & (datai['Volume [mL]'] <= endV)].mean()
            datai['UV [mAU]'] = datai['UV [mAU]'] - ((mean_UV_f - mean_UV_i) / (endV - 0)*datai['Volume [mL]']+mean_UV_i)
            datai['UV [mAU]'] = datai['UV [mAU]'].clip(lower=0)

        elif Config['BlankFilename'].iloc[i]<=len(Blank)-1:
            # Function to find the closest value in Blank['Volume [mL]'] for each value in Data['Volume [mL]']
            blanki = Blank[int(Config['BlankFilename'].iloc[i])].copy()
            #crop blanki
            vol_min = datai.iloc[0]['Volume [mL]']
            vol_max = datai.iloc[-1]['Volume [mL]']
            blanki = blanki[
                (blanki['Volume [mL]'] >= vol_min) &
                (blanki['Volume [mL]'] <= vol_max)
                ].copy()
            #initial baseline correction
            N = 50  # number of initial points
            datai['UV [mAU]'] = datai['UV [mAU]'] - datai['UV [mAU]'].iloc[:N].mean()
            blanki['UV [mAU]'] = blanki['UV [mAU]'] - blanki['UV [mAU]'].iloc[:N].mean()
            #interpolation blanki
            blank_interp = np.interp(
                datai['Volume [mL]'],
                blanki['Volume [mL]'],
                blanki['UV [mAU]']
            )
            M=2000
            a = np.dot(blank_interp[-M:], datai['UV [mAU]'][-M:]) / np.dot(blank_interp[-M:],blank_interp[-M:])
            #substract
            datai['UV [mAU]'] = datai['UV [mAU]']-blank_interp*a
        else:
            print('It looks like we have an NGC run with no blank run we are going to assume a standard elution profile')
            endV = datai[datai['B [%B]'] == datai['B [%B]'].max()].index[-1] #find the last index where the maximum value occurs 
            #truncated_data = datai['B [%B]'].truncate(0, endV)  # Truncate the data up to endV
            #startV = truncated_data[truncated_data == 0].index[-1]  # Find the last

            blanki = datai.copy()
            # Scale the 'B [%B]' column with the value of 'UV [muA]' at endV
            scaling_factor = datai['UV [mAU]'].iloc[endV]
            blanki['UV [mAU]'] = datai['B [%B]'] * scaling_factor/100
            datai['UV [mAU]'] = datai['UV [mAU]']-blanki['UV [mAU]']

        data_list.append(datai)
        if Plot:
            _plotStepChange(before, datai, ['UV [mAU]'], 'Elutant/blank correction')

    return data_list
