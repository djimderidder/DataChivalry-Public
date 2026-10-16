import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from lmfit import Model

from PlottingAktaNgcAnalyser import linePlotProcess, linePlotFit


def _runName(datai, i):
    return datai['Name'].iloc[0] if 'Name' in datai.columns else f'index {i}'


def _perRun(value, i):
    """Allow a setting to be one value for all runs, or a list/array with
    one value per run."""
    if value is None or np.isscalar(value):
        return value
    return value[i]


# ---------------------------------------------------------------------------
# Shared helper: show a "before vs after" plot for every sample in a list,
# for whichever of `columns` are actually present in the data.
# ---------------------------------------------------------------------------
def _plotStepChange(before, after, columns, stepName):
    """Returns {column: ax} for each column actually plotted, so a caller
    can draw more on top of one of them (e.g. correctElutant overlaying its
    baseline fit and scaled blank curve on the 'UV [mAU]' axis)."""
    axes = {}
    for col in columns:
        if col in before.columns and col in after.columns:
            title = f"{after['Name'].iloc[0]} \u2014 {stepName}"
            fig, ax = linePlotProcess(after, before, col, title=title)
            axes[col] = ax
    return axes


# ---------------------------------------------------------------------------
# correctDeadvolume
# ---------------------------------------------------------------------------
_DEADVOLUME_COLUMNS = ['B [%B]', '[Elutant] [mM]']  # signals set by the pump, not measured at the detector

def _shiftDeadvolume(dataset, cfg, Para):
    """Apply the dead-volume shift to every dataframe in `dataset`, using the
    system type listed in `cfg` (same length/order as `dataset`). Only
    shifts whichever of _DEADVOLUME_COLUMNS are present - e.g.
    '[Elutant] [mM]' is missing when 'Max Elutant [mM]' isn't set."""
    out = []
    deadV = None
    for i in range(len(dataset)):
        system = cfg['System'].iloc[i]
        if system == 'SystemA':
            deadV = float(Para['deadVolumeSystemA'])
        elif system == 'SystemB':
            deadV = float(Para['deadVolumeSystemB'])
        else:
            deadV = float(Para['deadVolume'])
            print(f'Assuming data of index {i} is a NGC run')

        datai = dataset[i].copy()
        vol_original = datai['Volume [mL]']
        vol_shifted = vol_original + deadV

        cols = [c for c in _DEADVOLUME_COLUMNS if c in datai.columns]
        if not cols:
            print(f"Warning: '{_runName(datai, i)}' has none of {_DEADVOLUME_COLUMNS} "
                  "- nothing to shift for the dead volume.")
        for col in cols:
            interp = interp1d(vol_shifted, datai[col], bounds_error=False, fill_value='extrapolate')
            datai[col] = interp(vol_original)

        out.append(datai)
    return out, deadV


def correctDeadvolume(Data, Config, Para, LogShift=None, Blank=None, BlankConfig=None, Plot=False):
    """Shift %B and elutant concentration by the system dead volume.

    Data / Config : the sample runs and their config.
    Blank         : optionally, the list of blank runs (e.g. from loadBlank).
                    When given, exactly the same correction is applied to
                    them, so data and blank stay on a consistent volume
                    basis for later steps (e.g. correctElutant).
    BlankConfig   : the config belonging to Blank (e.g. from loadBlank),
                    used to look up each blank's 'System'. Defaults to
                    Config if not given.
    Plot          : if True, show a before/after plot of 'B [%B]' and
                    '[Elutant] [mM]' for every sample (and every blank).

    Returns Data (and Blank, if given) with the correction applied; if
    LogShift is a number, the (last) dead volume + LogShift is appended.
    """
    elution, deadV = _shiftDeadvolume(Data, Config, Para)
    if Plot:
        for before, after in zip(Data, elution):
            _plotStepChange(before, after, _DEADVOLUME_COLUMNS, 'Dead volume correction')

    if Blank is not None:
        cfgBlank = BlankConfig if BlankConfig is not None else Config
        blankOut, _ = _shiftDeadvolume(Blank, cfgBlank, Para)
        if Plot:
            for before, after in zip(Blank, blankOut):
                _plotStepChange(before, after, _DEADVOLUME_COLUMNS, 'Dead volume correction (blank)')

        if isinstance(LogShift, (int, float)):
            return elution, blankOut, deadV + LogShift
        return elution, blankOut

    if isinstance(LogShift, (int, float)):
        return elution, deadV + LogShift
    return elution


# ---------------------------------------------------------------------------
# getGradientRange
# ---------------------------------------------------------------------------
def _cropGradientRange(dataset, ExtraV, Config=None, ExtraCV=None):
    out = []
    useCV = ExtraCV is not None and Config is not None and 'CV [mL]' in Config.columns
    if ExtraCV is not None and not useCV:
        print("Warning: ExtraCV was given but no 'CV [mL]' column was found in the "
              "config - using ExtraV (a row count) instead.")

    for i in range(len(dataset)):
        datai = dataset[i]
        dataiStart = datai['B [%B]'].ne(0).idxmax()
        zeroIdx = datai.loc[dataiStart:]['B [%B]'].eq(0).idxmax()

        cv = Config['CV [mL]'].iloc[i] if useCV else np.nan
        if useCV and pd.isna(cv):
            print(f"Warning: no 'CV [mL]' value for '{_runName(datai, i)}' - "
                  "using ExtraV (a row count) for this run instead.")

        if pd.notna(cv):
            # Volume-accurate path: extend by a number of column volumes,
            # independent of how densely the run was sampled.
            endVol = datai.loc[zeroIdx, 'Volume [mL]'] + ExtraCV * float(cv)
            tail = datai.loc[zeroIdx:]
            dataiEnd = tail[tail['Volume [mL]'] <= endVol].index[-1]
        else:
            # Legacy path: ExtraV is a row/point count, not a volume,
            # despite the name - kept as-is for backward compatibility.
            dataiEnd = zeroIdx + ExtraV

        out.append(datai.loc[dataiStart:dataiEnd].copy())
    return out


def getGradientRange(Data, ExtraV=0, ExtraCV=None, Config=None, Blank=None, BlankConfig=None, Plot=False):
    """Crop each run down to the gradient elution range.

    ExtraV      : extra rows to keep past the point where %B returns to 0.
                  This is a row/point count, not a volume, despite the name
                  (historical). Ignored for runs where ExtraCV is usable.
    ExtraCV     : extra volume to keep past the point where %B returns to 0,
                  as a number of column volumes. Looked up per run as
                  ExtraCV * Config['CV [mL]'], so it scales correctly across
                  runs/columns with different CVs. Requires Config (and
                  BlankConfig for the blanks) to have a 'CV [mL]' column;
                  falls back to ExtraV per run when that's missing/empty.
                  Keep this modest - too large and you'll crop into whatever
                  comes after elution (e.g. a wash step).
    Config      : needed only to look up 'CV [mL]' when ExtraCV is given.
    BlankConfig : same, for the blank runs; defaults to Config.
    Blank       : optional list of blank runs to crop the same way (only
                  sensible if the blanks followed the same gradient program).
    Plot        : if True, show a before/after plot of 'UV [mAU]' and
                  'B [%B]' for every sample (and every blank, if given).
    """
    elution = _cropGradientRange(Data, ExtraV, Config, ExtraCV)
    if Plot:
        for before, after in zip(Data, elution):
            _plotStepChange(before, after, ['UV [mAU]', 'B [%B]'], 'Gradient range extraction')

    if Blank is not None:
        cfgBlank = BlankConfig if BlankConfig is not None else Config
        blankOut = _cropGradientRange(Blank, ExtraV, cfgBlank, ExtraCV)
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
            raise ValueError(
                f"correctBaseline: TailV={TailV} mL is larger than the run "
                f"'{_runName(datai, i)}', which only spans {endV:.2f} mL - there's no data "
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
            end of the run means this keeps working regardless of how
            earlier steps shifted or cropped the volume axis.
    Blank : optional list of blank runs, baseline-corrected the same way.
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
def _baselineDrift(datai, EdgeV, Mode='linear'):
    """Estimate the UV baseline to subtract from a run, as either a flat
    offset or a straight line between the start and end of the run.

    EdgeV : width (mL) of the window averaged at each edge to estimate the
            baseline. Widen it if a run is noisy, narrow it if the window
            reaches into the gradient/elution itself. Since this runs after
            getGradientRange has cropped the run to its elution range, keep
            EdgeV small relative to that range, or the "baseline" windows
            average in real peak signal. Raises if the two edge windows
            would overlap.
    Mode  : 'start'  - flat offset: subtract the mean UV over the first
                        EdgeV mL only. Assumes no detector drift.
            'linear' - subtract a straight line from the mean UV over the
                        first EdgeV mL to the mean UV over the last EdgeV mL.
                        Corrects for a baseline that drifts over a run.
    """
    vol = datai['Volume [mL]']
    startV = vol.iloc[0]
    endV = vol.iloc[-1]
    span = endV - startV
    name = datai['Name'].iloc[0] if 'Name' in datai.columns else '?'

    if Mode == 'linear' and 2 * EdgeV >= span:
        raise ValueError(
            f"_baselineDrift: BaselineWindow={EdgeV} mL is too wide for "
            f"'{name}', which only spans {span:.2f} mL after cropping - the "
            f"start and end windows would overlap into the elution peak "
            f"itself. Use something well under {span / 2:.2f} mL."
        )
    elif EdgeV > span:
        raise ValueError(
            f"_baselineDrift: BaselineWindow={EdgeV} mL is wider than "
            f"'{name}''s own span ({span:.2f} mL)."
        )

    meanStart = datai['UV [mAU]'][(vol >= startV) & (vol <= startV + EdgeV)].mean()

    if Mode == 'start':
        return meanStart
    elif Mode == 'linear':
        meanEnd = datai['UV [mAU]'][(vol >= endV - EdgeV) & (vol <= endV)].mean()
        slope = (meanEnd - meanStart) / span
        return meanStart + slope * (vol - startV)
    else:
        raise ValueError(f"_baselineDrift: unknown Mode '{Mode}' - use 'start' or 'linear'.")


def _overlayBaselineFit(ax, vol, baseline, label='Baseline fit'):
    """Draw the baseline that was subtracted from a run on top of its
    before/after plot. baseline is a scalar for Mode='start', a Series
    aligned to vol for Mode='linear'."""
    if np.isscalar(baseline):
        ax.axhline(baseline, color='0.4', linestyle=':', linewidth=1, label=label)
    else:
        ax.plot(vol, baseline, color='0.4', linestyle=':', linewidth=1, label=label)


def _overlayBlankScale(ax, vol, scaledBlank, a, scaleStartV):
    """Draw the scaled blank curve that was subtracted, mark where the
    scale-fitting window starts, and print the fitted 'a' on the plot."""
    ax.plot(vol, scaledBlank, color='0.2', linestyle='-.', linewidth=1, label='Blank × a')
    ax.axvline(scaleStartV, color='0.6', linestyle=':', linewidth=1)
    ax.text(0.02, 0.95, f'a = {a:.3f}', transform=ax.transAxes, fontsize=8, va='top')
    ax.legend(fontsize=8)


def correctElutant(Data, Config, Blank, Plot=False, BaselineMode='linear', BaselineWindow=1, ScaleStartV=None):
    """Subtract the elutant/blank UV signal from each sample run.

    Three cases per run:
      - SystemA/SystemB (AKTA) runs: subtract a baseline (see BaselineMode)
        and clip at 0.
      - NGC runs with a blank assigned: baseline-correct both sample and
        blank, scale the blank to the sample over a window where only
        elutant/buffer signal is expected, and subtract it.
      - NGC runs without a blank: subtract a %B-shaped profile scaled to
        the UV signal at the end of the gradient.

    BaselineMode   : 'start' (flat offset from the first BaselineWindow mL)
                     or 'linear' (straight line from the first to the last
                     BaselineWindow mL). See _baselineDrift.
    BaselineWindow : width (mL) of the edge windows used to estimate the
                     baseline. Keep it small relative to the cropped
                     elution range (raises if the two windows overlap).
    ScaleStartV    : for NGC runs with a blank, the volume (mL) from which
                     the blank's scale factor 'a' is fitted - both signals
                     should contain only elutant/buffer signal there.
                     One value for all runs, or a list with one value per
                     run. Defaults to the volume of the blank's UV maximum,
                     which assumes the sample's peaks elute before that
                     point - override if that doesn't hold for your runs.
                     'a' is clipped to 0 (with a warning) if the fit comes
                     out negative.
    Plot           : if True, show a before/after plot of 'UV [mAU]' for
                     every sample, with the subtracted baseline and scaled
                     blank overlaid.

    Blank itself isn't modified here - make sure it has been through the
    same earlier steps as Data before calling this.
    """
    data_list = []
    for i in range(len(Data)):
        before = Data[i]
        datai = Data[i].copy()
        name = _runName(datai, i)
        baselineLine = None   # sample's fitted baseline, for the Plot overlay
        blankOverlay = None   # (scaled blank curve, a, scaleStart), for the Plot overlay
        blankIdx = Config['BlankFilename'].iloc[i]

        if Config['System'].iloc[i] in ('SystemA', 'SystemB'):
            baselineLine = _baselineDrift(datai, BaselineWindow, BaselineMode)
            datai['UV [mAU]'] = datai['UV [mAU]'] - baselineLine
            datai['UV [mAU]'] = datai['UV [mAU]'].clip(lower=0)

        elif pd.notna(blankIdx) and int(blankIdx) < len(Blank):
            blanki = Blank[int(blankIdx)].copy()
            # crop blank to the sample's volume range
            vol_min = datai['Volume [mL]'].iloc[0]
            vol_max = datai['Volume [mL]'].iloc[-1]
            blanki = blanki[
                (blanki['Volume [mL]'] >= vol_min) &
                (blanki['Volume [mL]'] <= vol_max)
                ].copy()
            # baseline-correct both before comparing them
            baselineLine = _baselineDrift(datai, BaselineWindow, BaselineMode)
            datai['UV [mAU]'] = datai['UV [mAU]'] - baselineLine
            blanki['UV [mAU]'] = blanki['UV [mAU]'] - _baselineDrift(blanki, BaselineWindow, BaselineMode)
            # interpolate blank onto the sample's volume points
            blank_interp = np.interp(
                datai['Volume [mL]'],
                blanki['Volume [mL]'],
                blanki['UV [mAU]']
            )
            # Fit the blank's amplitude only over a region with no sample peaks
            scaleStart = _perRun(ScaleStartV, i)
            if scaleStart is None:
                scaleStart = blanki['Volume [mL]'].iloc[blanki['UV [mAU]'].values.argmax()]
            scaleMask = (datai['Volume [mL]'] >= scaleStart).values
            if scaleMask.sum() < 20:
                raise ValueError(
                    f"correctElutant: the blank-scaling window for '{name}' "
                    f"(from {scaleStart:.2f} mL to the end of the run) only "
                    f"contains {scaleMask.sum()} points - too narrow to fit "
                    "'a' reliably. Pass an earlier ScaleStartV for this run."
                )

            a = np.dot(blank_interp[scaleMask], datai['UV [mAU]'][scaleMask]) / np.dot(blank_interp[scaleMask], blank_interp[scaleMask])
            if a < 0:
                # A negative scale factor would mean adding the blank back
                # in, which isn't physically meaningful - clip and flag it.
                print(f"Warning: fitted blank-scaling factor a={a:.3f} for '{name}' was negative - "
                      "clipped to 0 (no blank subtracted). Check the blank assignment and ScaleStartV for this run.")
                a = 0.0
            blankOverlay = (blank_interp * a, a, scaleStart)
            datai['UV [mAU]'] = datai['UV [mAU]'] - blank_interp * a

        else:
            print(f"'{name}' looks like an NGC run with no blank run - assuming a standard elution profile")
            # last index *label* where %B is at its maximum (use .loc, not
            # .iloc: after getGradientRange labels no longer match positions)
            endIdx = datai.index[datai['B [%B]'] == datai['B [%B]'].max()][-1]
            scaling_factor = datai['UV [mAU]'].loc[endIdx]
            datai['UV [mAU]'] = datai['UV [mAU]'] - datai['B [%B]'] * scaling_factor / 100

        data_list.append(datai)
        if Plot:
            axes = _plotStepChange(before, datai, ['UV [mAU]'], 'Elutant/blank correction')
            ax = axes.get('UV [mAU]')
            if ax is not None:
                if baselineLine is not None:
                    _overlayBaselineFit(ax, before['Volume [mL]'], baselineLine)
                if blankOverlay is not None:
                    scaledBlank, a, scaleStart = blankOverlay
                    _overlayBlankScale(ax, datai['Volume [mL]'], scaledBlank, a, scaleStart)
                elif baselineLine is not None:
                    ax.legend(fontsize=8)

    return data_list


# ---------------------------------------------------------------------------
# fitData
# ---------------------------------------------------------------------------
def _momentEstimate(x, y, name):
    """Centroid, variance and area of a profile - the model-independent
    starting point that Estimator turns into initial parameter guesses.
    Negative y values are ignored for the centroid/variance."""
    yPos = np.clip(y, 0, None)
    total = yPos.sum()
    if total <= 0:
        raise ValueError(f"fitData: '{name}' has no positive signal to fit.")
    mu = (x * yPos).sum() / total
    var = (yPos * (x - mu) ** 2).sum() / total
    amp = y.sum() * (x[1] - x[0])  # area, assuming even sampling
    return mu, var, amp


def _sigmaWindowBounds(x, mu, N, Offset):
    """Window used by _estimateSigma: it starts a fraction Offset of the way
    from mu to the end of the range and is N wide (same units as x).
    Returns (windowStart, windowEnd), or (None, None) if mu is at or past
    the end of the range."""
    span = x[-1] - mu
    if span <= 0:
        return None, None
    windowStart = mu + Offset * span
    return windowStart, windowStart + N


def _estimateSigma(x, y, mu, N=1.0, Offset=0.05):
    """Estimate a run's measurement noise (used as a uniform Sigma) from a
    small local window after the peak (see _sigmaWindowBounds).

    Keeping the window narrow (N) keeps it in a regime where a straight
    line is a good local approximation of the remaining peak shape, so the
    scatter around a linear detrend is a noise estimate rather than
    curvature. Offset sets how far past mu the window starts, as a fraction
    of the distance from mu to the end of the range - increase it if the
    window still sits on a steep part of the peak.

    mu is the pre-fit centroid, so this never looks at the fit it's meant
    to calibrate. Returns a scalar sigma, or None if there's no usable
    window (peak at the end of the range, or fewer than 10 points).
    """
    windowStart, windowEnd = _sigmaWindowBounds(x, mu, N, Offset)
    if windowStart is None:
        return None
    mask = (x >= windowStart) & (x <= windowEnd)
    if mask.sum() < 10:
        return None
    xw, yw = x[mask], y[mask]
    coeffs = np.polyfit(xw, yw, 1)
    resid = yw - np.polyval(coeffs, xw)
    sigma = resid.std(ddof=2)  # ddof=2: slope + intercept were fitted
    return sigma if sigma > 0 else None


def _fitWeights(x, y, mu, Sigma, SigmaWindowN, SigmaWindowOffset, name):
    """Returns (weights or None, sigmaUsed or None, (windowStart, windowEnd))."""
    if Sigma is None:
        return None, None, (None, None)
    if isinstance(Sigma, str):
        if Sigma != 'auto':
            raise ValueError(f"fitData: unknown Sigma='{Sigma}' - use a number, None, or 'auto'.")
        window = _sigmaWindowBounds(x, mu, SigmaWindowN, SigmaWindowOffset)
        sigma = _estimateSigma(x, y, mu, SigmaWindowN, SigmaWindowOffset)
        if sigma is None:
            print(f"Warning: 'auto' Sigma had no usable window for '{name}' - "
                  "falling back to an unweighted fit for this run.")
            return None, None, window
        return np.full(len(x), 1.0 / sigma), sigma, window
    return np.full(len(x), 1.0 / float(Sigma)), float(Sigma), (None, None)


def _peakSummary(xy, x, yFit, Xi):
    """Model-independent peak position from the fitted curve: its apex and
    its centroid, plus the elutant concentration at the centroid if
    '[Elutant] [mM]' is available."""
    summary = {f'Apex ({Xi})': x[np.argmax(yFit)]}
    w = np.clip(yFit, 0, None)
    if w.sum() > 0:
        centroid = (x * w).sum() / w.sum()
        summary[f'Centroid ({Xi})'] = centroid
        if '[Elutant] [mM]' in xy.columns:
            summary['[Elutant] at centroid [mM]'] = np.interp(centroid, x, xy['[Elutant] [mM]'].values)
    return summary


def _fitMetricsText(name, result, paramNames, guess, sigmaUsed=None, peak=None, derived=None):
    """Format fit-quality numbers into a short multi-line string for the
    side panel of the fit plot."""
    residual = result.data - result.best_fit
    ssRes = np.sum(residual ** 2)
    ssTot = np.sum((result.data - np.mean(result.data)) ** 2)
    r2 = 1 - ssRes / ssTot if ssTot else float('nan')

    lines = [name, '-' * len(name), '']
    if sigmaUsed is not None:
        lines.append(f"Sigma used = {sigmaUsed:.4g}")
    lines.append(f"Reduced χ² = {result.redchi:.4g}")
    lines.append(f"χ² = {result.chisqr:.4g}")
    lines.append(f"R² = {r2:.5f}")
    lines.append('')
    lines.append('Fitted (± stderr, %):')
    for p in paramNames:
        val = result.params[p].value
        stderr = result.params[p].stderr
        if stderr:
            pct = 100 * stderr / val if val else float('nan')
            lines.append(f"  {p} = {val:.4g} ± {stderr:.3g} ({pct:.2f}%)")
        else:
            lines.append(f"  {p} = {val:.4g}")
    lines.append('')
    lines.append('Initial guess:')
    for p, g in zip(paramNames, guess):
        lines.append(f"  {p} = {g:.4g}")
    for header, values in (('Peak:', peak), ('Derived:', derived)):
        if values:
            lines.append('')
            lines.append(header)
            for k, v in values.items():
                lines.append(f"  {k} = {v:.4g}")
    return '\n'.join(lines)


def fitData(Func, Data, Xi, Yi, Estimator, Sigma=None, SigmaWindowN=1.0, SigmaWindowOffset=0.05,
            Bounds=None, Derived=None, Plot=False):
    """Fit a user-supplied model Func to each run's (Xi, Yi) data via lmfit.

    Func     : model function Func(x, p1, p2, ...). The first argument must
               be called x; the remaining argument names become the fit
               parameter names.
    Estimator: Estimator(mu, var, amp) -> initial guesses for Func's
               parameters, in Func's argument order. mu/var/amp are the
               centroid, variance and area of each run's profile, so
               Estimator is where you translate those into your model's
               parameterisation.
    Sigma    : measurement-uncertainty weighting (uniform per run, passed to
               lmfit as weights=1/Sigma).
               - None   : unweighted fit. Fine for point estimates, but
                          (reduced) chi-square is then not a calibrated
                          goodness-of-fit statistic.
               - number : used as-is for every run.
               - 'auto' : estimated per run by _estimateSigma from a small
                          window after the peak, before the fit runs. Falls
                          back to an unweighted fit (with a warning) if
                          that window isn't usable for a run.
    SigmaWindowN      : 'auto' only - width of the noise window (Xi units).
                        Keep it small; widen it if a run's window ends up
                        with too few points.
    SigmaWindowOffset : 'auto' only - where the noise window starts, as a
                        fraction of the distance from the centroid to the
                        end of the range.
    Bounds   : optional {param_name: (min, max)}. Parameters not listed are
               bounded to (0, inf).
    Derived  : optional Derived(*fitted_params) -> {label: value}, for
               model-specific quantities you want reported per run.
    Plot     : if True, show data/initial guess/fit per run with a side
               panel of fit metrics (and the 'auto' Sigma window marked).
               If False, the same information is printed to the console.

    Returns (fitted parameters, initial guesses, [(chi2, reduced chi2)]),
    with parameters in Func's argument order, one row per run.
    """
    model = Model(Func, independent_vars=['x'])
    paramNames = model.param_names
    Bounds = Bounds or {}
    unknown = set(Bounds) - set(paramNames)
    if unknown:
        raise ValueError(f"fitData: Bounds given for unknown parameter(s) {sorted(unknown)} - "
                         f"Func's parameters are {paramNames}.")

    fitPara, guessPara, chi_squares = [], [], []

    for i in range(len(Data)):
        xy = Data[i]
        name = _runName(xy, i)
        print(f'====={name}=====')
        x = xy[Xi].values.astype(float)
        y = xy[Yi].values.astype(float)

        # ==== 1. Initial parameter estimation ====
        mu, var, amp = _momentEstimate(x, y, name)
        guess = list(Estimator(mu, var, amp))
        if len(guess) != len(paramNames):
            raise ValueError(
                f"fitData: Estimator returned {len(guess)} values but Func has "
                f"{len(paramNames)} parameters ({', '.join(paramNames)}) - return one "
                "initial guess per parameter, in Func's argument order."
            )

        # ==== 2. Parameters and bounds ====
        params = model.make_params(**dict(zip(paramNames, guess)))
        for p in paramNames:
            lo, hi = Bounds.get(p, (0, np.inf))
            params[p].set(min=lo, max=hi)

        # ==== 3. Fit ====
        weights, sigmaUsed, sigmaWindow = _fitWeights(x, y, mu, Sigma, SigmaWindowN, SigmaWindowOffset, name)
        result = model.fit(y, params, x=x, weights=weights)
        popt = [result.params[p].value for p in paramNames]

        peak = _peakSummary(xy, x, result.best_fit, Xi)
        derived = Derived(*popt) if Derived is not None else None

        # ==== 4. Report ====
        if not Plot:
            print(f'initial centroid = {mu}; variance = {var}; area = {amp}')
            print(f'initial parameters = {dict(zip(paramNames, guess))}')
            if sigmaUsed is not None:
                print(f'Sigma used = {sigmaUsed}')
            print(f'fitted parameters = {dict(zip(paramNames, popt))}')
            for k, v in peak.items():
                print(f'{k} = {v}')
            if derived:
                for k, v in derived.items():
                    print(f'{k} = {v}')
            print(result.fit_report())
        else:
            metricsText = _fitMetricsText(name, result, paramNames, guess, sigmaUsed, peak, derived)
            fig, ax, axText = linePlotFit(xy, Xi, Yi, Func, guess, popt, metricsText, title=name)
            windowStart, windowEnd = sigmaWindow
            if windowStart is not None:
                # Mark the window 'auto' Sigma used, to check it landed somewhere sensible
                ax.axvline(windowStart, color='0.5', linestyle=':', linewidth=1, label='Sigma window')
                ax.axvline(windowEnd, color='0.5', linestyle=':', linewidth=1)
                ax.legend(fontsize=8)

        fitPara.append(popt)
        guessPara.append(guess)
        chi_squares.append((result.chisqr, result.redchi))

    return np.array(fitPara), np.array(guessPara), chi_squares