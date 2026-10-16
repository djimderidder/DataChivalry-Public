#0 -Make sure you have installed all used packages
'''
py -m pip --version
pip install -r requirements.txt
'''
#======0 define directory
import os
cwdPath = os.path.dirname(os.path.abspath(__file__))
configName = "config_MAIN.xlsx" #TO DO add mock config file to Chromatogram_fitter

#======1 -Load directory where functions are stored
import sys
functionDir = os.path.join(cwdPath, 'functions') #this is the directory where the functions are stored
if functionDir not in sys.path:
    sys.path.append(functionDir)
del functionDir

#======2 -Load config
from LoadingAktaNgcAnalyser import loadConfig
config, para = loadConfig(os.path.join(cwdPath, configName))

#======3 -Load data and parse it
from LoadingAktaNgcAnalyser import loadAktaNgc, loadBlank, checkBlankGradientProfile
data = loadAktaNgc(config, cwdPath) #load the data
blank, config, blankConfig = loadBlank(config, cwdPath) #blankConfig tells us each blank's 'System' etc.
checkBlankGradientProfile(data, blank, config, ToleranceB=2.0) #warns (doesn't stop) if a run's %B gradient doesn't match its assigned blank

#======3.2 -Plot loaded data
from PlottingAktaNgcAnalyser import linePlot3axis, linePlotFractions
#for i in range(len(data)):
#    linePlot3axis(data[i])
    #linePlotFractions(data[i],config.iloc[i])

#======4 -Preprocessing data
from ProcessingAktaNgcAnalyser import correctDeadvolume, getGradientRange, correctBaseline, correctElutant, restartVolume

elution, blank = correctDeadvolume(data, config, para, Blank=blank, BlankConfig=blankConfig, Plot=False) #shift %B and elutant level with deadvolume, for data AND blank

elution, blank = correctBaseline(elution, TailV=10, Blank=blank, Plot=False) #set UV to 0 (TailV in mL)

elution, blank = getGradientRange(elution, ExtraCV=0.5, Config=config, BlankConfig=blankConfig, Blank=blank, Plot=False) #extract elution profile, extended by ExtraCV column volumes (needs a 'CV [mL]' config column, else falls back to ExtraV)

elution = correctElutant(elution, config, blank, Plot=False, BaselineMode='linear', BaselineWindow=0.2) #correct for elutant signal either by subtracting the scaled blank run or a baseline

elution, blank = restartVolume(elution, Blank=blank, Plot=False) #zero each run's volume axis

#======5 -Fit data
# ===========================================================================
#  EXAMPLE MODEL - REPLACE WITH YOUR OWN
#  The three functions below define a plain Gaussian peak. They're only here
#  so the script runs out of the box and to show what fitData expects.
#  A Gaussian is symmetric, so it's often a poor fit for real elution peaks
#  (which tend to tail) - replace func, estimator and derived with a model
#  that suits your own data. fitData itself doesn't need to change.
#
#  What fitData expects:
#   - func(x, p1, p2, ...): first argument must be named x; the remaining
#     argument names become the fit parameter names (shown in plots/output).
#   - estimator(mu, var, amp): turns each run's centroid (mu), variance (var)
#     and area (amp) into initial guesses, one per parameter, in the same
#     order as func's arguments.
#   - derived(*params) (optional): returns {label: value} for any extra
#     model-specific quantities you want reported per run.
#   - By default every parameter is bounded to >= 0; pass
#     Bounds={'name': (min, max)} to fitData to change that.
# ===========================================================================
import numpy as np
from ProcessingAktaNgcAnalyser import fitData

def func(x, area, center, width):
    """EXAMPLE: area-normalised Gaussian peak (width = standard deviation)."""
    return area / (width * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - center) / width) ** 2)

def estimator(mu, var, amp):
    """EXAMPLE: for a Gaussian the moments map directly onto the parameters."""
    return amp, mu, np.sqrt(var)

def derived(area, center, width):
    """EXAMPLE: full width at half maximum of the Gaussian."""
    return {'FWHM': 2 * np.sqrt(2 * np.log(2)) * width}
# ======================= END OF EXAMPLE MODEL ==============================

fitPara, estPara, chi_squares = fitData(func, elution, 'Volume [mL]', 'UV [mAU]', estimator,
                                        Sigma='auto', SigmaWindowN=0.5, Derived=derived, Plot=True)
#======6 -Optionally save fit results to the config spreadsheet (Excel must be closed)
#from LoadingAktaNgcAnalyser import saveConfig
#saveConfig(config, fitPara, chi_squares, os.path.join(cwdPath, configName))