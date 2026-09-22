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
functionDir= os.path.join(cwdPath,'functionsAktaNgcAnalyser') #this is the directory where the functions are stored
if functionDir not in sys.path:
    sys.path.append(functionDir)
del functionDir

#======2 -Load config
from LoadingAktaNgcAnalyser import loadConfig
config,para = loadConfig(os.path.join(cwdPath,configName))

#======3 -Load data and parse it
from LoadingAktaNgcAnalyser import loadAktaNgc, loadBlank, checkBlankGradientProfile
data = loadAktaNgc(config, cwdPath) #load the data
blank, config, blankConfig = loadBlank(config, cwdPath) #blankConfig tells us each blank's 'System' etc.
checkBlankGradientProfile(data, blank, config, ToleranceB=2.0) #warns if a run's %B gradient doesn't match its assigned blank

#======3.2 -Plot loaded data
from PlottingAktaNgcAnalyser import linePlot3axis, linePlotFractions
#for i in range(len(data)):
#    linePlot3axis(data[i])
    #linePlotFractions(data[i],config.iloc[i])

#======4 -Preprocessing data
from ProcessingAktaNgcAnalyser import correctDeadvolume, getGradientRange, correctBaseline, correctElutant, restartVolume

elution, blank = correctDeadvolume(data, config, para, Blank=blank, BlankConfig=blankConfig, Plot=False) #shift %B and elutant level with deadvolume, for data AND blank

elution, blank = correctBaseline(elution, TailV=10, Blank=blank, Plot=False) #set UV to 0 (TailV in mL)

elution, blank = getGradientRange(elution, ExtraV=500, Blank=blank, Plot=False) #extract elution profile, for data AND blank

elution = correctElutant(elution, config, blank, Plot=False) #correct for elutant signal either by subtracting blank run or by subtracting line from start to end

#elution, blank = restartVolume(elution, Blank=blank, Plot=False)

#======5 -Fit data
#todo: still need to port fitting code
