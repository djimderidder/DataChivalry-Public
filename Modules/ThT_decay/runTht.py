# -*- coding: utf-8 -*-
"""
Load a FLUOstar ThT disaggregation dataset for user-selected wells.
 
The data are normalized against reference controls.
The function visualizes both the raw and normalized fluorescence signals.
 
Data are not automatically saved.
@author: ridderdde
"""
#Define data
#1 -Load directory where functions are stored
import sys, os
from pathlib import Path

cwdPath = Path.cwd().parent

for functionDir in [
    cwdPath / ".." / "Shared",
    cwdPath / "ThT_decay" / "functions"
]:
    sys.path.append(str(functionDir.resolve()))

#1 -Give file name and path
fileName = #give filename of FLUOSTAR data as csv file
fileFolder =  #give folder where FLUOSTAR data is stored

#Load data and config
from loadData import FLUOstarData
data = FLUOstarData(fileFolder,fileName)

#2 -Give well names to load
roi = [key for key in data.dict][:6] #which wells are you loading

#roi = [' D1', ' D2', ' D3', ' D4', ' D5', ' D6', ' D7',
#       ' E1', ' E2', ' E3', ' E4', ' E5', ' E6', ' E7',
#       ' F1', ' F2', ' F3', ' F4', ' F5', ' F6', ' F7'] #example

import pandas as pd
#3 -To what wells are you normalizing nFibril is often without ATP and nBase is without fibers.
roi = pd.DataFrame({
    'well': roi,
    'nFibril': ['H11'] * len(roi),
    'nBase': ['H12' ] * len(roi)
})

#Reduce dataset to a single gain and cut first off first 5 minutes (equilibration time)
data.set_gain(Gain=3)
data.data = data.data[data.data['Time [s]']>600]

from processData import NormalizeThT
output = NormalizeThT(data,roi)

from visualizeData import PlotThT
PlotThT(Raw=data, Norm=output, Roi=roi)