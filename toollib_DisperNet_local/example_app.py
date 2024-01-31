import dispernet_local as dispernet
import numpy as np
import os


## Frequency Series For interpolation
freqSeries = np.arange(0,0.5,0.005)

## Lanuch DisperNetGUI
dispernet.App(filePath='./h5', curveFilePath='./curve/',freqSeries=freqSeries, trigerMode=False, searchStep=2, cmap='jet', periodCutRate=0.125, semiAutoRange=0.1, autoT=True)
# filePath: the file path of h5 files, you can use convertFormat.py to convert the format to h5.
# curveFilePath: the save path of curve files.
# freqSeries: frequency series for interpolation 
# trigerMode: all workflow of DisperNet (extract + mode separtion)
# searchStep: the raw search step of the CAE extractor, must be int data format.
# cmap: the colormap used in GUI display.
# periodCutRate: the ratio of freqency range used when mapping to peroid domain.
# semiAutoRange: the velocity searching range (ratio 0-1) for semi-auto add point.
# autoT: automatically transpose the spectra by velocity and frequency from h5 file


## Make the dataset of trainsfer training
#dispernet.createTrainSet('./trainSetDAS.h5', './h5/', './curve/')   # fileName, filePath, curveFilePath

