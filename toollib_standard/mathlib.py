# Add incidents into map
import sys
sys.path.append(r'/home/songshuhao/anaconda3/bin/')

import numpy as np
import obspy
from obspy import UTCDateTime
#from ccfj import CC
#from ccfj import GetStationPairs
from concurrent.futures import ThreadPoolExecutor
import os
import time
from geopy.distance import great_circle
import folium
import matplotlib.pyplot as plt

# frequency domain to time domain
def freq_time(ncfs):
    ncfst = np.zeros([np.size(ncfs,0),2*(np.size(ncfs,1)-1)])
    for i in range(len(ncfs)):
        ncfst[i,:] = np.real(np.fft.fftshift(np.fft.irfft(ncfs[i,:])))
    return ncfst

def time_freq(ncfst):
    ncfs= np.zeros([len(ncfst),int((np.size(ncfst,1)+2)/2)],dtype=complex)
    for i in range(len(ncfst)):
        ncfs[i, :] = np.fft.rfft(np.fft.ifftshift(ncfst[i, :]))
    return ncfs