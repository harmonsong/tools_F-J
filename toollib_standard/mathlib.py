# Add incidents into map
import sys
sys.path.append(r'/home/songshuhao/anaconda3/bin/')

import numpy as np

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

def GetStationPairs(nsta):
    StationPair = []
    for ii in range(nsta):
        for jj in range(ii+1,nsta):
            StationPair.append(ii)
            StationPair.append(jj)
    StationPair = np.array(StationPair,dtype=np.int32)
    return StationPair

def Pairs(sta):
    p = []
    nsta = len(sta)
    for ii in range(nsta):
        for jj in range(ii+1,nsta):
            p.append([sta[ii],sta[jj]])
    return p
def cal_indx(pair,nsta):
    indx = int(pair[0]*(2*nsta-pair[0]-1)/2+pair[1]-pair[0]-1)
    return indx