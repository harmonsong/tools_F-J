# plot day distribution
import sys
sys.path.append(r'/home/harmon/anaconda3/bin/')

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
import pandas as pd

from toollib_standard import mathlib


# ncfst
def plot_ncfst(ax,t,ncfst,r0,title0,flag_time,xlim,index):
    #fig,ax = plt.subplots(ncols=1,figsize=(10,4))
    for i in range(0,len(r0)):
        if np.max(ncfst[i,:]) == 0:
            continue
        ax.plot(t,ncfst[i,:]/np.max(ncfst[i,:])*flag_time+r0[i],'k',linewidth=0.5)
            
    ax.set_xlim(xlim)
    if index == 0:
        ax.set_title(title0)
    else:
        ax.set_title('('+chr(96+index)+')',loc='left')
    #ax.set_xlabel('time/s')
    #ax.set_ylabel('Station pair distance/km')
    return ax

# ncfst contrast
def plot_ncfst_contrast(ax,t,ncfst,label1,ncfst_or,label2,r0,title0,flag_time,xlim,index):
    #fig,ax = plt.subplots(ncols=1,figsize=(10,4))
    flag = 0
    for i in range(0,len(r0)):
        if np.max(ncfst[i,:]) == 0:
            continue
        ax.plot(t,ncfst_or[i,:]/np.max(ncfst_or[i,:])*flag_time+r0[i],'r',linewidth=0.5)
        ax.plot(t,ncfst[i,:]/np.max(ncfst[i,:])*flag_time+r0[i],'k',linewidth=0.5)
        if flag == 0:
            flag = 1
            ax.plot(t,ncfst_or[i,:]/np.max(ncfst_or[i,:])*flag_time+r0[i],'r',linewidth=0.5,label = label2)
            ax.plot(t,ncfst[i,:]/np.max(ncfst[i,:])*flag_time+r0[i],'k',linewidth=0.5,label = label1)
            
    ax.set_xlim(xlim)
    if index == 0:
        ax.set_title(title0)
    else:
        ax.set_title('('+chr(96+index)+')',loc='left')
    #ax.set_xlabel('time/s')
    #ax.set_ylabel('Station pair distance/km')
    ax.legend()
    return ax

# ncfst timewindoww
def plot_ncfst_timewindow(ax,t,ncfst,r0,title0,flag_time,v,tao_max,xlim,index):
    #fig,ax = plt.subplots(ncols=1,figsize=(10,4))
    for i in range(0,len(r0)):
        #if r0[i] < 100:
        #    continue
        if np.max(ncfst[i,:]) == 0:
            continue
        ax.plot(t,ncfst[i,:]/np.max(ncfst[i,:])*flag_time+r0[i],'k',linewidth=0.5)
        ax.plot(r0[i]/v-2*tao_max,r0[i],'r.')
        ax.plot(r0[i]/v+2*tao_max,r0[i],'r.')
        ax.plot(-r0[i]/v+2*tao_max,r0[i],'r.')
        ax.plot(-r0[i]/v-2*tao_max,r0[i],'r.')
            
    ax.set_xlim(xlim)
    if index == 0:
        ax.set_title(title0)
    else:
        ax.set_title('('+chr(96+index)+')',loc='left')
    #ax.set_xlabel('time/s')
    #ax.set_ylabel('Station pair distance/km')
    return ax



def plot_ncfst_contrast_timewindow(ax,t,ncfst,label1,ncfst_or,label2,r0,title0,flag_time,xlim,index):
    #fig,ax = plt.subplots(ncols=1,figsize=(10,4))
    flag = 0
    for i in range(0,len(r0)):
        #if np.max(ncfst[i,:]) == 0:
        #    continue
        ax.plot(t,ncfst_or[i,:]/np.max(ncfst_or[i,:])*flag_time+r0[i],'r',linewidth=0.5)
        ax.plot(t,ncfst[i,:]/np.max(ncfst[i,:])*flag_time+r0[i],'k',linewidth=0.5)
        
        if flag == 0:
            flag = 1
            ax.plot(t,ncfst_or[i,:]/np.max(ncfst_or[i,:])*flag_time+r0[i],'r',linewidth=0.5,label = label2)
            ax.plot(t,ncfst[i,:]/np.max(ncfst[i,:])*flag_time+r0[i],'k',linewidth=0.5,label = label1)
            
            
    ax.set_xlim(xlim)
    if index == 0:
        ax.set_title(title0)
    else:
        ax.set_title('('+chr(96+index)+')',loc='left')
    #ax.set_xlabel('time/s')
    #ax.set_ylabel('Station pair distance/km')
    ax.legend()
    return ax

# ncfs
def plot_ncfs(ax,f,ncfs0,r0,title0,xlim,index,flag_im = 0):
    indx = np.argsort(r0)
    r  = r0[indx]
    ncfs1 = ncfs0[indx,:]
    ncfs = np.zeros((len(r),len(f)),dtype = complex)
    for i in range(0,len(r)):
        if np.max(ncfs1[i,:]) == 0:
            continue
        ncfs[i,:] = ncfs1[i,:]/np.max(np.real(ncfs1[i,:]))

    im = ax.imshow(np.flip(np.real(ncfs),0), extent=[min(f),max(f), min(r0), max(r0)], aspect='auto',cmap='jet',vmin=-0.5,vmax=0.5)
    ax.set_xlim(xlim)
    if index == 0:
        ax.set_title(title0)
    else:
        ax.set_title('('+chr(96+index)+')',loc='left')
    #ax.set_xlabel('Frequency/Hz')
    #ax.set_ylabel('Station pair distance/km')
    if flag_im:
        return im,ax
    else:
        return ax


def plot_ncfs_pairs(ax,f,ncfs0,r0,title0,flag_time,xlim,index):
    indx = np.argsort(r0)
    r  = r0[indx]
    ncfs = ncfs0[indx,:]

    for i in range(0,len(r0)):
        if np.max(ncfs[i,:]) == 0:
            continue
        ax.plot(f,np.real(ncfs[i,:])/np.max(np.real(ncfs[i,:]))*flag_time+r[i],'k',linewidth=0.5)
            
    ax.set_xlim(xlim)
    if index == 0:
        ax.set_title(title0)
    else:
        ax.set_title('('+chr(96+index)+')',loc='left')
    #ax.set_xlabel('Frequency/Hz')
    #ax.set_ylabel('Station pair distance/km')
    
    return ax

def plot_ncfs_pair_contrast(ax,f,ncfs,label1,ncfs_or,label2,r0,title0,flag_time,xlim,index):
    #fig,ax = plt.subplots(ncols=1,figsize=(10,4))
    flag = 0
    for i in range(0,len(r0)):
        if np.max(ncfs[i,:]) == 0:
            continue
        ax.plot(f,ncfs_or[i,:]/np.max(ncfs_or[i,:])*flag_time+r0[i],'r',linewidth=0.5)
        ax.plot(f,ncfs[i,:]/np.max(ncfs[i,:])*flag_time+r0[i],'k--',linewidth=0.5)
        if flag == 0:
            flag = 1
            ax.plot(f,ncfs_or[i,:]/np.max(ncfs_or[i,:])*flag_time+r0[i],'r',linewidth=0.5,label = label2)
            ax.plot(f,ncfs[i,:]/np.max(ncfs[i,:])*flag_time+r0[i],'k',linewidth=0.5,label = label1)
            
    ax.set_xlim(xlim)
    if index == 0:
        ax.set_title(title0)
    else:
        ax.set_title('('+chr(96+index)+')',loc='left')
    #ax.set_xlabel('Normalized frequency/Hz')
    #ax.set_ylabel('Station pair distance/km')
    ax.legend()
    return ax


# fj
def plot_fj(ax,ds_linear,title0,f,c,index,v_min=0.1,v_max=1,c_map='jet',flag_im = 0 ):
    im = ax.imshow(np.flip(ds_linear,0),aspect='auto',extent=[min(f),max(f),min(c),max(c)],vmin=v_min,vmax = v_max, cmap = c_map)
    #plt.imshow(np.flip(ds_linear,0),extent=[min(f),max(f),min(c),max(c)],aspect='auto',cmap='jet',vmin=0,vmax=1)
    #ax.pcolormesh(f,c,ds_linear,cmap='jet',vmin=0,vmax=1)
    if index == 0:
        ax.set_title(title0)
    else:
        ax.set_title('('+chr(96+index)+')',loc='left')
    #ax.set_xlabel('Normalized Frequency/ Hz')
    #ax.set_ylabel('Velocity/ m/s')
    #ax.set_xlim(xlim)
    if flag_im:
        return im, ax
    else:
        return ax

"""
def plot_fj_T(ax,ds_linear,title0,T,c,index):
    #ax.pcolormesh(T,c,ds_linear,cmap='jet',vmin=0,vmax=1)
    #ax.imshow(np.flip(ds_linear,0),aspect='auto',extent=[min(T),max(T),min(c),max(c)],vmin=0.05, c_map = 'jet')
    ax.imshow(np.flip(ds_linear,0),aspect='auto',extent=[min(f),max(f),min(c),max(c)],vmin=v_min,vmax = v_max, cmap = c_map)
    if index == 0:
        ax.set_title(title0)
    else:
        ax.set_title('('+chr(96+index)+')',loc='left')
    #ax.set_xlabel('Period / s')
    #ax.set_ylabel('Velocity/ m/s')
    #ax.set_xlim(xlim)
    return ax
"""

# plot stations
def plot_area(ax,lon_all,lat_all,lon,lat,markersize=2,markersize2 = 2):
    ax.plot(lon_all,lat_all,'k*',markersize=markersize)
    ax.plot(lon,lat,'r*',markersize=markersize2)
    ax.set_xticks([])  #去掉横坐标值
    ax.set_yticks([])  #去掉纵坐标值
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    #ax.set_title('Station distribution')
    return ax

# smooth for F-J spectrogram
def smooth_ds(ds,level_f=10,level_c=5):
    ds_smooth = ds.T
    ds_smooth = pd.DataFrame(ds_smooth)
    ds_smooth = ds_smooth.rolling(level_f).mean()
    ds_smooth = np.array(ds_smooth)
    ds_smooth = ds_smooth.T
    ds_smooth = pd.DataFrame(ds_smooth)
    ds_smooth = ds_smooth.rolling(level_c).mean()
    ds_smooth = np.array(ds_smooth)
    return ds_smooth