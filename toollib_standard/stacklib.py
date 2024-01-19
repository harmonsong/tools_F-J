# stack
import sys
sys.path.append(r'/home/songshuhao/anaconda3/bin/')

# use the .py script will be faster
import numpy as np
import obspy
from obspy import UTCDateTime
from ccfj import CC
from ccfj import GetStationPairs
from concurrent.futures import ThreadPoolExecutor
import os
import time
from geopy.distance import great_circle
from scipy.signal import hilbert
import matplotlib.pyplot as plt


from toollib_standard import mathlib
from toollib_standard import plotlib


# read ccfs
def Pairs(sta):
    p = []
    nsta = len(sta)
    for ii in range(nsta):
        for jj in range(ii+1,nsta):
            p.append([sta[ii],sta[jj]])
    return p

def cal_indx(pair,nsta):
    pair0 = min(pair)
    pair1 = max(pair)
    indx = int(pair0*(2*nsta-pair0-1)/2+pair1-pair0-1)
    return indx
    

def readccf(proj_name,key_subwork):
    # read basic
    filename_basic = proj_name+'Basic_info.npy'
    info_basic = np.load(filename_basic, allow_pickle='TRUE').item()      # setting dictionary
    filename_subworks = proj_name+'sta_subworks.npy'
    sta_subworks = np.load(filename_subworks, allow_pickle='TRUE').item()      # setting dictionary

    y_start = info_basic['year'][key_subwork]
    y_end = y_start + 1
    d_start = 1
    d_end = 366
    #y_end = info_basic['y_end']
    #d_start = info_basic['d_start']
    #d_end = info_basic['d_end']
    stalistname = info_basic['stalistname']
    nf = info_basic['nf']
    f = info_basic['f']
    dir_stack= info_basic['dir_stack']
    d_len = info_basic['d_len']

    dir_CC = info_basic['dir_CC']

    # read all data
    f = info_basic['f']
    

    # read subwork data
    nsta = info_basic['nstaS'][key_subwork]
    nPairs = int(nsta*(nsta-1)/2)
    stalistname = proj_name+info_basic['stalistname'] +'-'+key_subwork
    
    stalist = list(sta_subworks[key_subwork].keys())
    ncfs = np.zeros([d_len,nPairs,nf],dtype = np.complex64)  
    ncfs_indx = []

    #read ncfs
    ## Read CCFs day by day
    id1s = []
    flag_day = -1
    StationPairs = GetStationPairs(nsta)
    for y in range(y_start,y_end):
        for d in range(d_start,d_end):
            year = str(y)
            day = "%03d"%d
            outname = os.path.join(dir_CC,year+'-'+day+'.npz')
            if os.path.exists(outname):
                #print(outname)
                data = np.load(outname)
                ncfs_all = data['ncfs']
                stalist_all = list(data['stalist'])
                nsta_all = len(stalist_all)
                StationPairs_all = GetStationPairs(nsta_all)
                nPairs_all = int(nsta_all*(nsta_all-1)/2)
                
                id1 = []
                for i in range(nPairs):
                    sta1 = StationPairs[2*i]
                    sta2 = StationPairs[2*i+1]
                    if not stalist[sta1] in stalist_all:
                        continue
                    if not stalist[sta2] in stalist_all:
                        continue
                    #idx1 = int(stalist_all.index(stalist[sta1]))
                    #idx2 = int(stalist_all.index(stalist[sta2]))
                    idx1 = np.min( [int(stalist_all.index(stalist[sta1])),int(stalist_all.index(stalist[sta2]))] )
                    idx2 = np.max( [int(stalist_all.index(stalist[sta1])),int(stalist_all.index(stalist[sta2]))] )
                    m = 0
                    for j in range(nsta_all-idx1,nsta_all):
                        m += j
                    num = m + idx2 - idx1 -1
                    ncfs[flag_day,i,:] = ncfs_all[num,:]
                    id1.append(i)
                if len(id1) > 0:
                    flag_day += 1
                    id1s.append(id1)

            if flag_day == d_len-1:
                idd = []
                for j in range(nPairs):
                    idd1 = []
                    for i in range(d_len):
                        #print(i)
                        if j in id1s[i]:
                            idd1.append(i)
                    idd.append(idd1)
                    #按每一个台站，存有这个台站的天数
                return ncfs,id1s,idd

    idd = []
    for j in range(nPairs):
        idd1 = []
        for i in range(d_len):
            #print(i)
            if j in id1s[i]:
                idd1.append(i)
        idd.append(idd1)
        #按每一个台站，存有这个台站的天数
    return ncfs,id1s,idd


# linear stack
def linear_stack(proj_name,key_subwork,id1s,ncfs,title):
    
    start0 = time.time()
    print('subarea ',key_subwork,' linear stack')
    
    # read file
    filename = proj_name+'Basic_info.npy'
    info_basic = np.load(filename, allow_pickle='TRUE').item()
    d_len = info_basic['d_len']
    stalistname = info_basic['stalistname']
        # output dirs
    dir_stack = info_basic['dir_stack']
    dir_image  = info_basic['dir_image']
        # paras
    nsta = info_basic['nstaS'][key_subwork]
    StationPairs = GetStationPairs(nsta)
    nPairs = int(nsta*(nsta-1)/2)
    nf = info_basic['nf']
    

    ncfs1 = np.zeros([nPairs,nf],dtype=np.complex64)
    count = np.zeros(nPairs)

    ## Read CCFs day by day
    for i in range(len(id1s)):
        id1 = id1s[i]
        ncfs1[id1,:] = ncfs1[id1,:] + ncfs[i,id1,:]
        count[id1] = count[id1] + 1

    ncfs_sum_linear = np.zeros(np.shape(ncfs1),dtype=np.complex64)
    for i in range(nPairs):
        if count[i]>0:
            ncfs_sum_linear[i,:] = ncfs1[i,:]/count[i] 
    
    # distances
    stalist = []
    lon = []
    lat =[]
    with open(stalistname,'r') as f:
        while True:
            tmp = f.readline()
            if tmp:
                stalist.append(tmp.split()[0])
                lon.append(float(tmp.split()[1]))
                lat.append(float(tmp.split()[2]))
            else:
                break

    r = np.zeros(nPairs)
    for i in range(len(r)):
        r[i] = great_circle((lat[StationPairs[i*2]],lon[StationPairs[i*2]]),(lat[StationPairs[i*2+1]],lon[StationPairs[i*2+1]])).km
    info_basic['r'] = r
    np.save(filename,info_basic)
    
    outname = title + ".npz"
    if os.path.exists(dir_stack+outname):
        os.remove(dir_stack+outname)
    np.savez(dir_stack+outname,ncfs= ncfs_sum_linear,r = r, StationPairs = StationPairs, stalist = stalist)
    
    #np.savez(dir_stack+"summed_linear.npz",ncfs= ncfs_sum_linear,r = r, StationPairs = StationPairs, stalist = stalist)
    print('time:', time.time()-start0, ' seconds')

    
    return r,ncfs_sum_linear



def linear_stack1(proj_name,info_num):
    
    start0 = time.time()
    print('subarea ',str(info_num),' linear stack')
    
    # read file
    filename = proj_name+'Basic_info.npy'
    info_basic = np.load(filename, allow_pickle='TRUE').item()
    
    y_start = info_basic['y_start']
    y_end = info_basic['y_end']
    d_start = info_basic['d_start']
    d_end = info_basic['d_end']
    d_len = info_basic['d_len']

    
    # set SAC data dir ,correspoding stalistname, and output dir
        # stalistname
    stalistname = info_basic['stalistname']
        # dirs
    outdir = info_basic['dir_correlation']
    dir_stack = info_basic['dir_stack']
    
    # Set Params
    nf = info_basic['nf']
    f = info_basic['f']
    nsta = info_basic['nstaS'][info_num]
    nPairs = int(nsta*(nsta-1)/2)
    StationPairs = GetStationPairs(nsta)
    
    # linear stack
    stalist = []
    lon = []
    lat =[]
    with open(stalistname,'r') as f:
        while True:
            tmp = f.readline()
            if tmp:
                stalist.append(tmp.split()[0])
                lon.append(float(tmp.split()[1]))
                lat.append(float(tmp.split()[2]))
            else:
                break

    ncfs1 = np.zeros([nPairs,nf],dtype=np.complex64)
    count = np.zeros(nPairs)

    ## Read CCFs day by day
    for y in range(y_start,y_end):
        for d in range(d_start,d_end):
            year = str(y)
            day = "%03d"%d
            outname = os.path.join(outdir,year+'-'+day+'.npz')
            if os.path.exists(outname):
                #print(year+'-'+day)
                data = []
                data = np.load(outname)
                #nsta0 = len(data["stalist"])
                indx = [stalist.index(i) for i in data["stalist"] ]
                pairs = Pairs(indx)
                id1 = [cal_indx(pair,nsta) for pair in pairs]

                ncfs1[id1,:] = ncfs1[id1,:]+data["ncfs"]
                count[id1] = count[id1]+1

    ncfs_sum_linear = np.zeros(np.shape(ncfs1),dtype=np.complex64)
    for i in range(nPairs):
        if count[i]>0:
            ncfs_sum_linear[i,:] = ncfs1[i,:]/count[i] 

    r = np.zeros(nPairs)
    for i in range(len(r)):
        r[i] = great_circle((lat[StationPairs[i*2]],lon[StationPairs[i*2]]),(lat[StationPairs[i*2+1]],lon[StationPairs[i*2+1]])).km
    info_basic['r'] = r
    np.save(filename,info)
    
    if os.path.exists(dir_stack+"summed_linear.npz"):
        os.remove(dir_stack+"summed_linear.npz")
    
    np.savez(dir_stack+"summed_linear.npz",ncfs= ncfs_sum_linear,r = r, StationPairs = StationPairs, stalist = stalist)
    print('time:', time.time()-start0, ' seconds')

    
    return r,ncfs_sum_linear,count


def cal_wins(ncfst_half,r,t,f,v_min,v_max,tao,nPairs,flag_signal):
    sigwins = np.zeros([nPairs,2],dtype=int)
    noisewins1 = np.zeros([nPairs,2],dtype=int)
    noisewins2 = np.zeros([nPairs,2],dtype=int)
    for i in range(nPairs):
        t_ref = r[i] / v_min
        if flag_signal == 0:
            t_min = r[i] / v_max
            t_max = r[i] / v_min
        elif flag_signal == 1:
            flag_max = np.argmax(ncfst_half[i])
            if np.abs(t[flag_max]-t_ref) > 2*tao:
                t_min = r[i]/v_min
            else:
                t_max = t[flag_max]
                t_min = t_max
        index_sig_left = np.argmin(np.abs(t-t_min+tao))
        index_sig_right = np.argmin(np.abs(t-t_max-tao))
        sigwin = [index_sig_left,index_sig_right]
        noisewin1 = [0,index_sig_left]
        noisewin2 = [-int(2*tao*np.max(f)),-1]
        sigwins[i,:] = sigwin
        noisewins1[i,:] = noisewin1
        noisewins2[i,:] = noisewin2
    return sigwins, noisewins1, noisewins2