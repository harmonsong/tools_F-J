# stack
import sys
sys.path.append(r'/home/songshuhao/anaconda3/bin/')
sys.path.append(r'../')

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
from statsmodels import robust
import yaml


from toollib_standard import mathlib
from toollib_standard import plotlib


# RMS stack
def rms(a):
    r = 0
    for i in range(len(a)):
        r+=a[i]*a[i]
    return np.sqrt(r/len(a))

def rmsr_ss(idd_this,data,sigwin,noisewin1,noisewin2):
    n = len(idd_this)
    idd_new_this = idd_this.copy()
    a = np.zeros(n)
    mean = np.mean(data,axis=0)
    rn = rms(mean[sigwin[0]:sigwin[1]])/rms(np.append(mean[noisewin1[0]:noisewin1[1]],mean[noisewin2[0]:noisewin2[1]]))
    #print(rn)
    flag_del = 0
    for i in range(n):
        l = list(range(n))
        del l[i]
        mean = np.mean(data[l,:],axis=0)
        #print(rms(mean[sigwin[0]:sigwin[1]]))
        #print(rms(np.append(mean[noisewin1[0]:noisewin1[1]],mean[noisewin2[0]:noisewin2[1]])))
        ri = rms(mean[sigwin[0]:sigwin[1]])/rms(np.append(mean[noisewin1[0]:noisewin1[1]],mean[noisewin2[0]:noisewin2[1]]))
        if (ri/rn) < 1:
            a[i] = 1
            #a[i] = ri/rn
        else:
            del idd_new_this[i-flag_del]
            flag_del += 1
    
    return a,idd_new_this


def rms_stack(proj_name,key_subwork,idd,ncfst_half,ncfs,k):

    start0 = time.time()
    #print('subarea ',key_subwork,' RMS stack')
    
    # read basic
    filename = proj_name+'Basic_info.yml'
    with open(filename, 'r', encoding='utf-8') as f:
        info_basic = yaml.load(f.read(), Loader=yaml.FullLoader)
    filename_bi = proj_name+'Basic_info.npy'
    info_basic_bi = np.load(filename_bi, allow_pickle='TRUE').item()      # setting dictionary

    d_len = info_basic['d_len']

    stalistname = info_basic['stalistname']
        # output dirs
    dir_stack = info_basic['dir_stack']
    dir_image  = info_basic['dir_image']
        # paras
    nsta = info_basic['nstaS'][key_subwork]
    nPairs = int(nsta*(nsta-1)/2)
    nf = info_basic['nf']
    f = info_basic_bi['f']

    v_min = info_basic_bi['v_min_period'][key_subwork][k]
    v_max = info_basic_bi['v_max_period'][key_subwork][k]
    sigwins = info_basic_bi['sigwins_period'][key_subwork][k]
    noisewins1 = info_basic_bi['noisewins1_period'][key_subwork][k]
    noisewins2 = info_basic_bi['noisewins2_period'][key_subwork][k]

    # RMS stack
    
    #count ak
    ak = np.zeros([nPairs,d_len])
    ak1 = np.ones([nPairs,d_len])   
    idd_new = []

    count_all = np.zeros(nPairs)

    for i in range(nPairs):
        sigwin = list(sigwins[i])
        noisewin1 = list(noisewins1[i])
        noisewin2 = list(noisewins2[i])
        idd_new_this = []
        if np.shape(ncfst_half)[0] > 1:
            if idd[i] == []:
                idd_new.append([])
                continue
            ak[i,idd[i]],idd_new_this = rmsr_ss(idd[i],ncfst_half[idd[i],i,:],sigwin,noisewin1,noisewin2)
            idd_new.append(idd_new_this)
            #ak[i,idd[i]] = rmsr_ss(len(idd[i]),ncfst_half[idd[i],i,:],sigwin,noisewin1,noisewin2)

    #final time domain stack
    ncfs1 = np.zeros([np.size(ncfs,1),np.size(ncfs,2)],dtype=np.complex64)
    ncfs1_rm = np.zeros([np.size(ncfs,1),np.size(ncfs,2)],dtype=np.complex64)
    ncfs_sum_rms = np.zeros([np.size(ncfs,1),np.size(ncfs,2)],dtype=np.complex64)
    ncfs_sum_rms_rm = np.zeros([np.size(ncfs,1),np.size(ncfs,2)],dtype=np.complex64)
    count = np.zeros(nPairs)
    for pair in range(nPairs):
        for day in range(d_len):
            ncfs1[pair,:] += ak[pair,day]*ncfs[day,pair,:] 
            ncfs1_rm[pair,:] += (ak1[pair,day]-ak[pair,day])*ncfs[day,pair,:]
            if ak[pair,day] == 1:
                count[pair] += 1
    for pair in range(nPairs):
        if count[pair]==0:
            continue
        count_all[pair] = 1
        ncfs_sum_rms[pair] = ncfs1[pair]/count[pair]
        #ncfs_sum_rms_rm[pair] = ncfs1_rm[pair]/(d_len-count[pair])
    # save
    #outname = title + ".npz"
    #if os.path.exists(dir_stack+outname):
    #    os.remove(dir_stack+outname)
    #np.savez(dir_stack+outname,ncfs= ncfs_sum_rms,ncfs_rm = ncfs_sum_rms_rm,idd_new=idd_new)
    #print('time:', time.time()-start0, ' seconds')
    
    #ak_rm = ak1-ak

    #return ak,ak_rm,idd_new

    return ncfs_sum_rms,count_all

# PWS stack
def PWS(ncfst0, v):
    m = len(ncfst0)
    n = np.size(ncfst0[1])

    c = np.zeros(n, dtype=complex)
    for i, tr in enumerate(ncfst0):
        h = hilbert(tr)
        c += np.nan_to_num(h/abs(h),nan=1)
    c = abs(c/m)
    
    s = np.zeros(n)
    for i in range(m):
        s += ncfst0[i,:]*c**v
    s = s/m

    return s

def PWS_stack(proj_name,info_num,v,title0,idd,ncfs,r):
    
    start0 = time.time()
    print('subarea ',str(info_num),' PWS stack')
    
    # read basic
    filename = proj_name+'Basic_info.yml'
    with open(filename, 'r', encoding='utf-8') as f:
        info_basic = yaml.load(f.read(), Loader=yaml.FullLoader)
    filename_bi = proj_name+'Basic_info.npy'
    info_basic_bi = np.load(filename_bi, allow_pickle='TRUE').item()      # setting dictionary

    # Set Params
    nf = info_basic['nf']
    nsta = info_basic['nstaS'][info_num]
    nPairs = int(nsta*(nsta-1)/2)
        # output dirs
    dir_stack = info_basic['dir_stack']
    
    # PWS stack
    ncfst1 = np.zeros([nPairs,int((nf-1)*2)])
    for i in range(nPairs):
        if len(idd[i])>1:
            ncfst = mathlib.freq_time(ncfs[idd[i],i,:])
            ncfst1[i,:] = PWS(ncfst,v)
    ncfs_pws = mathlib.time_freq(ncfst1)
    outname = title0+".npz"
    
    if os.path.exists(dir_stack+outname):
        os.remove(dir_stack+outname)
    
    np.savez(dir_stack+outname,ncfs = ncfs_pws,r=r)
    print('time:', time.time()-start0, ' seconds')


# frequency stack
def freq(nr1,nf,ncfs,idd,amp):
    std = np.zeros([nr1,nf],dtype = complex)
    mean = np.zeros([nr1,nf],dtype = complex)

    for i in range(nr1):
        if len(idd[i]) > 0:
            mean[i,:] = np.mean(ncfs[idd[i],i,:],axis = 0)
            std[i,:] = np.std(ncfs[idd[i],i,:],axis = 0)
    s1 = np.zeros_like(mean)
    for j in range(nr1):
        for k in range(nf):
            s1[j,k] = np.mean(ncfs[np.abs(ncfs[:,j,k]-mean[j,k]) <= amp*std[j,k],j,k])
    return s1

def freq_stack(proj_name,info_num,ncfs,idd,r,ncfs_sum_linear,amp,title):
    
    start0 = time.time()
    print('subarea ',str(info_num),' frequency stack')
    
    # read basic
    filename = proj_name+'Basic_info.yml'
    with open(filename, 'r', encoding='utf-8') as f:
        info_basic = yaml.load(f.read(), Loader=yaml.FullLoader)
    filename_bi = proj_name+'Basic_info.npy'
    info_basic_bi = np.load(filename_bi, allow_pickle='TRUE').item()      # setting dictionary  

    dir_stack = info_basic['dir_stack']
    d_len = info_basic['d_len']
    nf = info_basic['nf']

    nsta = info_basic['nstaS'][info_num]
    nPairs = int(nsta*(nsta-1)/2)
    
    # frequency stack
    s0 = freq(nPairs,nf,ncfs,idd,amp)
    s1 = np.nan_to_num(s0)
    outname = title + ".npz"
    
    if os.path.exists(dir_stack+outname):
        os.remove(dir_stack+outname)
    
    np.savez(dir_stack+outname,ncfs = s1,r=r)
    print('time:', time.time()-start0, ' seconds')

    # test part

# frequency stack -MAD
def freq_MAD(nr1,nf,ncfs,idd,amp):
    mad = np.zeros([nr1,nf],dtype = complex)
    mean = np.zeros([nr1,nf],dtype = complex)

    for i in range(nr1):
        if len(idd[i]) > 0:
            mean[i,:] = np.mean(ncfs[idd[i],i,:],axis = 0)
            mad[i,:] = robust.mad(ncfs[idd[i],i,:],axis = 0)
    s1 = np.zeros_like(mean)
    for j in range(nr1):
        for k in range(nf):
            s1[j,k] = np.mean(ncfs[np.abs(ncfs[:,j,k]-mean[j,k]) <= amp*mad[j,k],j,k])
    return s1

def freq_stack_MAD(proj_name,info_num,ncfs,idd,r,ncfs_sum_linear,amp,title):
    
    start0 = time.time()
    print('subarea ',str(info_num),' frequency stack -MAD')
    
    # read basic
    filename = proj_name+'Basic_info.yml'
    with open(filename, 'r', encoding='utf-8') as f:
        info_basic = yaml.load(f.read(), Loader=yaml.FullLoader)
    filename_bi = proj_name+'Basic_info.npy'
    info_basic_bi = np.load(filename_bi, allow_pickle='TRUE').item()      # setting dictionary

    dir_stack = info_basic['dir_stack']
    d_len = info_basic['d_len']
    nf = info_basic['nf']

    nsta = info_basic['nstaS'][info_num]
    nPairs = int(nsta*(nsta-1)/2)
    
    # frequency stack
    s0 = freq_MAD(nPairs,nf,ncfs,idd,amp)
    s1 = np.nan_to_num(s0)
    outname = title + ".npz"
    
    if os.path.exists(dir_stack+outname):
        os.remove(dir_stack+outname)
    
    np.savez(dir_stack+outname,ncfs = s1,r=r)
    print('time:', time.time()-start0, ' seconds')


def PWS_stack_period(proj_name,info_num,v,title0,idd,ncfs,T_bands,r):
    
    start0 = time.time()
    print('subarea ',str(info_num),' PWS-period stack, v=',str(v))
    
    # read basic
    filename = proj_name+'Basic_info.yml'
    with open(filename, 'r', encoding='utf-8') as f:
        info_basic = yaml.load(f.read(), Loader=yaml.FullLoader)
    filename_bi = proj_name+'Basic_info.npy'
    info_basic_bi = np.load(filename_bi, allow_pickle='TRUE').item()      # setting dictionary

    # Set Params
    nf = info_basic['nf']
    nsta = info_basic['nstaS'][info_num]
    nPairs = int(nsta*(nsta-1)/2)
    N_band = int(len(T_bands)/2)
    f = info_basic_bi['f']
    T = 1/f
        # output dirs
    dir_stack = info_basic['dir_stack']
    
    # PWS stack
    ncfs_sub = {}
    for i in range(N_band):
        T_band = T_bands[2*i:2*i+2]
        ncfs_sub[i] = np.zeros(np.shape(ncfs),dtype = complex)
        ncfs_sub[i][:,:,np.where((T>=T_band[0])&(T<=T_band[1]))] = ncfs[:,:,np.where((T>=T_band[0])&(T<=T_band[1]))]
    #ncfs_sub['others'] = np.zeros(np.shape(ncfs),dtype = complex)
    #ncfs_sub['others'][:,:,np.where((T< T_bands[0])|(T > T_bands[2*N_band-1]))] = ncfs[:,:,np.where((T< T_bands[0])|(T > T_bands[2*N_band-1]))]/10

    ncfst_pws = np.zeros([nPairs,int((nf-1)*2)])
    for k in range(N_band):
        ncfst1 = np.zeros([nPairs,int((nf-1)*2)])
        for i in range(nPairs):
            if len(idd[i])>1:
                ncfst = mathlib.freq_time(ncfs_sub[k][idd[i],i,:])
                ncfst1[i,:] = PWS(ncfst,v)
        ncfst_pws = ncfst_pws + ncfst1
    ncfst_pws = ncfst_pws/N_band

    ncfs_pws = mathlib.time_freq(ncfst_pws)
    outname = title0+".npz"
    
    if os.path.exists(dir_stack+outname):
        os.remove(dir_stack+outname)
    
    np.savez(dir_stack+outname,ncfs = ncfs_pws,r=r)
    print('time:', time.time()-start0, ' seconds')