# Add incidents into map
import sys
sys.path.append(r'/home/songshuhao/anaconda3/bin/')

import numpy as np
import os
import shutil

# remake files
def file_remake(proj_name,info_num,name):
    filename = proj_name+'config/info_'+str(info_num)+'.npy'
    info = np.load(filename, allow_pickle='TRUE').item()      # setting dictionary
    if os.path.exists(info[name]):
        shutil.rmtree(info[name])
        os.mkdir(info[name])
    else:
        os.mkdir(info[name])

def file_remove(proj_name,info_num):
    filename = proj_name+'config/info_'+str(info_num)+'.npy'
    if os.path.exists(filename):
        info = np.load(filename, allow_pickle='TRUE').item()      # setting dictionary

        if os.path.exists(info['dir_correlation']):
            shutil.rmtree(info['dir_correlation'])
        if os.path.exists(info['dir_stack']):
            shutil.rmtree(info['dir_stack'])
        if os.path.exists(info['dir_image']):
            shutil.rmtree(info['dir_image'])
        if os.path.exists(info['dir_ds']):
            shutil.rmtree(info['dir_ds'])
        if os.path.exists(info['stalistname']):
            os.remove(info['stalistname'])
        if os.path.exists(filename):
            os.remove(filename)