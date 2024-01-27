
#_____  _                     _   _   _      _     _                     _ 
#|  __ \(_)                   | | | \ | |    | |   | |                   | |
#| |  | |_ ___ _ __   ___ _ __| |_|  \| | ___| |_  | |     ___   ___ __ _| |
#| |  | | / __| '_ \ / _ \ '__| __| . ` |/ _ \ __| | |    / _ \ / __/ _` | |
#| |__| | \__ \ |_) |  __/ |  | |_| |\  |  __/ |_  | |___| (_) | (_| (_| | |
#|_____/|_|___/ .__/ \___|_|   \__|_| \_|\___|\__| |______\___/ \___\__,_|_|
#            | |                                                           
#            |_|                                                                  
#   DisperNet Local v2.00 beta 
#	Released on 20 Sep. 2022 for offline usage
#   powered by Dongsh (Dongsh@mail.ustc.edu.cn) from ESS.USTC

import requests
import numpy as np
import sklearn.cluster as sc
from scipy.cluster.vq import whiten
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
import matplotlib
import h5py
import os
import re
import scipy.ndimage as sn
import PIL
import random
from matplotlib.widgets import Cursor
import pandas as pd
import extractor
import yaml

import sys
sys.path.append(r'../tools_F-J/')
from toollib_standard import plotlib

np.set_printoptions(suppress=True)

def pick(spec, threshold=0.5, freq=[0.,0.3], velo=[2000, 6000], net='noise', errorbar=False, flipUp=False, searchStep=10, searchBorder=0, returnSpec=False, ind=-1, url='useless'):
	
	spec[np.isnan(spec)] = 0
	output = extractor.pickPoint(spec, threshold, freq, velo, searchStep, returnSpec)
	return output

def modeSeparation(curves, modes=2):

	if np.array(curves).ndim < 2:
		return []			
	if len(curves) == 0:
		return []
	
	curve_whiten = whiten(curves[:,0:2])      
	cluster_pred = sc.AgglomerativeClustering(n_clusters=int(modes),linkage='single',compute_full_tree=True).fit_predict(curve_whiten)
	
	
	vRange = np.max(curves[:, 1]) - np.min(curves[:, 1])
	fRange = np.max(curves[:, 0]) - np.min(curves[:, 0])
	
	m_value = np.zeros(modes)
	
	for mode in range(modes):
		m_value[mode] = np.mean(curves[cluster_pred==mode, 0]/fRange)**2 + np.mean(curves[cluster_pred==mode, 1]/vRange)**2
	
	m_c = np.vstack([m_value, np.arange(modes)])

	m_c = m_c[:, m_c[0,:].argsort()]

	cluster_out = cluster_pred.copy()
	for mode in range(modes):

		cluster_out[cluster_pred==m_c[1,mode]] = mode
	
	if curves.shape[1] == 2 or curves.shape[1] == 4:
		out = np.column_stack([curves, cluster_out])
	
	else:
		out = np.column_stack([curves[:, 0:-1], cluster_out])
	
	out = out[np.argsort(out[:,-1])]
	for mode in range(modes):
		curveInMode = out[out[:,-1] == mode]
		out[out[:,-1] == mode] = curveInMode[np.argsort(curveInMode[:,0])]
	
	return out
	
def sortCurve(curve):
	modes = int(max(curve[:,-1])) + 1
	out = curve[np.argsort(curve[:,-1])]
	for mode in range(modes):
		curveInMode = out[out[:,-1] == mode]
		out[out[:,-1] == mode] = curveInMode[np.argsort(curveInMode[:,0])]
	
	return out
	
def autoSeparation(curves, to=0.04, maxMode=5, veloBias=800):
	
	if np.array(curves).ndim < 2:
		return []			
	if len(curves) == 0:
		return []
	
	
	fMax = max(curves[:,0])
	fMin = min(curves[:,0])
	cMax = max(curves[:,1])
	cMin = min(curves[:,1])
	
	if cMax - cMin < veloBias:
		cMax += veloBias
	
#	fMax = 50
#	fMin = 0
#	cMax = 3800
#	cMin = 1500
	
	fSearchStart = 0.05 * (fMax - fMin) + fMin
	cJumpRangeLimit = to * (cMax - cMin)
	exitFlag = False
	for modePre in range(int(maxMode)):
		curvePre = modeSeparation(curves, int(modePre)+1)
		for mode in range(modePre+1):
			curveInMode = curvePre[curvePre[:,-1] == mode]
			curveInMode = curveInMode[curveInMode[:,0]> fSearchStart]
			curveInMode = curveInMode[np.argsort(curveInMode[:,0])]
			
			if np.std(np.diff(curveInMode[:,1])) > cJumpRangeLimit:
				exitFlag = False
				break
			else:
				exitFlag = True

		if exitFlag:
			break
	
	return curvePre
	
	
def leapDetect(curves, to=0.04):
	fMax = max(curves[:,0])
	fMin = min(curves[:,0])
	cMax = max(curves[:,1])
	cMin = min(curves[:,1])
	
	fSearchStart = 0.05 * (fMax - fMin) + fMin
	cJumpRangeLimit = to * (cMax - cMin)
	exitFlag = False
	
	for mode in range(int(np.max(curves[:,2]))+1):
		curveInMode = curves[curves[:,-1] == mode]
		curveInMode = curveInMode[curveInMode[:,0]> fSearchStart]
		curveInMode = curveInMode[np.argsort(curveInMode[:,0])]
		
		if np.std(np.diff(curveInMode[:,1])) > cJumpRangeLimit:
			return True

	return False
	
	
def discPointRemove(curve, threshold=2):
	curveRemoved = []
	for mode in range(int(max(curve[:,-1]))+1):
		curveInMode = curve[curve[:,-1]==mode]
		if len(curveInMode) > threshold:
			curveRemoved.append(curveInMode)
			
	curveRemoved = np.vstack(curveRemoved)
	return curveRemoved

def modeSeparationSuper(curve, freq, velo):
	return extractor.modeSuperSeparation(curve, freq, velo)

"""
def freq2Period(inputSpec, freq, maxPeriod=0.5, scale=18, freqCount=0, kind='linear', cutRate=0.125,max_freq = 10):
	if maxPeriod == 0:
		maxPeriod =  np.min([1 / np.min(freq[freq>0]), 130])
#		if maxPeriod
	minFreq = 1 / maxPeriod
	inputP = inputSpec.copy()
	veloRange = np.arange(0,inputP.shape[0],scale)
	inputP = inputP[:inputP.shape[0]:scale,freq>=minFreq]
	freqP = freq[freq>=minFreq]
	
	if freqCount==0:
		freqCount = int(inputSpec.shape[1]*cutRate)
		
	inputP = inputP[:,:freqCount]
	period = 1/np.array(freqP[:freqCount])
	
	pp, vv  = np.meshgrid(period, veloRange)
	try:
		itt = interpolate.interp2d(pp.flatten(),vv.flatten(),inputP.flatten(), kind=kind)
	except RuntimeWarning:
		print('[warning] unsuccssed interp in period.')
		
	newPeriod = np.linspace(min(period), max(period), freqCount)
	return itt(newPeriod, veloRange), newPeriod
"""

def freq2Period(inputSpec, freq, maxPeriod=0.8, scale=18, freqCount=0, kind='linear', cutRate=0.125,max_freq = 4):
	if maxPeriod == 0:
		maxPeriod =  np.min([1 / np.min(freq[freq>0]), 130])
#		if maxPeriod
	minFreq = 1 / maxPeriod
	inputP = inputSpec.copy()
	veloRange = np.arange(0,inputP.shape[0],scale)
	inputP = inputP[:inputP.shape[0]:scale,(freq>=minFreq) & (freq<=max_freq)]
	freqP = freq[(freq>=minFreq) & (freq<=max_freq)]
	
	if freqCount==0:
		freqCount = int(inputSpec.shape[1]*cutRate)
		
	inputP = inputP[:,:freqCount]
	period = 1/np.array(freqP[:freqCount])
	
	pp, vv  = np.meshgrid(period, veloRange)
	try:
		itt = interpolate.interp2d(pp.flatten(),vv.flatten(),inputP.flatten(), kind=kind)
	except RuntimeWarning:
		print('[warning] unsuccssed interp in period.')
		
	newPeriod = np.linspace(min(period), max(period), freqCount)
	return itt(newPeriod, veloRange), newPeriod


def show(spec,curve,r_this = 0,freq=[0.,0.3], velo=[2000, 6000], unit=[], s=10, ax=[], holdon=False, cmap='viridis', vmin=None, vmax=None, xLabel='Frequency (Hz)', autoT=True):
	fMax = max(freq)
	fMin = min(freq)
	cMax = max(velo)
	cMin = min(velo)
	
	if ax == []:
		ax = plt.gca()
		
	if fMax == fMin:
		raise ValueError('freq must be a range or array')
		
	if cMax == cMin:
		raise ValueError('velo must be a range or array')
		
	if autoT and (spec.shape == (len(freq), len(velo))):
		print('[info] Auto Transposed, if you do not need this feature, set autoT=True when plotting or applicaion launching.')
		ax.imshow(np.flip(spec.T ,0),aspect='auto', extent=[fMin, fMax, cMin, cMax], cmap=cmap, vmin=vmin, vmax=vmax)
	else:	
		ax.imshow(np.flip(spec,0),aspect='auto', extent=[fMin, fMax, cMin, cMax], cmap=cmap, vmin=vmin, vmax=vmax)
		
		
	if len(unit) == 0:
		if cMax / 1e2 > 2:
			unit = 'm'
		else:
			unit = 'km'
			
	#ax.set_xlabel(xLabel)
	ax.set_ylabel('Phase Velocity ('+unit+')')
	
	markerList=['*','o','v','^','s','p','d','<','>', '1', '$10$', '$11$','$12$', '$13$', '$14$', '$15$', '$16$']
	
	if len(curve)>0:
		if curve.shape[1] == 2 or curve.shape[1] == 4:
			ax.scatter(curve[...,0],curve[...,1],s=s, edgecolors='w')
		else:
			for ii in range(int(max(curve[...,-1]))+1):
				curve_in_mode = curve[curve[:,-1] == ii]
				ax.scatter(curve_in_mode[...,0], curve_in_mode[...,1],label='mode '+str(ii),s=s,edgecolors='k',marker=markerList[ii])
				
			ax.legend()
	

	if r_this != 0:
		cc = np.linspace(cMin, cMax, 100)
		ff = cc/r_this*2
		ax.plot(ff,cc,'k--')
	if not holdon:
		plt.show()

def show_partrition(self):
		info_basic = self.info_basic
		faults = self.faults
		lat_all = self.lat_all
		lon_all = self.lon_all

		lat = []
		lon = []

		key_subwork = self.fileName[self.fileName.find('_')+1:self.fileName.find('.')]
		if '_' in key_subwork:
			key_subwork = key_subwork[:key_subwork.find('_')]

		sta_info_all = self.sta_info_all

		stalist = sta_info_all['stations'][key_subwork]
		lat = sta_info_all['lat'][key_subwork]
		lon = sta_info_all['lon'][key_subwork]

		self.ax_partrition.cla()
		self.ax_partrition =plotlib.plot_area(self.ax_partrition,lon_all,lat_all,lon,lat,markersize = 1 ,markersize2 = 4)
		if faults is not None:
			for i in range(len(faults)):
				self.ax_partrition.plot(faults['clark'+str(i+1)]['lon'], faults['clark'+str(i+1)]['lat'], 'k')

		lon_this = np.mean(lon)
		lat_this = np.mean(lat)
		self.ax_partrition.scatter(lon_this,lat_this,marker='^',color='g',s=40)

		self.ax_partrition.axis('off')
		plt.draw()
		plt.show()
		
def save2h5(spectrum, freq, velo, fileName='',spectrum_or = []):
	if fileName == '':
		fileName = 'demoSpectra.h5'
	
	if fileName[-3:] != '.h5':
		fileName = fileName + '.h5'
		print('[Warning] the filename was changed to \'' + fileName + '\'')

	with h5py.File(fileName, 'w') as fw:
		fw.create_dataset('f', data=freq)
		fw.create_dataset('c', data=velo)
		fw.create_dataset('amp', data=spectrum)
		if spectrum_or is not []:
			fw.create_dataset('amp_or', data=spectrum_or)
		
		
def readh5(fileName, key='amp'):
	with h5py.File(fileName, 'r') as fr:
		freq = np.array(fr['f'])
		velo = np.array(fr['c'])
		if 'ds' in fr.keys():
			spec = np.array(fr['ds'])
		else:
			spec = np.array(fr[key])
	
	return spec, freq, velo
		
def createTrainSet(setFileName='',filePath='./', curveFilePath = '', widthList=[], randomSelectRate=0):
	
	if filePath[:-1] != '/':
		filePath = filePath + '/'
	
	if curveFilePath == '':
		curveFilePath = filePath
	
	if curveFilePath[:-1] != '/':
		curveFilePath = curveFilePath + '/'
			
	lineWidthList = np.ones(16)*3
	if widthList != []:
		if len(widthList) <= 7:
			lineWidthList[:len(widthList)] = widthList

	def get_file_list(basis_dir="./", begin="", end=""):
		path_list = os.listdir(basis_dir)
		list_final = []
		for partial in path_list:
			if begin and end:
				if partial[:len(begin)] == begin and partial[-len(end):] == end:
					list_final.append(partial)
					
			elif end:
				if partial[-len(end):] == end:
					list_final.append(partial)
			
			elif begin:
				if partial[:len(begin)] == begin:
					list_final.append(partial)
					
			else:
				list_final.append(partial)
				
		return list_final
	
	def natural_sort(l): 
		convert = lambda text: int(text) if text.isdigit() else text.lower() 
		alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
		return sorted(l, key = alphanum_key)
		
	specList = get_file_list(filePath, end='.h5')
	specList = natural_sort(specList)
	setLength = len(specList )
	print('Found ' + str(setLength) + ' *.h5 File in ' + filePath)
	
	if setLength > 0:
		imageSet = []
		maskSet = []
		
		for specFileName in specList:
			
			if randomSelectRate > 0:
				if random.random() > randomSelectRate:
					continue
				
#			print(specFileName)
			curvePath = curveFilePath + specFileName[:-3] + 'curve.txt'
			specFilePath = filePath + specFileName
			
			if not os.path.exists(curvePath):
				print('curve file of \'' + specFileName +'\' is no found, pass' )
				continue
				
#			print(specFileName)
			
			with h5py.File(specFilePath, 'r') as fr:
				freq = np.array(fr['f'])
				velo = np.array(fr['c'])
				if 'ds' in fr.keys():
					spec = np.array(fr['ds'])
				else:
					spec = np.array(fr['amp'])
				
			fMax = max(freq)
			fMin = min(freq)
			cMax = max(velo)
			cMin = min(velo)
			
			
			plt.figure(figsize=[5,5])
			
			try:
				curve_arti = np.loadtxt(curvePath, delimiter='  ')
			except ValueError:
				curve_arti = np.loadtxt(curvePath, delimiter=' ')
			
			for ii in range(int(max(curve_arti[:,2]))+1):
				curve_in_mode_arti = curve_arti[curve_arti[:,2] == ii]
				if ii == 0:
					plt.plot(curve_in_mode_arti[:,0],curve_in_mode_arti[:,1], linewidth=lineWidthList[0], c='Black')
				else:
					plt.plot(curve_in_mode_arti[:,0],curve_in_mode_arti[:,1], linewidth=lineWidthList[ii], c='Black')
				
			axes = plt.gca()
			axes.set_xlim([fMin, fMax])
			axes.set_ylim([cMin, cMax])
			png_name = 'test' + '.png'
			plt.axis('off')
			plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
			plt.savefig(png_name, dpi=200, bbox_inches='tight', pad_inches=0)
			plt.close()

			image = np.sum(matplotlib.image.imread(png_name),2)
			image = 1-(image-1)/3
			image = PIL.Image.fromarray(image*255).convert('L')
			image = image.resize((512, 512))
			image = np.asarray(image)/255
			
			os.remove(png_name)
				
			spec = np.flip(spec, 0)
			spec[np.isnan(spec)] = 0
			
			imageSet.append(spec)
			maskSet.append(image)
			
		imageSet = np.array(imageSet)
		maskSet = np.array(maskSet)
			
		if len(maskSet) > 0:
			if setFileName == '':
				setFileName = 'trainSet' + str(np.random.rand()) + '.h5'
			
			print('save training set in \''+ setFileName + '\' with size:')
			print(imageSet.shape)
			print(maskSet.shape)
			
			with h5py.File(setFileName, 'w') as new_file:
				new_file.create_dataset('image',data=imageSet)
				new_file.create_dataset('mask',data=maskSet)
		else:
			print('NO training set CREATED because curve file is no found.')

def get_file_list(basis_dir="./", begin="", end=""):
	path_list = os.listdir(basis_dir)
	list_final = []
	for partial in path_list:
		if begin and end:
			if partial[:len(begin)] == begin and partial[-len(end):] == end:
				list_final.append(partial)
				
		elif end:
			if partial[-len(end):] == end:
				list_final.append(partial)
		
		elif begin:
			if partial[:len(begin)] == begin:
				list_final.append(partial)
				
		else:
			list_final.append(partial)
			
	return list_final
	
def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)	
			
def createTrainSetPlus(setFileName='',filePath='./', curveFilePath = '', widthList=[], randomSelectRate=0):
	
	if filePath[:-1] != '/':
		filePath = filePath + '/'
	
	if curveFilePath == '':
		curveFilePath = filePath
	
	if curveFilePath[:-1] != '/':
		curveFilePath = curveFilePath + '/'
			
	lineWidthList = np.ones(16)*3
	if widthList != []:
		if len(widthList) <= 7:
			lineWidthList[:len(widthList)] = widthList

		
	specList = get_file_list(filePath, end='.h5')
	specList = natural_sort(specList)
	setLength = len(specList )
	print('Found ' + str(setLength) + ' *.h5 File in ' + filePath)
	
	if setLength > 0:
		imageSet = []
		maskSet = []
		maskModeSet = []
		
		
		
		for specFileName in specList:
			
			if randomSelectRate > 0:
				if random.random() > randomSelectRate:
					continue
				
			print(specFileName)
			curvePath = curveFilePath + specFileName[:-3] + 'curve.txt'
			specFilePath = filePath + specFileName
			
			if not os.path.exists(curvePath):
				print('curve file of \'' + specFileName +'\' is no found, pass' )
				continue
				
#			print(specFileName)
			
			with h5py.File(specFilePath, 'r') as fr:
				freq = np.array(fr['f'])
				velo = np.array(fr['c'])
				if 'ds' in fr.keys():
					spec = np.array(fr['ds'])
				else:
					spec = np.array(fr['amp'])
				
			fMax = max(freq)
			fMin = min(freq)
			cMax = max(velo)
			cMin = min(velo)
			
			modeImage = np.zeros([512,512])
			plt.figure(figsize=[5,5])
			
			try:
				curve_arti = np.loadtxt(curvePath, delimiter='  ')
			except ValueError:
				curve_arti = np.loadtxt(curvePath, delimiter=' ')
			
			for ii in range(int(max(curve_arti[:,2]))+1):
				curve_in_mode_arti = curve_arti[curve_arti[:,2] == ii]
				if ii == 0:
					plt.plot(curve_in_mode_arti[:,0],curve_in_mode_arti[:,1], linewidth=lineWidthList[0], c='Black')
				else:
					plt.plot(curve_in_mode_arti[:,0],curve_in_mode_arti[:,1], linewidth=lineWidthList[ii], c='Black')
				
			axes = plt.gca()
			axes.set_xlim([fMin, fMax])
			axes.set_ylim([cMin, cMax])
			png_name = 'test' + '.png'
			plt.axis('off')
			plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
			plt.savefig(png_name, dpi=200, bbox_inches='tight', pad_inches=0)
			plt.close()

			image = np.sum(matplotlib.image.imread(png_name),2)
			image = 1-(image-1)/3
			image = PIL.Image.fromarray(image*255).convert('L')
			image = image.resize((512, 512))
			image = np.asarray(image)/255
			
			os.remove(png_name)
				
			spec = np.flip(spec, 0)
			spec[np.isnan(spec)] = 0
			
			
			for ii in range(int(max(curve_arti[:,2]))+1):
				plt.figure(figsize=[5,5])
				curve_in_mode_arti = curve_arti[curve_arti[:,2] == ii]
				if ii == 0:
					plt.plot(curve_in_mode_arti[:,0],curve_in_mode_arti[:,1], linewidth=lineWidthList[0]+12, c='Black')
				else:
					plt.plot(curve_in_mode_arti[:,0],curve_in_mode_arti[:,1], linewidth=lineWidthList[ii]+8, c='Black')
				axes = plt.gca()
				axes.set_xlim([fMin, fMax])
				axes.set_ylim([cMin, cMax])
#				png_name = 'test' + '.png'
				plt.axis('off')
				plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
				plt.savefig(png_name, dpi=200, bbox_inches='tight', pad_inches=0)
				plt.close()
				imageInMode = np.sum(matplotlib.image.imread(png_name),2)
				imageInMode = 1-(imageInMode-1)/3
				imageInMode = PIL.Image.fromarray(imageInMode*255).convert('L')
				imageInMode = imageInMode.resize((512, 512))
				imageInMode = np.asarray(imageInMode)/255
				
				imageInMode = imageInMode * (6 -ii)
				
				os.remove(png_name)
				
				modeImage += imageInMode
				
						
			imageSet.append(spec)
			maskSet.append(image)
			maskModeSet.append(modeImage)
			
			
			
		imageSet = np.array(imageSet)
		maskSet = np.array(maskSet)
		maskModeSet = np.array(maskModeSet)
		
		if len(maskSet) > 0:
			if setFileName == '':
				setFileName = 'trainSet' + str(np.random.rand()) + '.h5'
			
			print('save training set in \''+ setFileName + '\' with size:')
			print(imageSet.shape)
			print(maskSet.shape)
			print(maskModeSet.shape)
			
			with h5py.File(setFileName, 'w') as new_file:
				new_file.create_dataset('image',data=imageSet)
				new_file.create_dataset('mask',data=maskSet)
				new_file.create_dataset('mode',data=maskModeSet)
		else:
			print('NO training set CREATED because curve file is no found.')


	
def curveInterp(curve, freqSeries=[], autoReSeparation=True):
	curve = np.array(curve)
	if np.array(curve).ndim < 2:
		return []			
	if len(curve) == 0:
		return []
	curve = curve[np.argsort(curve[:,-1])]
	
	if curve.shape[1] == 2 or curve.shape[1] == 4:
		raise TypeError('mode value is NO FOUND, plesase use the function \'dispernet.modeSeparation\' to divide the curve to different modes.')
		
		
	if len(freqSeries) == 0:
		freqSeries = np.linspace(0,10,101)
	
	outputCurve = []
	reClusterFlag = False
	for mode in range(int(max(curve[:,-1]))+1):
		curveInMode = curve[curve[:,-1] == mode]
		curveInMode = curveInMode[np.argsort(curveInMode[:,0])]
		curveInMode = curveInMode[curveInMode[:,0] < max(freqSeries)]
		curveInMode = curveInMode[curveInMode[:,0] > min(freqSeries)]
		if len(curveInMode) == 1:
			print('[Warning] dispersion curve at mode ' + str(mode) + ' has less than 2 points, which can not be interpolated')
			print('Modes has been reclustered')
			reClusterFlag = True
			if mode != 0:
				outputCurve = np.concatenate((outputCurve, curveInMode),axis=0)
				
			continue
			
		if len(curveInMode) == 0:
			print('[Warning] No dispersion curve at mode ' + str(mode) + ' , modes has been reclustered')
			reClusterFlag = True
			continue
		
		fMax = max(curveInMode[:,0])
		fMin = min(curveInMode[:,0])
		
		freqSeriesPart = freqSeries[freqSeries <= fMax]
		freqSeriesPart = freqSeriesPart[freqSeriesPart >= fMin]
		
		if len(freqSeriesPart) == 0:
			print('[Warning] points in dispersion curve at mode ' + str(mode) + ' are too close, which can not be interpolated')
			if mode != 0:
				outputCurve = np.vstack([outputCurve, curveInMode])
			continue
		
		veloInterp = np.interp(freqSeriesPart, curveInMode[:,0],curveInMode[:,1])
		
		if curve.shape[1] > 3:
			veloMaxInterp = np.interp(freqSeriesPart, curveInMode[:,0],curveInMode[:,2])
			veloMinInterp = np.interp(freqSeriesPart, curveInMode[:,0],curveInMode[:,3])
			if mode == 0:
				outputCurve = np.vstack([freqSeriesPart, veloInterp, veloMaxInterp, veloMinInterp, mode*np.ones(len(freqSeriesPart))]).T
			else:
				outputCurve = np.vstack([outputCurve, np.vstack([freqSeriesPart, veloInterp, veloMaxInterp, veloMinInterp, mode*np.ones(len(freqSeriesPart))]).T])
		else:
			if outputCurve == []:
				outputCurve = np.vstack([freqSeriesPart, veloInterp, mode*np.ones(len(freqSeriesPart))]).T
			else:
				outputCurve = np.vstack([outputCurve,np.vstack([freqSeriesPart, veloInterp, mode*np.ones(len(freqSeriesPart))]).T])
	
	if reClusterFlag and autoReSeparation:
		autoSeparation(outputCurve)
		
	return np.squeeze(outputCurve)
	
def curveSmooth(curve, sigma=1):

	for mode in range(int(max(curve[:,2])+1)):
		curveInMode = curve[curve[:,-1] == mode]
		curveInMode = curveInMode[np.argsort(curveInMode[:,0])]
		curve_smooth = sn.gaussian_filter1d(curveInMode[:,1],sigma)
		curve[curve[:,2]==mode,1] = curve_smooth
		curve[curve[:,2]==mode,0] = curveInMode[:,0]
		
	return curve
	
def extract(spec, threshold=0.5, freq=[0.,0.3], velo=[2000, 6000], net='noise', mode=0, leapLimit=0.1, freqLimits=[], freqSeries=[], errorbar=False, flipUp=False, searchStep=10, searchBorder=0, returnSpec=False, maxMode=15, ind=-1, url = 'http://10.20.11.42:8514'):
	curve = pick(spec, threshold, freq, velo, net, errorbar, flipUp, searchStep, searchBorder, returnSpec, ind, url)
	
	if returnSpec:
		return curve
		
	else:
		if freqLimits != []:
			fLMin = min(freqLimits)
			fLMax = max(freqLimits)
			
			curve = curve[curve[:,0] < fLMax]
			curve = curve[curve[:,0] > fLMin]
		
		if mode > 0:
			curve = modeSeparation(curve, mode)
		else:
			curve = autoSeparation(curve, leapLimit, maxMode)
		if len(freqSeries) > 0:
			curve = curveInterp(curve, freqSeries)
		
		return curve

def extractPlus(spec, threshold=0.5, freq=[0.,0.3], velo=[2000, 6000], net='noise', mode=0, leapLimit=0.1, freqLimits=[], freqSeries=[], errorbar=False, flipUp=False, searchStep=10, searchBorder=0, returnSpec=False, maxMode=15, ind=-1, url = 'http://10.20.11.42:8514'):
	curve = extractor.pickPointPlus(spec, threshold, freq, velo , searchStep , returnSpec)	
	if returnSpec:
		return curve
	
	else:
		if freqLimits != []:
			fLMin = min(freqLimits)
			fLMax = max(freqLimits)
			
			curve = curve[curve[:,0] < fLMax]
			curve = curve[curve[:,0] > fLMin]
			
#		if mode > 0:
#			curve = modeSeparation(curve, mode)
#		else:
#			curve = autoSeparation(curve, leapLimit, maxMode)
		if len(freqSeries) > 0:
			curve = curveInterp(curve, freqSeries)
			
		return curve


def del_curve_by_mode(curves, mode):
	reCurve = []
	for point in curves:
		if int(point[2])!=mode:
			reCurve.append(point)
	return np.array(reCurve)

def find_nearset(self,key_subwork,key_all,loc_all):
	lon_this = self.loc_info[key_subwork][0]
	lat_this = self.loc_info[key_subwork][1]
	lon_all = []
	lat_all = []
	key_file_all = [str(key) for key in loc_all['key_subwork'].tolist()]
	#print(key_all)
	for key in key_all:
		lon_all.append(loc_all['lon_centroid'][key_file_all.index(str(key))])
		lat_all.append(loc_all['lat_centroid'][key_file_all.index(str(key))])
	#lon_all = loc_all['lon_centroid']
	#lat_all = loc_all['lat_centroid']
	# 找到最近的
	dist = np.sqrt((lon_all-lon_this)**2+(lat_all-lat_this)**2)
	indx = np.argsort(dist)
	key_nearset = key_all[indx[0]]
	return key_nearset
		
class App(object):
	ind = 0
	cidclick = None
	cidDelete = None
	cidSemiAutoAdd = None
	flagDelCurveInMode = False
	
	threshold_set = 0.5
	curve = []
	modeInClick = 0
	net_type_preset = 'noise'
	freq_type_preset = 'Freq.'
	maxMode = 15
	
	trigerMode = True
	searchStep = 2
	autoT = True
	periodCutRate=[]
	semiVeloRange=0.1
	
	def __init__(self,info_basic,lon_all,lat_all,fileList,faults = None,file_project='a-project.yml',flag_plot_or = 0,flag_plot_partrition=0 ,r_flag  = 0,oldfile='None',oldkeys=[],fundfile='None',overfile = 'None',fundkeys=[],filePath='./', curveFilePath = '', freqSeries=[], cmap='viridis', vmin=None, vmax=None, url='http://10.20.11.42:8514', maxMode=0, trigerMode=False , searchStep=2, autoT=True, periodCutRate=0.125, semiAutoRange=0.1):
		self.flag_plot_partrition = flag_plot_partrition	
		self.flag_plot_or = flag_plot_or
		self.r_flag = r_flag
		self.oldfile = oldfile
		self.info_basic = info_basic

		self.oldkeys = oldkeys

		self.fundfile = fundfile
		self.overfile = overfile

		self.fundkeys = fundkeys

		self.autoT = True
		self.periodCutRate=periodCutRate
		self.trigerMode = trigerMode
		self.searchStep = searchStep
		self.semiVeloRange = semiAutoRange
		self.file_project = file_project

		self.lon_all = lon_all
		self.lat_all = lat_all
		self.faults = faults
		
		if maxMode > 0:
			self.maxMode = maxMode
			
		self.fig = plt.figure(figsize=[16,9])
		if self.flag_plot_or == 1:
			self.ax1=plt.subplot(212)
			self.ax2=plt.subplot(211)
		else:
			self.ax1=plt.subplot(111)
		#self.ax1=plt.subplot2grid((4, 1), (1, 0),rowspan=3) 
		#self.ax2=plt.subplot2grid((4, 1), (0, 0))
		plt.subplots_adjust(bottom=0.15, right=0.75,left = 0.05,top=0.95)
		#self.ax3 = plt.subplot(221)
		if flag_plot_partrition == 1:
			self.ax_partrition = plt.axes([0.75, 0.0, 0.25, 0.35])  # 左下角坐标为 (0, 0)，宽度和高度为 0.2
		
		
		
		self.axUpload = plt.axes([0.67, 0.01, 0.08, 0.05])
		self.buttonUpload = Button(self.axUpload, 'Upload to\nDisperNet')
		
		self.axModeDivide = plt.axes([0.05, 0.01, 0.08, 0.05])
		self.buttonModeDivide = Button(self.axModeDivide, 'Mode Divide')
		
		self.axAutoModeDivide = plt.axes([0.05, 0.07, 0.08, 0.05])
		self.buttonAutoModeDivide = Button(self.axAutoModeDivide, 'Automatic\nMode Divide')
		
		self.axAdd = plt.axes([0.15, 0.01, 0.08, 0.05])
		self.buttonAdd = Button(self.axAdd, 'Add Point')
		
		self.axSemiAutoAdd = plt.axes([0.15, 0.07, 0.08, 0.05])
		self.buttonSemiAutoAdd = Button(self.axSemiAutoAdd, 'Semi-auto\nAdd Point')
		
		self.axDelete = plt.axes([0.25, 0.01, 0.08, 0.05])
		self.buttonDelete = Button(self.axDelete, 'Delete Point')
		
		self.axSave = plt.axes([0.35, 0.01, 0.08, 0.05])
		self.buttonSave = Button(self.axSave, 'Save\nPoint')
		
		self.axprev = plt.axes([0.57, 0.07, 0.08, 0.05])
		self.axnext = plt.axes([0.67, 0.07, 0.08, 0.05])
		
		self.bnext = Button(self.axnext, 'Next')
		self.bprev = Button(self.axprev, 'Previous')
		
		modeButtonXLoc = 0.825
		modeButtonXLocShift = 0.1
		
		self.axMode7 = plt.axes([modeButtonXLoc, 0.9, 0.05, 0.05])
		self.axMode6 = plt.axes([modeButtonXLoc, 0.825, 0.05, 0.05])
		self.axMode5 = plt.axes([modeButtonXLoc, 0.75, 0.05, 0.05])
		
		self.axMode4 = plt.axes([modeButtonXLoc, 0.675, 0.05, 0.05])
		self.axMode3 = plt.axes([modeButtonXLoc, 0.6, 0.05, 0.05])
		self.axMode2 = plt.axes([modeButtonXLoc, 0.525, 0.05, 0.05])
		self.axMode1 = plt.axes([modeButtonXLoc, 0.45, 0.05, 0.05])
		self.axMode0 = plt.axes([modeButtonXLoc, 0.375, 0.05, 0.05])
		
		"""
		self.axMode15 = plt.axes([modeButtonXLoc + modeButtonXLocShift, 0.825, 0.05, 0.05])
		self.axMode14 = plt.axes([modeButtonXLoc + modeButtonXLocShift, 0.75, 0.05, 0.05])
		self.axMode13 = plt.axes([modeButtonXLoc+ modeButtonXLocShift, 0.675, 0.05, 0.05])
		
		self.axMode12 = plt.axes([modeButtonXLoc + modeButtonXLocShift, 0.6, 0.05, 0.05])
		self.axMode11 = plt.axes([modeButtonXLoc + modeButtonXLocShift, 0.525, 0.05, 0.05])
		self.axMode10 = plt.axes([modeButtonXLoc + modeButtonXLocShift, 0.45, 0.05, 0.05])
		self.axMode9 = plt.axes([modeButtonXLoc + modeButtonXLocShift, 0.375, 0.05, 0.05])
		self.axMode8 = plt.axes([modeButtonXLoc + modeButtonXLocShift, 0.3, 0.05, 0.05])
		"""
		
		self.buttonMode0 = Button(self.axMode0, 'm 0:')
		self.buttonMode1 = Button(self.axMode1, 'm 1:')
		self.buttonMode2 = Button(self.axMode2, 'm 2:')
		self.buttonMode3 = Button(self.axMode3, 'm 3:')
		self.buttonMode4 = Button(self.axMode4, 'm 4:')
		
		self.buttonMode5 = Button(self.axMode5, 'm 5:')
		self.buttonMode6 = Button(self.axMode6, 'm 6:')
		self.buttonMode7 = Button(self.axMode7, 'm 7:')
		"""
		self.buttonMode8 = Button(self.axMode8, 'm 8:')
		self.buttonMode9 = Button(self.axMode9, 'm 9:')
		self.buttonMode10= Button(self.axMode10, 'm10:')
		self.buttonMode11= Button(self.axMode11, 'm11:')
		self.buttonMode12= Button(self.axMode12, 'm12:')
		
		self.buttonMode13= Button(self.axMode13, 'm13:')
		self.buttonMode14= Button(self.axMode14, 'm14:')
		self.buttonMode15= Button(self.axMode15, 'm15:')
		"""
		self.textM7 = self.fig.text(modeButtonXLoc+0.07,0.92, '0')
		self.textM6 = self.fig.text(modeButtonXLoc+0.07 ,0.845, '0')
		self.textM5 = self.fig.text(modeButtonXLoc+0.07,0.77, '0')
		
		self.textM4 = self.fig.text(modeButtonXLoc+0.07,0.695, '0')
		self.textM3 = self.fig.text(modeButtonXLoc+0.07,0.62, '0')
		self.textM2 = self.fig.text(modeButtonXLoc+0.07,0.545, '0')
		self.textM1 = self.fig.text(modeButtonXLoc+0.07,0.47, '0')
		self.textM0 = self.fig.text(modeButtonXLoc+0.07,0.395, '0')
		
		"""
		self.textM15= self.fig.text(modeButtonXLoc+ modeButtonXLocShift+0.07,0.845, '0')
		self.textM14= self.fig.text(modeButtonXLoc+ modeButtonXLocShift+0.07 ,0.77, '0')
		self.textM13= self.fig.text(modeButtonXLoc+ modeButtonXLocShift+0.07,0.695, '0')
		
		self.textM12= self.fig.text(modeButtonXLoc+ modeButtonXLocShift+0.07,0.62, '0')
		self.textM11= self.fig.text(modeButtonXLoc+ modeButtonXLocShift+0.07,0.545, '0')
		self.textM10= self.fig.text(modeButtonXLoc+ modeButtonXLocShift+0.07,0.47, '0')
		self.textM9 = self.fig.text(modeButtonXLoc+ modeButtonXLocShift+0.07,0.395, '0')
		self.textM8 = self.fig.text(modeButtonXLoc+ modeButtonXLocShift+0.07,0.32, '0')
		"""
		self.axInterp = plt.axes([0.35, 0.07, 0.08, 0.05])
		self.buttonInterp = Button(self.axInterp, 'Curve\nInterp')
		
		self.axSmooth = plt.axes([0.25, 0.07, 0.08, 0.05])
		self.buttonSmooth= Button(self.axSmooth, 'Curve\nSmooth')
		
		
		self.axth = plt.axes([0.835, 0.34, 0.14, 0.03])
		self.slth = Slider(self.axth, 'th', 0, 1.0, valinit=0.5)
		
		self.axNetType = plt.axes([0.92, 0.78, 0.06, 0.15])
		self.checkNetType = CheckButtons(self.axNetType, ['noise','event','noise2', 'noise3','toLB','toLB2'],[1,0,0,0,0,0])
		self.axNetType.set_title('Net Type')	
		
		self.axFreqType = plt.axes([0.92, 0.4, 0.06, 0.1])
		self.checkFreqType = CheckButtons(self.axFreqType, ['Freq.','Period'],[1,0])
		self.axFreqType.set_title('Disp. Mode')
		
		self.axDelCurveInMode = plt.axes([0.57, 0.01, 0.08, 0.05])
		self.bDelCurveInMode = Button(self.axDelCurveInMode, 'Delete Mode')
			
		self.buttonUpload.on_clicked(self.upload)
		self.buttonModeDivide.on_clicked(self.modeDivide)
		self.buttonAdd.on_clicked(self.add_mode_on)
		self.buttonDelete.on_clicked(self.deletePoint)
		self.buttonSemiAutoAdd.on_clicked(self.semi_auto_add_on)
		self.buttonMode0.on_clicked(self.mode0ButtonClick)
		self.buttonMode1.on_clicked(self.mode1ButtonClick)
		self.buttonMode2.on_clicked(self.mode2ButtonClick)
		self.buttonMode3.on_clicked(self.mode3ButtonClick)
		self.buttonMode4.on_clicked(self.mode4ButtonClick)
		self.buttonMode5.on_clicked(self.mode5ButtonClick)
		self.buttonMode6.on_clicked(self.mode6ButtonClick)
		self.buttonMode7.on_clicked(self.mode7ButtonClick)
		"""
		self.buttonMode8.on_clicked(self.mode8ButtonClick)
		self.buttonMode9.on_clicked(self.mode9ButtonClick)
		self.buttonMode10.on_clicked(self.mode10ButtonClick)
		self.buttonMode11.on_clicked(self.mode11ButtonClick)
		self.buttonMode12.on_clicked(self.mode12ButtonClick)
		self.buttonMode13.on_clicked(self.mode13ButtonClick)
		self.buttonMode14.on_clicked(self.mode14ButtonClick)
		self.buttonMode15.on_clicked(self.mode15ButtonClick)
		"""
		self.bnext.on_clicked(self.next)
		self.bprev.on_clicked(self.prev)
		self.buttonSave.on_clicked(self.save)
		self.buttonInterp.on_clicked(self.curveInterpButton)
		self.buttonSmooth.on_clicked(self.curveSmoothButton)
		self.buttonAutoModeDivide.on_clicked(self.autoDivide)
		self.slth.on_changed(self.threshold_changed)
		self.checkNetType.on_clicked(self.set_net_type)
		self.checkFreqType.on_clicked(self.set_freq_type)
		
		self.bDelCurveInMode.on_clicked(self.delete_curve_in_mode)
		
		self.cmap = cmap
		self.vmin = vmin
		self.vmax = vmax
		
		self.url = url
		

		#self.fileList = self.get_file_list(filePath, end='.h5')
		#self.fileList = self.natural_sort(self.fileList)
		"""
		fileList = self.get_file_list(filePath, end='.h5')
		num_fileList = [int(file[file.find('--')+2:file.find('.h5')]) for file in fileList ]
		index = np.argsort(num_fileList)
		fileList = np.array(fileList)[index]
		self.fileList = list(fileList)
		"""
		self.fileList = fileList


		self.filePath = filePath  + '/'
		if self.fileList == []:
			raise IOError('No *.h5 file found in the given Path, please check. \nYou can use the function dispernet.save2h5 to transfer the sptectrum to the specific *.h5 file')
			
		self.fileName = self.fileList[0]
		
		self.modeNum = 2
		
		if curveFilePath == '':
			self.curveFilePath = filePath
		else:
			self.curveFilePath = curveFilePath + '/'
			
		print(self.filePath + self.fileName)
		with h5py.File(self.filePath + self.fileName, 'r') as fr:
			self.freq = np.array(fr['f'])
			self.velo = np.array(fr['c'])
			if 'ds' in fr.keys():
				self.spec = np.array(fr['ds'])
			else:
				self.spec = np.array(fr['amp'])
			
			if self.flag_plot_or == 1:
				self.spec_or = np.array(fr['amp_or'])
				
		if self.autoT and (self.spec.shape == (len(self.freq), len(self.velo))):
			print('[info] Auto Transposed, if you do not need this feature, set autoT=True when applicaion inital.')
			self.spec = self.spec.T
			
		curveFileName = self.fileName[:-3] + 'curve.txt'
		curveFileExist = self.get_file_list(self.curveFilePath, begin=curveFileName)
		if len(curveFileExist) > 0:
			self.curve = np.loadtxt(os.path.join(self.curveFilePath, curveFileName))
			
		else:
			self.curve = []
			
		self.pSpec = []
		self.period = []
		self.r_this = 0
		
		if len(freqSeries) > 0:
			self.freqSeriesForInterp = np.array(freqSeries)
		else:
			self.freqSeriesForInterp = np.linspace(np.min(self.freq), np.max(self.freq), 100)

		# read locations
		with open(self.file_project, 'r', encoding='utf-8') as f:
			proj = yaml.load(f.read(), Loader=yaml.FullLoader)
		name_project = proj['name']
		with open('0_config.yml', 'r', encoding='utf-8') as f:
			dir_config = yaml.load(f.read(), Loader=yaml.FullLoader)
		dir_project_workspace = dir_config['dir_project_workspace']
		dir_project = os.path.join(dir_project_workspace, name_project)
		self.dir_project = dir_project

		

		# read station info
		#loc_info = pd.read_excel(self.dir_project+'subwork_location.xlsx',sheet_name='location')
		#self.loc_info = loc_info
		loc_info = {}
		for key_subwork in self.info_basic['key_subworks']:
			filePath = self.dir_project + self.info_basic['dir_partition'] + str(key_subwork) + '.txt'
			stations, lat, lon = np.loadtxt(filePath, dtype='str' , unpack=True)
			stations = stations.tolist()
			lat = lat.tolist()
			lon = lon.tolist()
			loc_info[key_subwork] = [np.mean(np.array(lon).astype(float)),np.mean(np.array(lat).astype(float))]
		self.loc_info = loc_info
			

		# find this key
		key_this = self.fileName[self.fileName.find('_')+1:self.fileName.find('.')]
		if '_' in key_this:
			key_this = key_this[:key_this.find('_')]
		self.key_subworks = info_basic['key_subworks']

		if r_flag != 0:
			r_this = r_flag[key_this]
		else:
			r_this = 0
		self.r_this = r_this
		
		# find old key
		key_olds = self.oldkeys
		curve_old_FileExist = os.path.exists(self.oldfile)
		if curve_old_FileExist:
			#stalistname = self.dir_project+ info_basic['stalistname']
			loc_old = pd.read_excel(self.oldfile+'subwork_location.xlsx',sheet_name='location')
			key_find = find_nearset(self,key_this,key_olds,loc_old)

			curve_old_File = self.oldfile + 'ds_'+key_find + 'curve.txt'

			curve_old = np.loadtxt(curve_old_File)

			self.curve_old = np.array(curve_old)

		curve_fund_FileExist = os.path.exists(self.fundfile)
		if curve_fund_FileExist:
			curve_fund_File = self.fundfile + 'disp_model_'+key_this + '.txt'

			if os.path.exists(curve_fund_File):
				curve_fund = np.loadtxt(curve_fund_File)
				self.curve_fund = np.array(curve_fund)
			else:
				curve_fund_FileExist = False


		info_basic = self.info_basic
		#stalistname_all = info_basic['stalistname_all']
		#stainfo = pd.read_excel(stalistname_all)
		#self.lat_all = stainfo['latitude'].tolist() 
		#self.lon_all = stainfo['longitude'].tolist()
		#self.faults = np.load('clark_faults.npy', allow_pickle='TRUE').item()



		self.axBefore = plt.axes([0.92, 0.55, 0.06, 0.03])
		self.buttonBefore= Button(self.axBefore, 'add over')
		self.buttonBefore.on_clicked(self.Before)

		self.axSaveBefore = plt.axes([0.92, 0.70, 0.06, 0.03])
		self.buttonSaveBefore= Button(self.axSaveBefore, 'save before')
		self.buttonSaveBefore.on_clicked(self.SaveBefore)

		self.axNear = plt.axes([0.92, 0.74, 0.06, 0.03])
		self.buttonNear= Button(self.axNear, 'save Near')
		self.buttonNear.on_clicked(self.SaveNear)

		self.axFundamental = plt.axes([0.92, 0.66, 0.06, 0.03])
		self.buttonFundamental= Button(self.axFundamental, 'add funda')
		self.buttonFundamental.on_clicked(self.Fundamental)

		self.axSaveOver = plt.axes([0.92, 0.59, 0.06, 0.03])
		self.buttonSaveOver= Button(self.axSaveOver, 'save Over')
		self.buttonSaveOver.on_clicked(self.SaveOver)
			
		if self.flag_plot_or == 1:
			#self.show_or()
			self.ax2.cla()
			show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		if self.flag_plot_partrition == 1:
			#sta_info_all = pd.read_excel(self.dir_project+self.info_basic['stalistname'],sheet_name=None)
			#self.sta_info_all = sta_info_all
			sta_info_all = {}
			stations = {}
			lat = {}
			lon = {}
			for key_subwork in self.key_subworks:
				filePath = self.dir_project + self.info_basic['dir_partition'] + str(key_subwork) + '.txt'
				stations_this, lat_this, lon_this = np.loadtxt(filePath, dtype='str' , unpack=True)
				stations[key_subwork] = stations_this.tolist()
				lat[key_subwork] = lat_this.astype(float).tolist()
				lon[key_subwork] = lon_this.astype(float).tolist()
			sta_info_all['stations'] = stations
			sta_info_all['lat'] = lat
			sta_info_all['lon'] = lon
			self.sta_info_all = sta_info_all

			show_partrition(self)
		
		
		self.ax1.set_title(self.fileName)
		cursor = Cursor(self.ax1, useblit=True, color='w', linewidth=1)
		plt.show()

	def show_or(self):
		self.ax2.imshow(np.flip(self.spec_or,0),aspect='auto',cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
		self.ax2.axis('off')

	



	def SaveBefore(self,event):
		curveFileName = self.fileName[:-3] + 'curve.txt'
		curveFileExist = self.get_file_list(self.curveFilePath, begin=curveFileName)

		key_this = self.fileName[self.fileName.find('_')+1:self.fileName.find('.')]
		if '_' in key_this:
			key_this = key_this[:key_this.find('_')]
		key_olds = self.oldkeys

		r_flag = self.r_flag
		if r_flag != 0:
			r_this = r_flag[key_this]
		else:
			r_this = 0
		self.r_this = r_this
		
		curve_old_FileExist = os.path.exists(self.oldfile)
		if curve_old_FileExist:
			loc_old = pd.read_excel(self.oldfile+'subwork_location.xlsx',sheet_name='location')
			key_find = find_nearset(self,key_this,key_olds,loc_old)

			curve_old_File = self.oldfile + 'ds_'+key_find + 'curve.txt'

			curve_old = np.loadtxt(curve_old_File)

			self.curve_old = np.array(curve_old)
			print('key_ref='+key_find)
			if len(curveFileExist) > 0:
				self.curve = np.append(self.curve, self.curve_old, axis=0)
			else:
				self.curve = self.curve_old
		show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		if self.flag_plot_partrition == 1:
			show_partrition(self)
		if self.flag_plot_or == 1:
			#self.show_or()
			self.ax2.cla()
			show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		plt.draw()

	def Before(self,event):
		key_this = self.fileName[self.fileName.find('_')+1:self.fileName.find('.')]
		if '_' in key_this:
			key_this = key_this[:key_this.find('_')]
		print(key_this)
		
		curve_over_FileExist = os.path.exists(self.overfile)
		if curve_over_FileExist:
			curve_over_File = self.overfile + 'disp_model_'+key_this + '.txt'

			if os.path.exists(curve_over_File):
				curve_over = np.loadtxt(curve_over_File)
				self.curve_over = np.array(curve_over)
				print('key_ref='+key_this)
				show(self.spec,self.curve_over,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
				if self.flag_plot_or == 1:
					#self.show_or()
					self.ax2.cla()
					show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
				plt.draw()
				#if len(curveFileExist) > 0:
				#	self.curve = np.append(self.curve, self.curve_over, axis=0)
				#else:
			#		self.curve = self.curve_over
			else:
				print('do not exist this fundamentle file')
			#show(self.spec,self.curve,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
			#plt.draw()

	def SaveNear(self,event):
		curveFileName = self.fileName[:-3] + 'curve.txt'
		curveFileExist = self.get_file_list(self.curveFilePath, begin=curveFileName)

		key_this = self.fileName[self.fileName.find('_')+1:self.fileName.find('.')]
		if '_' in key_this:
			key_this = key_this[:key_this.find('_')]
		key_all = [file[file.find('_')+1:file.find('curve')] for file in os.listdir(self.curveFilePath)]
		if key_this in key_all:
			key_all.remove(key_this)

		key_find = find_nearset(self,key_this,key_all,self.loc_info)

		curve_old_File = self.curveFilePath + 'ds_'+key_find + 'curve.txt'

		curve_old = np.loadtxt(curve_old_File)

		self.curve_old = np.array(curve_old)
		print('key_ref='+key_find)
		if len(curveFileExist) > 0:
			self.curve = np.append(self.curve, self.curve_old, axis=0)
		else:
			self.curve = self.curve_old



		show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		if self.flag_plot_or == 1:
			#self.show_or()
			self.ax2.cla()
			show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		plt.draw()

	def Fundamental(self,event):
		key_this = self.fileName[self.fileName.find('_')+1:self.fileName.find('.')]
		if '_' in key_this:
			key_this = key_this[:key_this.find('_')]
		key_fund = self.fundkeys
		
		curve_fund_FileExist = os.path.exists(self.fundfile)
		if curve_fund_FileExist:
			curve_fund_File = self.fundfile + 'disp_model_'+key_this + '.txt'

			if os.path.exists(curve_fund_File):
				curve_fund = np.loadtxt(curve_fund_File)
				self.curve_fund = np.array(curve_fund)
				print('key_ref='+key_this)
				show(self.spec,self.curve_fund,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
				if self.flag_plot_or == 1:
					#self.show_or()
					self.ax2.cla()
					show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
				plt.draw()
			else:
				print('do not exist this fundamentle file')

	def SaveOver(self,event):
		curveFileName = self.fileName[:-3] + 'curve.txt'
		curveFileExist = self.get_file_list(self.curveFilePath, begin=curveFileName)

		key_this = self.fileName[self.fileName.find('_')+1:self.fileName.find('.')]
		if '_' in key_this:
			key_this = key_this[:key_this.find('_')]

		r_flag = self.r_flag
		if r_flag != 0:
			r_this = r_flag[key_this]
		else:
			r_this = 0
		self.r_this = r_this
		
		curve_over_FileExist = os.path.exists(self.overfile)
		if curve_over_FileExist:
			curve_over_File = self.overfile + 'disp_model_'+key_this + '.txt'
			#print('test')

			if os.path.exists(curve_over_File):
				curve_over = np.loadtxt(curve_over_File)
				self.curve_over = np.array(curve_over)
				print('key_ref='+key_this)
			if len(curveFileExist) > 0:
				self.curve = np.append(self.curve, self.curve_over, axis=0)
			else:
				self.curve = self.curve_over
		show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		if self.flag_plot_or == 1:
			#self.show_or()
			self.ax2.cla()
			show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		plt.draw()


	
		
		
	def threshold_changed(self,event):
		self.threshold_set = float(event)
		
	def curveInterpButton(self,event):
		
		if self.curve == []:
			self.ax1.set_title('Please UPLOAD or manually pick the curve FIRST!!')
			return
		
		if self.curve.shape[1] == 2:
			self.ax1.set_title('Please Divide the curve to different mode FIRST!!')
			return
		
		self.curve = curveInterp(self.curve, self.freqSeriesForInterp)
		
		
		self.ax1.cla()
		
		if self.freq_type_preset == 'Period':
			curveP = self.curve.copy()
			curveP[:, 0] = 1 / curveP[:, 0]
			
			show(self.pSpec,curveP,r_this = self.r_this,freq=self.period, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, xLabel='Period (s)')
		else:
			show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
			
		self.ax1.set_title('Curve Interpolation')
		plt.draw()
		
	def curveSmoothButton(self,event):
		
		if self.curve == []:
			self.ax1.set_title('Please UPLOAD or manually pick the curve FIRST!!')
			return
		
		if self.curve.shape[1] == 2:
			self.ax1.set_title('Please Divide the curve to different mode FIRST!!')
			return
		
		self.curve = curveSmooth(self.curve, 1)
		self.ax1.cla()
		if self.freq_type_preset == 'Period':
			curveP = self.curve.copy()
			curveP[:, 0] = 1 / curveP[:, 0]
			
			show(self.pSpec,curveP,r_this = self.r_this,freq=self.period, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, xLabel='Period (s)')
		else:
			show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		self.ax1.set_title('Curve Smoothed')
		plt.draw()
		
		
	
	def delete_curve_in_mode(self, event):
		if self.curve == []:
			self.ax1.set_title('Please UPLOAD or manually pick the curve FIRST!!')
			return
		
		if self.curve.shape[1] == 2:
			self.ax1.set_title('Please Divide the curve to different mode FIRST!!')
			return
		if not self.flagDelCurveInMode:
			self.flagDelCurveInMode = True
			self.ax1.set_title('Delete Curve in Mode, Choose Mode Number and Delete Total Curve.')
			
		else:
			self.flagDelCurveInMode = False
			self.ax1.set_title('[off] Delete Curve in Mode.')
			
		plt.draw()
		return

		
	
	def save(self, event):
		if self.curve == []:
			self.ax1.set_title('No curves to save yet.')
		else:
			if len(self.curve) > 1:
				if self.curve.shape[1] !=2 and  self.curve.shape[1] !=4:
					self.curve = self.curve[np.argsort(self.curve[:,-1])]
					for mode in range(int(max(self.curve[:,-1])+1)):
						curveInMode = self.curve[self.curve[:,-1] == mode]					
						self.curve[self.curve[:,-1] == mode] = curveInMode[np.argsort(curveInMode[:,0])]
						
			if self.curve.shape[1] > 2:		
				np.savetxt(self.curveFilePath + self.fileName[:-3] + 'curve.txt', self.curve, fmt='%.6f  %.6f  %i')
			else:
				np.savetxt(self.curveFilePath + self.fileName[:-3] + 'curve.txt', self.curve, fmt='%.6f  %.6f')
			self.ax1.set_title('Curve file saved. ('+str(len(self.curve)) + ' points)')
			
		plt.draw()
		
		
	def set_net_type(self,event):
		self.net_type_preset = str(event)
		netNameList = ['noise','event','noise2','noise3','toLB','toLB2']
		changedValue = np.zeros(len(netNameList))
		for ind,name in enumerate(netNameList):
			if self.net_type_preset == name:
				changedValue[ind] = 1
				
		self.axNetType.cla()
		self.checkNetType = CheckButtons(self.axNetType, ['noise','event','noise2', 'noise3','toLB','toLB2'],changedValue)
		self.checkNetType.on_clicked(self.set_net_type)
		self.axNetType.set_title('Net Type')
		
		plt.draw()
		
	def set_freq_type(self,event):
		self.freq_type_preset = str(event)
		freqNameList = ['Freq.','Period']
		changedValue = np.zeros(len(freqNameList))
		for ind,name in enumerate(freqNameList):
			if self.freq_type_preset == name:
				changedValue[ind] = 1
				
		if len(self.pSpec) == 0:
			self.pSpec, self.period = freq2Period(self.spec, self.freq, cutRate=self.periodCutRate)
			
		self.axFreqType.cla()
		self.checkFreqType = CheckButtons(self.axFreqType, ['Freq.','Period'],changedValue)
		self.checkFreqType.on_clicked(self.set_freq_type)
		self.axFreqType.set_title('Disp. Mode')
		
		if self.freq_type_preset == 'Period':
			self.ax1.cla()
			if len(self.curve) > 0:
				curveP = self.curve.copy()
				curveP[:, 0] = 1 / curveP[:, 0]
				show(self.pSpec,curveP,r_this = self.r_this,freq=self.period, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, xLabel='Period (s)')
			else:
				show(self.pSpec,[],r_this = self.r_this,freq=self.period, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, xLabel='Period (s)')
				
			self.axUpload.cla()
			self.buttonUpload = Button(self.axUpload, 'Auto Pick\nin Period')
			self.buttonUpload.on_clicked(self.autoSearchInPeriod)
			
		else:
			self.axUpload.cla()
			self.buttonUpload = Button(self.axUpload, 'Upload to \nDisperNet')
			self.buttonUpload.on_clicked(self.upload)
			
			self.ax1.cla()
			show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
			
		plt.draw()
		
	def next(self, event):
		self.ind += 1
		
		if self.ind >= len(self.fileList):
			self.ind = 0
			
		self.fileName  = self.fileList[self.ind]
		#self.key_this = self.key_subworks[self.ind]
		with h5py.File(self.filePath + self.fileName, 'r') as fr:
			self.freq = np.array(fr['f'])
			self.velo = np.array(fr['c'])
#			self.spec = np.array(fr['amp'])
			if 'ds' in fr.keys():
				self.spec = np.array(fr['ds'])
			else:
				self.spec = np.array(fr['amp'])
			if self.flag_plot_or == 1:
				self.spec_or = np.array(fr['amp_or'])
				
		curveFileName = self.fileName[:-3] + 'curve.txt'
		curveFileExist = self.get_file_list(self.curveFilePath, begin=curveFileName)
		if len(curveFileExist) > 0:
			self.curve = np.loadtxt(os.path.join(self.curveFilePath, curveFileName))
			
		else:
			self.curve = []
		self.pSpec = []
		self.ax1.cla()
		show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		if self.flag_plot_partrition == 1:
			show_partrition(self)
		if self.flag_plot_or == 1:
			#self.show_or()
			self.ax2.cla()
			show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		self.ax1.set_title(self.fileName)
		
		if self.freq_type_preset == 'Period':
			self.axFreqType.cla()
			self.checkFreqType = CheckButtons(self.axFreqType, ['Freq.','Period'], [1, 0])
			self.checkFreqType.on_clicked(self.set_freq_type)
			self.axFreqType.set_title('Disp. Mode')
			self.freq_type_preset = 'Freq.'
			
			self.axUpload.cla()
			self.buttonUpload = Button(self.axUpload, 'Upload to \nDisperNet')
			self.buttonUpload.on_clicked(self.upload)
			
		plt.draw()
		
#		#### Triger Period MODE !!!!!
#		self.set_freq_type('Period')
#		self.autoSearchInPeriod(event)
#		self.curveInterpButton(event)
#		self.curveSmoothButton(event)
#		self.save(event)
#		self.next(event)
#		#### Triger Period MODE !!!!!
		
	def prev(self, event):
		self.ind -= 1
		if self.ind < 0:
			self.ind = len(self.fileList) - 1
			
		self.fileName  = self.fileList[self.ind]
		
		with h5py.File(self.filePath + self.fileName, 'r') as fr:
			self.freq = np.array(fr['f'])
			self.velo = np.array(fr['c'])
#			self.spec = np.array(fr['amp'])
			if 'ds' in fr.keys():
				self.spec = np.array(fr['ds'])
			else:
				self.spec = np.array(fr['amp'])
			if self.flag_plot_or == 1:
				self.spec_or = np.array(fr['amp_or'])
		curveFileName = self.fileName[:-3] + 'curve.txt'
		
		curveFileExist = self.get_file_list(self.curveFilePath, begin=curveFileName)
		if len(curveFileExist) > 0:
			self.curve = np.loadtxt(os.path.join(self.curveFilePath, curveFileName))
			
		else:
			self.curve = []
		self.pSpec = []
		self.ax1.cla()
		show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		self.ax1.set_title(self.fileName)

		if self.flag_plot_partrition == 1:
			show_partrition(self)
		if self.flag_plot_or == 1:
			#self.show_or()
			self.ax2.cla()
			show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		
		if self.freq_type_preset == 'Period':
			self.axFreqType.cla()
			self.checkFreqType = CheckButtons(self.axFreqType, ['Freq.','Period'], [1, 0])
			self.checkFreqType.on_clicked(self.set_freq_type)
			self.axFreqType.set_title('Disp. Mode')
			self.freq_type_preset = 'Freq.'
			
			self.axUpload.cla()
			self.buttonUpload = Button(self.axUpload, 'Upload to \nDisperNet')
			self.buttonUpload.on_clicked(self.upload)
			
		plt.draw()
		
		
	def on_click(self, event):
		
		x = event.xdata
		y = event.ydata
		
		if x == None:
			self.fig.canvas.mpl_disconnect(self.cidclick)
			self.ax1.set_title('Add Mode off')
			plt.draw()
		else:
			if event.inaxes == self.ax1:
				if self.freq_type_preset == 'Period':
					newPoint = [1/x, y, self.modeInClick]
				else:
					newPoint = [x, y, self.modeInClick]
					
				if self.curve == []:
					self.curve = np.array(newPoint)
					
				else:
					if self.curve.ndim > 1:
						if self.curve.shape[1] == 2:
							self.curve = np.vstack([self.curve, newPoint[0:2]])
						else:
							self.curve = np.vstack([self.curve, newPoint])
					else:
						if len(self.curve) == 2:
							self.curve = np.vstack([self.curve, newPoint[0:2]])
						else:
							self.curve = np.vstack([self.curve, newPoint])
							
					self.ax1.cla()
					if self.freq_type_preset == 'Period':
						
						curveP = self.curve.copy()
						curveP[:, 0] = 1 / curveP[:, 0]
						
						show(self.pSpec,curveP,r_this = self.r_this,freq=self.period, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, xLabel='Period (s)')
					else:
						show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
						if self.flag_plot_or == 1:
							#self.show_or()
							self.ax2.cla()
							show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
					self.ax1.set_title('Added :' + str(x) + ', '+ str(y))
					self.reflashModeLabel()
					plt.draw()
					
					
	def on_semi_auto_event(self, event):
		
		
		x = event.xdata
		y = event.ydata
		
		
		
		if x == None:
			self.fig.canvas.mpl_disconnect(self.cidSemiAutoAdd)
			self.ax1.set_title('Semi-auto Add Mode off')
			plt.draw()
		else:
			if event.inaxes == self.ax1:
				if self.freq_type_preset == 'Period':
					xIndx = np.around(((x - np.min(self.period)) / (np.max(self.period) - np.min(self.period)) * self.pSpec.shape[1]))
					yIndx = np.around(((y - np.min(self.velo)) / (np.max(self.velo) - np.min(self.velo)) * self.pSpec.shape[0]))
					ySearchRange= int(self.pSpec.shape[0] * self.semiVeloRange)
					searchLowerBound = np.max([int(yIndx - ySearchRange), 0])
					searchUpperBound = np.min([int(yIndx + ySearchRange), self.pSpec.shape[0]-1])
					
					searchSeries = self.pSpec[searchLowerBound:searchUpperBound, np.int(xIndx)]
					maxAmp = np.max(searchSeries)
					yMaxInd = np.mean(np.argwhere(searchSeries==maxAmp)) + searchLowerBound
#					yMaxInd = np.argmax(searchSeries) + searchLowerBound
					newY = yMaxInd / self.pSpec.shape[0] * (np.max(self.velo) - np.min(self.velo)) + np.min(self.velo)
					newPoint = [1/x, newY, self.modeInClick]
				else:
					xIndx = np.around(((x - np.min(self.freq)) / (np.max(self.freq) - np.min(self.freq)) * self.spec.shape[1]))
					yIndx = np.around(((y - np.min(self.velo)) / (np.max(self.velo) - np.min(self.velo)) * self.spec.shape[0]))
					ySearchRange= int(self.spec.shape[0] * self.semiVeloRange)
					searchLowerBound = np.max([int(yIndx - ySearchRange), 0])
					searchUpperBound = np.min([int(yIndx + ySearchRange), self.spec.shape[0]-1])
					
					searchSeries = self.spec[searchLowerBound:searchUpperBound, int(xIndx)]
					maxAmp = np.max(searchSeries)
					yMaxInd = np.mean(np.argwhere(searchSeries==maxAmp)) + searchLowerBound
					newY = yMaxInd / self.spec.shape[0] * (np.max(self.velo) - np.min(self.velo)) + np.min(self.velo)
					
					newPoint = [x, newY, self.modeInClick]
					
				if self.curve == []:
					self.curve = np.array(newPoint)
					
				else:
					if self.curve.ndim > 1:
						if self.curve.shape[1] == 2:
							self.curve = np.vstack([self.curve, newPoint[0:2]])
						else:
							self.curve = np.vstack([self.curve, newPoint])
					else:
						if len(self.curve) == 2:
							self.curve = np.vstack([self.curve, newPoint[0:2]])
						else:
							self.curve = np.vstack([self.curve, newPoint])
							
					self.ax1.cla()
					if self.freq_type_preset == 'Period':
						
						curveP = self.curve.copy()
						curveP[:, 0] = 1 / curveP[:, 0]
						
						show(self.pSpec,curveP,r_this = self.r_this,freq=self.period, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, xLabel='Period (s)')
					else:
						show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
						
					self.ax1.set_title('Semi-auto Added:\n' + str(x) + ', '+ str(newY))
					self.reflashModeLabel()
					plt.draw()
					
					
	def semi_auto_add_on(self,event):
		if self.cidDelete: 
			self.fig.canvas.mpl_disconnect(self.cidDelete)
			
		if self.cidclick:
			self.fig.canvas.mpl_disconnect(self.cidclick)
			
		self.cidSemiAutoAdd = self.fig.canvas.mpl_connect("button_press_event", self.on_semi_auto_event)
		self.ax1.set_title('Semi-auto Add Mode On')
		plt.draw()
		
	def add_mode_on(self,event):
		if self.flagDelCurveInMode:
			self.flagDelCurveInMode = False
			
		if self.cidDelete: 
			self.fig.canvas.mpl_disconnect(self.cidDelete)
			
		if self.cidSemiAutoAdd:
			self.fig.canvas.mpl_disconnect(self.cidSemiAutoAdd)
			
		self.cidclick = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
		self.ax1.set_title('Add Mode On')
		plt.draw()
		
	def deletePoint(self, event):
		if self.cidclick:
			self.fig.canvas.mpl_disconnect(self.cidclick)
			
		if self.cidSemiAutoAdd:
			self.fig.canvas.mpl_disconnect(self.cidSemiAutoAdd)
			
		self.cidDelete = self.fig.canvas.mpl_connect("button_press_event", self.on_delete_event)
		self.ax1.set_title('Delete Mode On')
		plt.draw()
		
	def on_delete_event(self, event):
		x = event.xdata
		y = event.ydata
		
		if x == None:
			self.fig.canvas.mpl_disconnect(self.cidDelete)
			self.ax1.set_title('Delete Mode off')
			plt.draw()
			
		else:
			if event.inaxes == self.ax1 and self.curve != []:
				
				if self.freq_type_preset == 'Period':
					errorRangeX = (max(self.period) - min(self.period)) / 100
				else:
					errorRangeX = (max(self.freq) - min(self.freq)) / 100
					
				errorRangeY = (max(self.velo) - min(self.velo)) / 100
				
				deleteList = []
				for ind, point in enumerate(self.curve):
					if self.freq_type_preset == 'Period':
						if (1/point[0] < x+errorRangeX ) and (1/point[0] > x-errorRangeX) and(point[1] < y+errorRangeY) and (point[1] > y-errorRangeY):
							deleteList.append(ind)
					else:
						if (point[0] < x+errorRangeX ) and (point[0] > x-errorRangeX) and(point[1] < y+errorRangeY) and (point[1] > y-errorRangeY):
							deleteList.append(ind)
							
							
				if deleteList != []:
					self.curve = np.delete(self.curve, deleteList,0)
					
					self.ax1.cla()
					if self.freq_type_preset == 'Period':
						
						curveP = self.curve.copy()
						curveP[:, 0] = 1 / curveP[:, 0]
						
						show(self.pSpec,curveP,r_this = self.r_this,freq=self.period, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, xLabel='Period (s)')
					else:
						
						show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
					if self.flag_plot_or == 1:
						#self.show_or()
						self.ax2.cla()
						show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
					self.ax1.set_title('Deleted :' + str(x) + ', '+ str(y))
					self.reflashModeLabel()
					plt.draw()
					
			else:
				self.fig.canvas.mpl_disconnect(self.cidDelete)
				
	def reflashModeLabel(self):
		if self.curve == [] or len(self.curve)==1: 
			self.ax1.set_title('Please UPLOAD or manually pick the curve FIRST!!')
		else:
			self.textM0.set_text(str(len(self.curve[self.curve[:,-1]==0])))
			self.textM1.set_text(str(len(self.curve[self.curve[:,-1]==1])))
			self.textM2.set_text(str(len(self.curve[self.curve[:,-1]==2])))
			self.textM3.set_text(str(len(self.curve[self.curve[:,-1]==3])))
			self.textM4.set_text(str(len(self.curve[self.curve[:,-1]==4])))
			
			self.textM5.set_text(str(len(self.curve[self.curve[:,-1]==5])))
			self.textM6.set_text(str(len(self.curve[self.curve[:,-1]==6])))
			self.textM7.set_text(str(len(self.curve[self.curve[:,-1]==7])))
			"""
			self.textM8.set_text(str(len(self.curve[self.curve[:,-1]==8])))
			self.textM9.set_text(str(len(self.curve[self.curve[:,-1]==9])))
			self.textM10.set_text(str(len(self.curve[self.curve[:,-1]==10])))
			self.textM11.set_text(str(len(self.curve[self.curve[:,-1]==11])))
			self.textM12.set_text(str(len(self.curve[self.curve[:,-1]==12])))
			
			self.textM13.set_text(str(len(self.curve[self.curve[:,-1]==13])))
			self.textM14.set_text(str(len(self.curve[self.curve[:,-1]==14])))
			self.textM15.set_text(str(len(self.curve[self.curve[:,-1]==15])))
			"""
	def upload(self, event):
		self.ax1.cla()
		try:
			self.curve = pick(self.spec, freq=self.freq, velo=self.velo, net=self.net_type_preset, threshold=self.threshold_set, searchStep=self.searchStep,url=self.url)
			show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
			self.ax1.set_title("DisperNet Picked: " + self.fileName)
			if self.flag_plot_or == 1:
				#self.show_or()
				self.ax2.cla()
				show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		except:
			show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
			self.ax1.set_title("Network Connection Error @ " + self.url + "\n Please contact developer -> Dongsh" )
		
		
			
		### Trigger Mode
		if self.trigerMode:
			self.autoDivide(event)
			self.curveInterpButton(event)
			
		plt.draw()
		
	def autoSearchInPeriod(self, event):
		curveInPeriod = []
		
		for pdsInd in range(10, self.pSpec.shape[1]-10, 5):
			pds = self.period[pdsInd]
			flag = 0
			for veloId, velo in enumerate(np.linspace(np.min(self.velo), np.max(self.velo), self.pSpec.shape[0])):
				
				if self.pSpec[veloId, pdsInd] > self.threshold_set and flag == 0:
					flag = 1
					begin_velo = velo
					
				elif flag == 1 and self.pSpec[veloId, pdsInd] <  self.threshold_set:
					mid_velo = begin_velo + (velo - begin_velo)/2
					curveInPeriod.append([1/pds, mid_velo])
					flag = 0
					
		curveInPeriod = np.array(curveInPeriod)
		if len(self.curve) == 0:
			self.curve = curveInPeriod
		else:
			if self.curve.shape[1] == 3:
				curveInPeriod = np.hstack((curveInPeriod, np.zeros([len(curveInPeriod),1])))
				
			self.curve = self.curve[self.curve[:,0] > 1/self.period[15]]
			self.curve = np.vstack([curveInPeriod, self.curve])
			
		self.ax1.cla()
		
		curveP = self.curve.copy()
		if len(curveP) > 0:
			curveP[:, 0] = 1 / curveP[:, 0]
		show(self.pSpec,curveP,r_this = self.r_this,freq=self.period, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, xLabel='Period (s)')
		self.ax1.set_title("Auto Pick in Period: " + self.fileName)
		plt.draw()
		
	def modeDivide(self, event):
		if self.flagDelCurveInMode:
			self.flagDelCurveInMode = False
			self.ax1.set_title('[off] Delete Curve in Mode.')
		
		if self.curve == []:
			self.ax1.set_title('Please UPLOAD or manually pick the curve FIRST!!')
		else:
			self.curve = modeSeparation(self.curve, self.modeNum)
			
			if self.freq_type_preset == 'Period':
				self.axFreqType.cla()
				self.checkFreqType = CheckButtons(self.axFreqType, ['Freq.','Period'], [1, 0])
				self.checkFreqType.on_clicked(self.set_freq_type)
				self.axFreqType.set_title('Disp. Mode')
				self.freq_type_preset = 'Freq.'
				
				self.axUpload.cla()
				self.buttonUpload = Button(self.axUpload, 'Upload to \nDisperNet')
				self.buttonUpload.on_clicked(self.upload)
				
			self.ax1.cla()
			show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
			self.ax1.set_title(self.fileName)
			self.reflashModeLabel()
			if self.flag_plot_or == 1:
				#self.show_or()
				self.ax2.cla()
				show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
			plt.draw()
			
	def autoDivide(self, event):
		if self.curve == []:
			self.ax1.set_title('Please UPLOAD or manually pick the curve FIRST!!')
		else:
			self.curve = sortCurve(self.curve)
			self.curve = autoSeparation(self.curve, maxMode=self.maxMode)
			self.modeNum = int(max(self.curve[:,2])) + 1
			
			if self.freq_type_preset == 'Period':
				self.axFreqType.cla()
				self.checkFreqType = CheckButtons(self.axFreqType, ['Freq.','Period'], [1, 0])
				self.checkFreqType.on_clicked(self.set_freq_type)
				self.axFreqType.set_title('Disp. Mode')
				self.freq_type_preset = 'Freq.'
				
				self.axUpload.cla()
				self.buttonUpload = Button(self.axUpload, 'Upload to \nDisperNet')
				self.buttonUpload.on_clicked(self.upload)
				
			self.ax1.cla()
			show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
			self.ax1.set_title('Auto Divided into '+str(self.modeNum) + ' mode(s)')
			self.reflashModeLabel()
			if self.flag_plot_or == 1:
				#self.show_or()
				self.ax2.cla()
				show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
			plt.draw()
			
	
	def delCurveModeButtomGo(self,event, modeNum,):
		print('delete mode:',modeNum)
		self.curve = del_curve_by_mode(self.curve, modeNum)
		self.ax1.cla()
		show(self.spec,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax1,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		self.ax1.set_title('Deleted Points in Mode '+str(modeNum))
		self.reflashModeLabel()
		if self.flag_plot_or == 1:
			#self.show_or()
			self.ax2.cla()
			show(self.spec_or,self.curve,r_this = self.r_this,freq=self.freq, velo=self.velo, s=15,ax=self.ax2,holdon=True, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, autoT=self.autoT)
		plt.draw()
		
	def mode0ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event,0)
		else:
			self.modeInClick = 0
			self.modeNum = 1
		
	def mode1ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 1)
		else:
			self.modeInClick = 1
			self.modeNum = 2
		
	def mode2ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 2)
		else:
			self.modeInClick = 2
			self.modeNum = 3
		
	def mode3ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 3)
		else:
			self.modeInClick = 3
			self.modeNum = 4
		
	def mode4ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 4)
		else:
			self.modeInClick = 4
			self.modeNum = 5
		
	def mode5ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 5)
		else:
			self.modeInClick = 5
			self.modeNum = 6
		
	def mode6ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 6)
		else:
			self.modeInClick = 6
			self.modeNum = 7
		
	def mode7ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 7)
		else:
			self.modeInClick = 7
			self.modeNum = 8
		
	def mode8ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 8)
		else:
			self.modeInClick = 8
			self.modeNum = 9
		
	def mode9ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 9)
		else:
			self.modeInClick = 9
			self.modeNum = 10
		
	def mode10ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 10)
		else:
			self.modeInClick = 10
			self.modeNum = 11
		
	def mode11ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 11)
		else:
			self.modeInClick = 11
			self.modeNum = 12
		
	def mode12ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 12)
		else:
			self.modeInClick = 12
			self.modeNum = 13
		
	def mode13ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 13)
		else:
			self.modeInClick = 13
			self.modeNum = 14
		
	def mode14ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 14)
		else:
			self.modeInClick = 14
			self.modeNum = 15
		
	def mode15ButtonClick(self, event):
		if self.flagDelCurveInMode:
			self.delCurveModeButtomGo(event, 15)
		else:
			self.modeInClick = 15
			self.modeNum = 16
		

		
	def get_file_list(self, basis_dir="./", begin="", end=""):
		path_list = os.listdir(basis_dir)
		list_final = []
		for partial in path_list:
			if begin and end:
				if partial[:len(begin)] == begin and partial[-len(end):] == end:
					list_final.append(partial)
					
			elif end:
				if partial[-len(end):] == end:
					list_final.append(partial)
					
			elif begin:
				if partial[:len(begin)] == begin:
					list_final.append(partial)
					
			else:
				list_final.append(partial)
				
		return list_final
	
	def natural_sort(self, l): 
		convert = lambda text: int(text) if text.isdigit() else text.lower() 
		alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
		return sorted(l, key = alphanum_key)
	
def help():
	print('Welcome to DisperNet(py)!\n\nThe DisperNet(py) is a tool provides a simple and convenient way to extract the dispersion curve from the spectra automatically. If this is your first time using DisperNet, you should definitely check out the readme.md document firstly.\n\nThe  DisperNet(py) mainly contains the functions:\n\n1. save2h5(spectrum, freq, velo, fileName): save the np.array of dispersion image(spectrum) to a specific h5fs file format with amp, freq and velo information.\n2. pick(spec, threshold, freq, velo, net, errorbar, flipUp, searchStep, searchBorder, returnSpec, ind, url): pick the dispersion curve from the arguments, which need internet connection in SUSTech. This function uploads the data to the online server and fetch the curve based on the spectrum and the extra settings in the arguments.\n3. modeSeparation(curves, modes): separate the picked dispersion curves to different modes, based on locally unsupervised classification(hierarchical clustering analyzation).\n4. show(spec,curve,freq, velo, unit, s, ax, holdon, cmap, vmin, vmax): A simple tool to plot the figure of dispersion spectrum and the curves. \n5. curveInterp(curve, freqSeries): Interpolation of the separated dispersion curve, transfer the curve a smooth and continuous series.\n6. extract(spec, threshold, freq, velo, net, mode,freqSeries ,errorbar, flipUp, searchStep, searchBorder, returnSpec, ind, ur): the function that fuse all the functions above together, if you have decided all the parameters already, this function will promote your code :-)\n\nWe also provide a application with GUI, you can easily launch it by dispernet.App(). \n\nNOT every arguments above are necessary, instead most of them are optional. You can refer to the readme.md for more details.\n\nLastly, we list the optional network type for \'net\' argument:\n 1. noise: For abient noise data from Gaoxiong Wu\'s work. (default)\n 2. event: For earthquakes events data from Zhengbo Li\'s work\n 3. noise2\n 4. noise3\n 5. toLB: transfer learning by Long Beach City data.')

if __name__ == '__main__':
	
	help()
	
	
	
	
	