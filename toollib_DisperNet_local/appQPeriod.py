import dispernet_local_latest as dispernet
import numpy as np
import os
import matplotlib.pyplot as plt


workPath = '/23SepPick/'
filePath = workPath + 'grid_Vor300'
curveFilePath = workPath + 'grid_Vor300_curve3/'
threshold = 0.9
freqSeries = np.arange(0,100)/100


def get_file_list(basis_dir="./", begin="", end=""):
		path_list = os.listdir(basis_dir)
		list_final = []
		for partial in path_list:
			if begin and end:
				if partial[:len(begin)] == begin and partial[-len(end):] == end:
					list_final.append(partial)
					
			elif end:
				if partial[-len(end):] == end and partial[0] != '.':
					list_final.append(partial)
			
			elif begin:
				if partial[:len(begin)] == begin:
					list_final.append(partial)
					
			else:
				list_final.append(partial)
				
		return list_final
		
def curve_save(curve, fileName, curveFilePath='./'):
		if curve == [] or np.array(curve).ndim < 2:
			print('No curves to save yet.')
		else:
			if len(curve) > 1:
				if curve.shape[1] !=2 and  curve.shape[1] !=4:
					curve = curve[np.argsort(curve[:,-1])]
					for mode in range(int(max(curve[:,-1])+1)):
						curveInMode = curve[curve[:,-1] == mode]					
						curve[curve[:,-1] == mode] = curveInMode[np.argsort(curveInMode[:,0])]
			
			if curve.shape[1] > 2:		
				np.savetxt(curveFilePath + fileName[:-3] + 'curve.txt', curve, fmt='%.6f  %.6f  %i')
			else:
				np.savetxt(curveFilePath + fileName[:-3] + 'curve.txt', curve, fmt='%.6f  %.6f')
			print('Curve file saved. ('+str(len(curve)) + ' points)')


		

h5FileList = get_file_list(filePath, end='.h5')


ii = 0

modeImage = []

for fileName in h5FileList:
	
	ii +=1
	
#	print(fileName[3:7])
	print(fileName)
	

		
	fileFullPath = os.path.join(filePath, fileName)
	
	spec, freq, velo = dispernet.readh5(fileFullPath)
	
	spec[np.isnan(spec)] = 0
	
	if np.sum(spec) == 0:
		continue

	threshold_set=threshold
	
	curve = []
	
	pSpec, period = dispernet.freq2Period(spec, freq, cutRate=0.1)
	pSpec = pSpec/np.max(pSpec)
	print(pSpec.shape)
	curveInPeriod = []

	for pdsInd in range(len(period)):
		pds = period[pdsInd]
		flag = 0
		for veloId, veloValue in enumerate(np.linspace(np.min(velo), np.max(velo), pSpec.shape[0])):
			
			if pSpec[veloId, pdsInd] > threshold_set and flag == 0:
				flag = 1
				begin_velo = veloValue
				
			elif flag == 1 and pSpec[veloId, pdsInd] <  threshold_set:
				mid_velo = begin_velo + (veloValue - begin_velo)/2
#				curveInPeriod.append([1/pds, begin_velo, 0])
				curveInPeriod.append([1/pds, mid_velo])
#				curveInPeriod.append([1/pds, veloValue,2])
				flag = 0
				
				
	curveInPeriod = np.array(curveInPeriod)
	if len(curve) == 0:
		curve = curveInPeriod
	else:
		if curve.shape[1] == 3:
			curveInPeriod = np.hstack((curveInPeriod, np.zeros([len(curveInPeriod),1])))
			
		curve = curve[curve[:,0] > 0.1]
		curve = np.vstack([curveInPeriod, curve])
		
	curve_save(curve, fileName , curveFilePath)
		
#	
#	print(curve)
#	curveP = curve.copy()
#	if len(curveP) > 0:
#		curveP[:, 0] = 1 / curveP[:, 0]
#	
#	plt.figure(figsize=(10,5))
#	ax1 = plt.subplot(121)
#	dispernet.show(spec,curve, freq, velo, ax=ax1, holdon=True, cmap='jet')
#	
#	
#
#	ax2 = plt.subplot(122)
#	
#	dispernet.show(pSpec, curveP,freq=period, velo=velo, xLabel='Period (s)', ax=ax2)
#	plt.show()
#	

	
	