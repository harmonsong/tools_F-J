import dispernet_local as dispernet
import numpy as np
import matplotlib.pyplot as plt


filePath = './h5/'
curveFilePath = './curve/'
fileList = dispernet.get_file_list(filePath, end='.h5')

for fileName in fileList:
	
	# read file
	spec, freq, velo = dispernet.readh5(filePath+'/' + fileName)
	
	# pick curve 
	curve = dispernet.pick(spec, freq=freq, velo=velo, threshold=0.5, net='noise', searchStep=2)
	# mode separation
	curve = dispernet.autoSeparation(curve)
	# save curve to file 
	np.savetxt(curveFilePath+'/'+ fileName[:-3] + 'curve.txt', curve, fmt='%.6f  %.6f  %i')
	
	# plot figure
	fig = plt.figure(figsize=[10,5])
	dispernet.show(spec, [], freq, velo, ax=plt.subplot(121),holdon=True)
	dispernet.show(spec, curve, freq, velo, ax=plt.subplot(122),holdon=True)
	plt.show()
	plt.close()

	
	
