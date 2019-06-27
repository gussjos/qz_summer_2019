import matplotlib.pyplot as plt
import numpy as np
import pandas
from mpl_toolkits import mplot3d
from collections import defaultdict

def get_qtm_data():

	file = 'QTM_HMD_tracking_20190627_6D.tsv'

	qtm_data = pandas.read_csv(file, sep='\t')
	qtm_data_matrix = qtm_data.to_numpy()
	qtm_data = pandas.read_csv(file, sep='\t')

	x = qtm_data_matrix[:, 0]
	y = qtm_data_matrix[:, 1]
	z = qtm_data_matrix[:, 2]

	nbrFrames = len(qtm_data_matrix)

	## qtm_rot contains the rotational matrix for every frame
	qtm_rot = np.zeros((3,3,nbrFrames))

	for i in range(0,2):
	    for j in range(0, 2):
	        qtm_rot[i,j] = qtm_data_matrix[:,6+i+j] # 5-16 contains rotation elements

	#print(qtm_rot[:,:,0])
	return x,y,z

	## Assign variables from QTM data





### 3d-plot ###
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#
#max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
#mid_x = (x.max()+x.min()) * 0.5
#mid_y = (y.max()+y.min()) * 0.5
#mid_z = (z.max()+z.min()) * 0.5
#ax.set_xlim(mid_x - max_range, mid_x + max_range)
#ax.set_ylim(mid_y - max_range, mid_y + max_range)
#ax.set_zlim(mid_z - max_range, mid_z + max_range)
#plt.title('QTM trajectory')
#
#ax.plot3D(x,y,z)
#plt.show()