import matplotlib.pyplot as plt
import numpy as np
import pandas
from mpl_toolkits import mplot3d
from collections import defaultdict
import sys
#sys.path.insert(0, sys.path[0] + '/../smaple QTM OR ZED data')

def translate(x,y,z): #translates a list of (x,y,z) coordinates so that the first one is the origin
	x_translated = np.array([u - x[0] for u in x])
	y_translated = np.array([v - y[0] for v in y])
	z_translated = np.array([w - z[0] for w in z])

	return x_translated,y_translated,z_translated


def get_qtm_data():

	file = 'QTM_HMD_tracking_20190627_6D.tsv'

	qtm_data = pandas.read_csv(file, sep='\t')
	qtm_data_matrix = qtm_data.to_numpy()
	qtm_data = pandas.read_csv(file, sep='\t')

	x = qtm_data_matrix[:, 0]/1000 #convert from mm to m
	y = qtm_data_matrix[:, 1]/1000 #convert from mm to m
	z = qtm_data_matrix[:, 2]/1000 #convert from mm to m

	nbrFrames = len(qtm_data_matrix)

	## qtm_rot contains the rotational matrix for every frame
	qtm_rot = np.zeros((3,3,nbrFrames))

	for i in range(0,2):
	    for j in range(0, 2):
	        qtm_rot[i,j] = qtm_data_matrix[:,6+i+j] # 5-16 contains rotation elements

	x, y, z = translate(x,y,z) #ONLY FOR VISUAL COMPARISON BEFORE CALIBRATION
	return x, y, z

	
def get_zed_data():

	file = 'zed_pose_data.txt'

	df = pandas.read_csv(file)
	df_np = df.to_numpy()

	x = df_np[:,0]
	y = df_np[:,1]
	z = df_np[:,2]
	qx = df_np[:,3]
	qy = df_np[:,4]
	qz = df_np[:,5]
	qw = df_np[:,6]

	x, y, z = translate(x,y,z) #ONLY FOR VISUAL COMPARISON BEFORE CALIBRATION
	return x, y, z


def get_rift_data():

	file = 'ORtracking_PositionAccelerationRotationData.txt'

	df = pandas.read_csv(file)
	df_np = df.to_numpy()

	x = df_np[:,0]
	y = df_np[:,1]
	z = df_np[:,2]
	qx = df_np[:,3]
	qy = df_np[:,4]
	qz = df_np[:,5]
	qw = df_np[:,6]

	i_array=[]
	for i,_ in enumerate(x): #removes all (0,0,0):s
		if (x[i]==0.0) * (y[i]==0.0) * (z[i]==0.0):
			i_array.append(i)

	x = np.delete(x,i_array)
	y = np.delete(y,i_array)
	z = np.delete(z,i_array)


	### translate everything to the origin ###
	x_translated = np.array([u - x[0] for u in x])
	y_translated = np.array([v - y[0] for v in y])
	z_translated = np.array([w - z[0] for w in z])

	return z_translated,-x_translated,y_translated #according to QTMS:s coordinate system





def plotTrajectories():

	### Remove all the (0.0, 0.0, 0.0):s ###
	index_array = [] #initiate
	for index,_ in enumerate(x):
		if (x[index]==0.0) * (y[index]==0.0) * (z[index]==0.0):
			index_array.append(index)


	x = np.delete(x,index_array)
	y = np.delete(y,index_array)
	z = np.delete(z,index_array)

	x, y, z = translate(x,y,z) #ONLY FOR VISUAL COMPARISON BEFORE CALIBRATION
	return x, y, z


def plot_trajectories():

	qtm_traj = get_qtm_data()
	or_traj = get_rift_data()
	zed_traj = get_zed_data()

	### 3d-plot ###
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	x = qtm_traj[0]
	y = qtm_traj[1]
	z = qtm_traj[2]
	max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
	mid_x = (x.max()+x.min()) * 0.5
	mid_y = (y.max()+y.min()) * 0.5
	mid_z = (z.max()+z.min()) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)
	plt.title('QTM, OR, & ZED trajectories')

	ax.plot3D(qtm_traj[0],qtm_traj[1],qtm_traj[2], color='r', label='QTM')
	ax.plot3D(or_traj[0], or_traj[1], or_traj[2], color='b', label='OR')
	#ax.plot3D(zed_traj[0], zed_traj[1], zed_traj[2], color='g', label='ZED')
	plt.xlabel('x', fontsize=24)
	plt.ylabel('y', fontsize=24)

	plt.legend()
	plt.show()








