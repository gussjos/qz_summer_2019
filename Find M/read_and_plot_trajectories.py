import matplotlib.pyplot as plt
import numpy as np
import pandas
from mpl_toolkits import mplot3d
from collections import defaultdict
import sys
#path = sys.path[0] + '/../sample_QTM-OR-ZED_data/'	#if running sample data
path = sys.path[0] + '/'							#if running like normal

def translate(x,y,z): #translates a list of (x,y,z) coordinates so that the first one is the origin
	x_translated = np.array([u - x[0] for u in x])
	y_translated = np.array([v - y[0] for v in y])
	z_translated = np.array([w - z[0] for w in z])

	return x_translated,y_translated,z_translated
	
def get_qtm_data(): #depracated
	file = path + 'QTMtracking_PositionRotationData.txt'
	df = pandas.read_csv(file)
	df_np = df.to_numpy()

	x = df_np[:,0]
	y = df_np[:,1]
	z = df_np[:,2]
	qx = df_np[:,3]
	qy = df_np[:,4]
	qz = df_np[:,5]
	qw = df_np[:,6]

	#x, y, z = translate(x,y,z) #ONLY FOR VISUAL COMPARISON BEFORE CALIBRATION
	return np.array([x, y, z]).T #returns list of 3vectors ad row vectors

def get_zed_data(): #depracated

	file = path + 'zed_pose_data.txt'

	df = pandas.read_csv(file)
	df_np = df.to_numpy()

	x = df_np[:,0]
	y = df_np[:,1]
	z = df_np[:,2]
	qx = df_np[:,3]
	qy = df_np[:,4]
	qz = df_np[:,5]
	qw = df_np[:,6]

	#x, y, z = translate(x,y,z) #ONLY FOR VISUAL COMPARISON BEFORE CALIBRATION
	return np.array([x, y, z]).T #returns list of 3vectors ad row vectors

def get_rift_data(): #depracated 

	file = path + 'ORtracking_PositionAccelerationRotationData.txt'

	df = pandas.read_csv(file)
	df_np = df.to_numpy()

	x = df_np[:,0]
	y = df_np[:,1]
	z = df_np[:,2]
	qx = df_np[:,3]
	qy = df_np[:,4]
	qz = df_np[:,5]
	qw = df_np[:,6]

	### Remove all the (0.0, 0.0, 0.0):s ###
	index_array = [] #initiate
	for index,_ in enumerate(x):
		if (x[index]==0.0) * (y[index]==0.0) * (z[index]==0.0):
			index_array.append(index)


	x = np.delete(x,index_array)
	y = np.delete(y,index_array)
	z = np.delete(z,index_array)

	#x, y, z = translate(x,y,z) #ONLY FOR VISUAL COMPARISON BEFORE CALIBRATION
	return np.array([x, y, z]).T #returns list of 3vectors ad row vectors

def get_qtm_rift_data(): #TODO: make more concise
	qtm_file = path + 'QTMtracking_PositionRotationData.txt'
	or_file = path + 'ORtracking_PositionAccelerationRotationData.txt'

	df_qtm = pandas.read_csv(qtm_file)
	df_np_qtm = df_qtm.to_numpy()
	df_or = pandas.read_csv(or_file)
	df_np_or = df_or.to_numpy()

	x_qtm = df_np_qtm[:,0]
	y_qtm = df_np_qtm[:,1]
	z_qtm = df_np_qtm[:,2]
	qx_qtm = df_np_qtm[:,3]
	qy_qtm = df_np_qtm[:,4]
	qz_qtm = df_np_qtm[:,5]
	qw_qtm = df_np_qtm[:,6]
	x_or = df_np_or[:,0]
	y_or = df_np_or[:,1]
	z_or = df_np_or[:,2]
	qx_or = df_np_or[:,3]
	qy_or = df_np_or[:,4]
	qz_or = df_np_or[:,5]
	qw_or = df_np_or[:,6]

	### Remove all the (0.0, 0.0, 0.0):s ###
	index_array = [] #initiate
	for index,_ in enumerate(x_or):
		if (x_or[index]==0.0) * (y_or[index]==0.0) * (z_or[index]==0.0):
			index_array.append(index)

	x_or = np.delete(x_or,index_array)
	y_or = np.delete(y_or,index_array)
	z_or = np.delete(z_or,index_array)
	x_qtm = np.delete(x_qtm,index_array)
	y_qtm = np.delete(y_qtm,index_array)
	z_qtm = np.delete(z_qtm,index_array)

	start_index = 1000 #remove first N frames
	x_or = x_or[start_index:-1]
	y_or = y_or[start_index:-1]
	z_or = z_or[start_index:-1]
	x_qtm = x_qtm[start_index:-1]
	y_qtm = y_qtm[start_index:-1]
	z_qtm = z_qtm[start_index:-1]

	#x, y, z = translate(x,y,z) #ONLY FOR VISUAL COMPARISON BEFORE CALIBRATION
	return np.array([x_qtm, y_qtm, z_qtm]).T, np.array([x_or, y_or, z_or]).T #returns list of 3vectors ad row vectors

def plot_trajectories(): #TODO: plot_trajectories(qtm_traj, or_traj, zed_traj) makes more sense

	qtm_traj = get_qtm_data().T
	or_traj = get_rift_data().T
	zed_traj = get_zed_data().T

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




