import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import pandas
from mpl_toolkits import mplot3d
from collections import defaultdict
import sys
#path = sys.path[0] + '/../sample_QTM-OR-ZED_data/'	#if running sample data
path = sys.path[0] + '/data_files/'							#if running like normal

def translate(x,y,z): #translates a list of (x,y,z) coordinates so that the first one is the origin
	x_translated = np.array([u - x[0] for u in x])
	y_translated = np.array([v - y[0] for v in y])
	z_translated = np.array([w - z[0] for w in z])

	return x_translated,y_translated,z_translated

qtm_file = path + 'QTMtracking_PositionRotationData.txt'
or_file = path + 'ORtracking_PositionRotationData.txt'

df_qtm = pandas.read_csv(qtm_file)
df_np_qtm = df_qtm.to_numpy()
df_or = pandas.read_csv(or_file)
df_np_or = df_or.to_numpy()

start_index = 1000 #TODO: make this less bodgy
x_qtm = df_np_qtm[start_index:-1,0]
y_qtm = df_np_qtm[start_index:-1,1]
z_qtm = df_np_qtm[start_index:-1,2]
qx_qtm = df_np_qtm[start_index:-1,3]
qy_qtm = df_np_qtm[start_index:-1,4]
qz_qtm = df_np_qtm[start_index:-1,5]
qw_qtm = df_np_qtm[start_index:-1,6]
x_or = df_np_or[start_index:-1,0]
y_or = df_np_or[start_index:-1,1]
z_or = df_np_or[start_index:-1,2]
qx_or = df_np_or[start_index:-1,3]
qy_or = df_np_or[start_index:-1,4]
qz_or = df_np_or[start_index:-1,5]
qw_or = df_np_or[start_index:-1,6]

R0_or = np.array(Rotation.from_quat([qx_or[0], qy_or[0], qz_or[0], qw_or[0]]).as_dcm())
R0_or_inv = R0_or.T #R is orthogonal so transpose is inverse
R_or = [np.array(Rotation.from_quat([qx_or[i], qy_or[i], qz_or[i], qw_or[i]]).as_dcm()).dot(R0_or_inv) for i,_ in enumerate(qx_or)] #initiate list of Rotation matrices

quat_path = sys.path[0] + '/quaternions/'

# np.savetxt(filename, R_or, fmt='%f', delimiter=",")

R0_qtm = np.array(Rotation.from_quat([qx_qtm[0], qy_qtm[0], qz_qtm[0], qw_qtm[0]]).as_dcm())
R0_qtm_inv = R0_qtm.T #R is orthogonal so transpose is inverse
R_qtm = [np.array(Rotation.from_quat([qx_qtm[i], qy_qtm[i], qz_qtm[i], qw_qtm[i]]).as_dcm()).dot(R0_qtm_inv) for i,_ in enumerate(qx_qtm)] #initiate list of Rotation matrices

#print((R_or)[0][0:3])
#print((R_or)[0][0:3])

#print((R_qtm)[0][0:3])
#print((R_qtm)[0][0:3])

norm_sum = 0
for i in range(1000,1200):

	#print(R_or[i][0:3])

	r_or = Rotation.from_dcm(R_or[i][0:3])
	r_qtm = Rotation.from_dcm(R_qtm[i][0:3])
	r_qtm_to_or = r_qtm.inv()*r_or

	qtm_quat = r_qtm.as_quat()
	or_quat = r_or.as_quat()
	quat_from_qtm_to_rift = r_qtm_to_or.as_quat()

	#print(qtm_quat)
	#print(or_quat)
	print('quat_from_qtm_to_rift = ' + str(quat_from_qtm_to_rift))






x_qtm, y_qtm, z_qtm = translate(x_qtm,y_qtm,z_qtm)
x_or, y_or, z_or = translate(x_or,y_or,z_or)

def get_qtm_pos_data():
	return np.array([x_qtm, y_qtm, z_qtm]).T #returns list of 3vectors as row vectors

def get_or_pos_data():
	return np.array([x_or, y_or, z_or]).T #returns list of 3vectors as row vectors

def get_qtm_orientation_data(): #TODO
	return R_qtm #returns rotation matrix

def get_or_orientation_data(): #TODO
	return R_or #returns rotation matrix

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
	max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
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

