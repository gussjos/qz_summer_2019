import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas
from mpl_toolkits import mplot3d
from collections import defaultdict
#import sys
#path = sys.path[0] + '/../sample_QTM-OR-ZED_data/'	#if running sample data
#path = sys.path[0]									#if running like normal

from read_and_plot_trajectories import get_qtm_data
from read_and_plot_trajectories import get_rift_data
from read_and_plot_trajectories import plot_trajectories #BODGE
from scale_and_match_trajectories import match_trajectories
qtm_traj_matched, or_traj_matched = match_trajectories(get_qtm_data(), get_rift_data())

def get_rotation_and_translation(pdata, qdata): #where p/qdata are lists of row vectors
	pdata = np.array(pdata)
	qdata = np.array(qdata)	

	#define centroids
	centroid_p = np.mean(pdata, axis=0)
	centroid_q = np.mean(qdata, axis=0)

	#transform centroids to origin
	pdata = pdata - centroid_p

	H = np.zeros((3,3)) #initiate
	for i,_ in enumerate(pdata[0]):
		p = pdata[i,:]
		q = qdata[i,:]

		H += np.outer(p,q)

	U, S, V = np.linalg.svd(H)


	R = V.dot(U.transpose())
	t = -R.dot(centroid_p) + centroid_q

	return R,t


def plot_transformed_trajectories(pdata, qdata):
	R, t = get_rotation_and_translation(pdata, qdata)

	### BODGE ###
	pdata_transformed = np.zeros_like(qdata)
	for i,p in enumerate(pdata):
		#print(q)
		pdata_transformed[i,:] = (R.dot(p) + t)
	print(pdata_transformed)
	or_traj = get_rift_data()
#	zed_traj = get_zed_data()

	### 3d-plot ###
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	x = pdata_transformed[:,0]
	y = pdata_transformed[:,1]
	z = pdata_transformed[:,2]
	max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
	mid_x = (x.max()+x.min()) * 0.5
	mid_y = (y.max()+y.min()) * 0.5
	mid_z = (z.max()+z.min()) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)
	plt.title('QTM, OR, & ZED trajectories')

	ax.plot3D(x,y,z, color='r', label='QTM')
	ax.plot3D(or_traj[0], or_traj[1], or_traj[2], color='b', label='OR')
	#ax.plot3D(zed_traj[0], zed_traj[1], zed_traj[2], color='g', label='ZED')
	plt.xlabel('x', fontsize=24)
	plt.ylabel('y', fontsize=24)
	plt.legend()
	plt.show()


plot_transformed_trajectories(qtm_traj_matched, or_traj_matched) #BODGE