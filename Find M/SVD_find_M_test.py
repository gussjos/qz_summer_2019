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
#tmp0 = qtm_traj_matched[:,0]
#tmp1 = qtm_traj_matched[:,1]
#tmp2 = qtm_traj_matched[:,2]
#qtm_traj_matched[:,0] = tmp1
#qtm_traj_matched[:,1] = -tmp0
#qtm_traj_matched[:,2] = qtm_traj_matched[:,2]

#plt.plot(qtm_traj_matched[:,0], 'r')
#plt.plot(qtm_traj_matched[:,1], 'g')
#plt.plot(qtm_traj_matched[:,2], 'b')
#plt.plot(or_traj_matched[:,0], 'r:')
#plt.plot(or_traj_matched[:,1], 'g:')
#plt.plot(or_traj_matched[:,2], 'b:')
#plt.show()

def get_rotation_and_translation(qdata, pdata): # finds R & t in eq  p=Rq+t where p/qdata are lists of row vectors
	qdata = np.array(qdata)
	pdata = np.array(pdata)

	#define centroids
	centroid_q = np.mean(qdata, axis=0)
	centroid_p = np.mean(pdata, axis=0)

	#transform centroids to origin
	qdata = qdata - centroid_q
	pdata = pdata - centroid_p

	H = np.zeros((3,3)) #initiate
	for i,_ in enumerate(pdata):
		p = pdata[i]
		q = qdata[i]

#		print(np.outer(q.transpose(), p.transpose()))
		H += np.outer(q, p) #outer product of q & p

	U, S, V = np.linalg.svd(H)

	d = np.linalg.det(V.dot(U.T))
	D = np.array([[d, 0, 0], [0, d, 0], [0, 0, 1]])

	R = V.dot(D).dot(U.T)
	t = centroid_p - R.dot(centroid_q) #p=Rq+t

	return R,t


def plot_transformed_trajectories(qdata, pdata): #p=Rq+t
	R, t = get_rotation_and_translation(qdata, pdata)
		

	### BODGE ###
	qdata_transformed = np.zeros_like(qdata)
	for i,q in enumerate(qdata):
		qdata_transformed[i,:] = R.dot(q) + t
#	zed_traj = get_zed_data()

	### 3d-plot ###
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	x = qdata_transformed[:,0]
	y = qdata_transformed[:,1]
	z = qdata_transformed[:,2]
	max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
	mid_x = (x.max()+x.min()) * 0.5
	mid_y = (y.max()+y.min()) * 0.5
	mid_z = (z.max()+z.min()) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)
	plt.title('QTM & OR trajectories')

	ax.plot3D(qdata_transformed[:,0], qdata_transformed[:,1], qdata_transformed[:,2], color='r', label='QTM')
	ax.plot3D(or_traj_matched[:,0], or_traj_matched[:,1], or_traj_matched[:,2], color='b', label='OR')
	#ax.plot3D(zed_traj[0], zed_traj[1], zed_traj[2], color='g', label='ZED')
	plt.xlabel('x', fontsize=24)
	plt.ylabel('y', fontsize=24)
	plt.legend()
	plt.show()


plot_transformed_trajectories(qtm_traj_matched, or_traj_matched) #BODGE