import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas
from mpl_toolkits import mplot3d
from read_and_plot_trajectories import get_qtm_data, get_rift_data
from scale_and_match_trajectories import match_trajectories
qtm_traj_matched, or_traj_matched = match_trajectories(get_qtm_data(), get_rift_data())

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
	for i,_ in enumerate(qdata): #
		q = qdata[i]
		p = pdata[i]
		H += np.outer(q, p) #outer product of q & p

	U, S, Vt = np.linalg.svd(H)
	V = Vt.T

	R = V.dot(U.T)
	t = centroid_p - R.dot(centroid_q) #p=Rq+t

	return R,t

def plot_transformed_trajectories(qdata, pdata): #p=Rq+t
	R, t = get_rotation_and_translation(qdata, pdata)
	
	qdata_transformed = np.zeros_like(qdata) #initiate
	for i,q in enumerate(qdata):
		qdata_transformed[i,:] = R.dot(q) + t

	###Root Mean Square Error ###
	err = np.sum(np.sum((qdata_transformed - pdata)**2)) 
	err_rms = np.sqrt(err/len(pdata));
	print('RMS error: ' + str(round(err_rms, 4)))

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
	ax.plot3D(pdata[:,0], pdata[:,1], pdata[:,2], color='b', label='OR')
	plt.xlabel('x', fontsize=24)
	plt.ylabel('y', fontsize=24)
	plt.legend()
	plt.show()

R, t = get_rotation_and_translation(qtm_traj_matched, or_traj_matched)
filename = 'RotationFromCalibration.txt'
np.savetxt(filename, R)
filename = 'TranslationFromCalibration.txt'
np.savetxt(filename, t)
