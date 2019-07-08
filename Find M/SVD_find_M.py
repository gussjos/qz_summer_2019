import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas
from mpl_toolkits import mplot3d
#from read_and_plot_trajectories import get_qtm_data, get_rift_data
from read_and_plot_trajectories import get_qtm_pos_data, get_or_pos_data, get_or_orientation_data, get_qtm_orientation_data

def get_rotation_and_translation(qdata, pdata, rotdata): # finds R & t in eq  p=Rq+t where p/qdata are lists of row vectors
	R = np.array([[-0.599147, 0.042707, -0.799499], [-0.042771, 0.995443, 0.085227], [0.799496, 0.085259, -0.594590]]) #from SVD_find_Q

	### define centroids ###
	S0 = rotdata[0]
	U = [S - S0 for S in rotdata]
	UR = [u.dot(R) for u in U]
	Uq = [U[i].T.dot(q) for i,q in enumerate(qdata)]
	URp = [U[i].T.dot(R).dot(p) for i,p in enumerate(pdata)]

	centroid_q = np.linalg.solve(sum(U), sum(Uq))
	#centroid_q = np.mean(qdata, axis=0)
	centroid_p = np.linalg.solve(sum(UR), sum(URp))
	#centroid_p = np.mean(pdata, axis=0)
	qdata = qdata - centroid_q
	pdata = pdata - centroid_p

	### calculate R ###
	H = np.zeros((3,3)) #initiate
	for i,_ in enumerate(qdata):
		q = qdata[i]
		p = pdata[i]
		H += np.outer(q, p) #outer product of q & p

	u, sigma, vt = np.linalg.svd(H)
	v = vt.T
	R = v.dot(u.T)

	### calculate s ###
	sum_URU = np.zeros((3,3)) #initiate
	tmp = np.zeros((3)) #initiate
	for i,S in enumerate(rotdata):
		U = (S - S0)
		sum_URU += U.T.dot(R).dot(U)
		tmp += U.T.dot(pdata[i] - R.dot(qdata[i]))

	# n = len(pdata)
	# s = np.linalg.solve(sum_URU, n*U.dot(centroid_p - R.dot(centroid_q))) #p=Rq+Us

	s = np.linalg.solve(sum_URU, tmp) #p=Rq+Us

	return R, s

def plot_transformed_trajectories(qdata, pdata, rotdata): #p=Rq+t
	R, s = get_rotation_and_translation(qdata, pdata, rotdata)
	
	s =  np.array([ 0.00717889,  0.0079406 , -0.0131751]) #BODGE
	## transform qdata to fit pdata ###
	S0 = rotdata[0]
	qdata_transformed = np.zeros_like(qdata) #initiate
	for i,q in enumerate(qdata):
		S = rotdata[i]
		U = (S - S0)
		qdata_transformed[i] = R.dot(q + U.dot(s))

	### Root Mean Square Error (assumes qdata & pdata matched) ### 
	err = np.sum(np.sum((qdata_transformed - pdata)**2))
	err_rms = np.sqrt(err/len(pdata)) 
	print('RMS error: ' + str(round(err_rms, 4)))

	### 3d-plot ###
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	x = qdata_transformed[:,0]
	y = qdata_transformed[:,1]
	z = qdata_transformed[:,2]
	max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
	mid_x = (x.max() + x.min()) * 0.5
	mid_y = (y.max() + y.min()) * 0.5
	mid_z = (z.max() + z.min()) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)
	plt.title('QTM & OR trajectories')

	ax.plot3D(qdata_transformed[:,0], qdata_transformed[:,1], qdata_transformed[:,2], color='r', label='transformed QTM')
	ax.plot3D(qdata[:,0], qdata[:,1], qdata[:,2], color='r', linestyle=':', label='QTM')
	ax.plot3D(pdata[:,0], pdata[:,1], pdata[:,2], color='b', label='OR')
	
	plt.xlabel('x', fontsize=24)
	plt.ylabel('y', fontsize=24)
	plt.legend()
	plt.show()

qtm_data = get_qtm_pos_data()
or_data = get_or_pos_data()
rotation_data = get_or_orientation_data()
#rotation_data = get_qtm_orientation_data()
plot_transformed_trajectories(qtm_data, or_data, rotation_data)
R, s = get_rotation_and_translation(qtm_data, or_data, rotation_data)
print('s = ' + str(s))
filename = 'RotationFromCalibration.txt'
np.savetxt(filename, R, fmt='%f')
filename = 'TranslationFromCalibration.txt'
np.savetxt(filename, s, fmt='%f')
