#this script tries to compare orientation data for rift & qtm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from read_unity_data import get_qtm_orientation_data, get_or_orientation_data

def get_rotation_and_translation(qdata, pdata):
	"""finds R & t in eq p=Rq+t where p/q-data are lists of row vectors"""

	### define centroids ###
	# centroid_q = np.mean(qdata, axis=0)
	# print(centroid_q)
	# centroid_p = np.mean(pdata, axis=0)
	# qdata = qdata - centroid_q
	# pdata = pdata - centroid_p

	### calculate R ###
	H = np.zeros((3,3)) #initiate
	for i,_ in enumerate(qdata):
		q = qdata[i]
		p = pdata[i]
		H += np.outer(q, p) #outer product of q & p

	u, sigma, vt = np.linalg.svd(H)
	v = vt.T
	R = v.dot(u.T)

	# t = -R.dot(centroid_q) + centroid_p #p=Rq+Us
	t = 0
	return R, t

### initiate some vectors ###
s0 = np.array([1, 0, 0])
R = np.loadtxt("data_files/R_from_calibration.txt") #rotation matrix taking global QTM coordinate system to global OR coordinate system

### define lists of orientation matrices & let them act on s0 ###
R_qtm = [R.dot(tmp) for tmp in get_qtm_orientation_data()] #transform QTM to OR coordinate system
R_or = get_or_orientation_data()
Q = [R_qtm[i].dot(R_or[i].T) for i,_ in enumerate(R_qtm)] #Q[i] = R_qtm[i]/R_or[i]
s_qtm = np.array([R_i.dot(s0) for R_i in R_qtm])
s_or = np.array([R_i.dot(s0) for R_i in R_or])


R_SVD, t = get_rotation_and_translation(s_qtm, s_or)
s_qtm_transformed = np.array([R_SVD.dot(s) for s in s_qtm])
s_quotient = np.array([Q_i.dot(s0) for Q_i in Q])


### draw orientation as curve on sphere ###
fig = plt.figure()
ax = plt.axes(projection="3d")
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 40)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.1)

ax.plot3D(s_qtm[:,0], s_qtm[:,1], s_qtm[:,2], "red", label="s_qtm")
ax.plot3D(s_or[:,0], s_or[:,1], s_or[:,2], "blue", label="s_or")
ax.plot3D(s_quotient[:,0], s_quotient[:,1], s_quotient[:,2], "purple", label="s_quotient")
ax.plot3D(s_qtm_transformed[:,0], s_qtm_transformed[:,1], s_qtm_transformed[:,2], ":r", label="s_qtm_transformed")
plt.legend()
plt.show()







