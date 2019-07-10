import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from read_and_plot_trajectories import *

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,\
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,\
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,\
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def R_from_quaternion(q): #returns rotation 3x3-matrix given a quaternion of the form a + bi + cj + dk
	#q = [a, b, c, d]
	return np.array([[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3]            , 2*q[1]*q[3] + 2*q[0]*q[2]], \
				 	 [2*q[1]*q[2] + 2*q[0]*q[3]            , q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]], \
				 	 [2*q[1]*q[3] - 2*q[0]*q[2]            , 2*q[2]*q[3] + 2*q[0]*q[1]            , q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]])

quaternions_qtm = get_qtm_quaternion_data()
quaternions_or = get_or_quaternion_data()

# q0_qtm = quaternions_qtm[0]
# q0_qtm_inv = np.array([q0_qtm[0], -q0_qtm[1], -q0_qtm[2], -q0_qtm[3]]) #normalized so conjugate is inverse
# q0_or = quaternions_or[0]
# q0_or_inv = np.array([q0_or[0], -q0_or[1], -q0_or[2], -q0_or[3]]) #normalized so conjugate is inverse

# for i,q in enumerate(quaternions_qtm):
# 	quaternions_qtm[i] = quaternion_multiply(q, q0_qtm_inv)
# for i,q in enumerate(quaternions_or):
# 	quaternions_or[i] = quaternion_multiply(q, q0_or_inv)

r_qtm = np.zeros((len(quaternions_qtm), 3)) #initiate list of position 3Vectors
r_or = np.zeros((len(quaternions_or), 3)) #initiate list of position 3Vectors
for i,q in enumerate(quaternions_qtm):
	r_qtm[i] = R_from_quaternion(q).dot(np.array([1, 0, 0]))
for i,q in enumerate(quaternions_or):
	r_or[i] = R_from_quaternion(q).dot(np.array([1, 0, 0]))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(r_qtm[:,0], r_qtm[:,1], r_qtm[:,2], '-r')
ax.plot(r_or[:,0], r_or[:,1], r_or[:,2], '-b')
plt.show()

