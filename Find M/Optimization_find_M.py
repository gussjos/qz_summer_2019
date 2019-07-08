import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from read_and_plot_trajectories import get_qtm_pos_data, get_or_pos_data, get_or_orientation_data, get_qtm_orientation_data

r_qtm = get_qtm_pos_data()
r_or = get_or_pos_data()
rotdata = get_qtm_orientation_data()
U = [S - rotdata[0] for S in rotdata]

def QTM2OR(R, q, U, s):
	return R.dot(q - U.dot(s))
 
def R_from_quaternion(q): #returns rotation 3x3-matrix given a quaternion of the form a + bi + cj + dk
	#q = [a, b, c, d]
	q = q/np.linalg.norm(q) #BODGE
	return np.array([[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3]            , 2*q[1]*q[3] + 2*q[0]*q[2]], \
				 	 [2*q[1]*q[2] + 2*q[0]*q[3]            , q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]], \
				 	 [2*q[1]*q[3] - 2*q[0]*q[2]            , 2*q[2]*q[3] + 2*q[0]*q[1]            , q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]])


def R_from_euler(theta):#TODO
	return Rotation.from_euler(theta).as_dcm()


def obj_fun(params): #params = [sx sy sz a b c d]
	s = np.array(params[0:3])
	# R = R_from_quaternion(params[3:7])
	R = np.array([[-0.599147, 0.042707, -0.799499], [-0.042771, 0.995443, 0.085227], [0.799496, 0.085259, -0.594590]])

	# R = R_from_euler('zyx', params[3:6])

	sum = 0
	for i,_ in enumerate(r_qtm):
		sum += np.linalg.norm(r_or[i] - QTM2OR(R, r_qtm[i], U[i], s))**2
	return sum/len(r_qtm) #returns RMS-error

s_guess = np.array([-0.05626681,  0.07828984, -0.05447945])
R_guess = np.array([[-0.599147, 0.042707, -0.799499], [-0.042771, 0.995443, 0.085227], [0.799496, 0.085259, -0.594590]])
quaternion_guess = Rotation.from_dcm(R_guess).as_quat()
euler_guess = Rotation.from_dcm(R_guess).as_euler('zyx')
guess = np.append(s_guess, quaternion_guess)
# guess = np.append(s_guess, euler_guess)
res = minimize(obj_fun, guess, method='nelder-mead', tol=1e-6)

print(res)
s = res['x'][0:3]
quaternion = res['x'][3:7]
R = Rotation.from_quat(quaternion).as_dcm()

def plot_trajectories(qdata, pdata):
	### 3d-plot ###
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	x = qdata[:,0]
	y = qdata[:,1]
	z = qdata[:,2]
	max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
	mid_x = (x.max() + x.min()) * 0.5
	mid_y = (y.max() + y.min()) * 0.5
	mid_z = (z.max() + z.min()) * 0.5
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)
	plt.title('QTM & OR trajectories')

	ax.plot3D(r_qtm[:,0], r_qtm[:,1], r_qtm[:,2], color='r', linestyle=':', label='QTM')
	ax.plot3D(qdata[:,0], qdata[:,1], qdata[:,2], color='r', label='QTM transformed')
	ax.plot3D(pdata[:,0], pdata[:,1], pdata[:,2], color='b', label='OR')
	
	plt.xlabel('x', fontsize=24)
	plt.ylabel('y', fontsize=24)
	plt.legend()
	plt.show()

r_qtm_transformed = np.array([QTM2OR(R, q, U[i], s) for i,q in enumerate(r_qtm)])
plot_trajectories(r_or, r_qtm_transformed)

