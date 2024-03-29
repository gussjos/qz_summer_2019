import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from read_unity_data import *
import xml.etree.ElementTree as ET
from tkinter import filedialog

r_qtm = get_qtm_pos_data()
r_or = get_or_pos_data()
rotdata = get_qtm_orientation_data()
U = [S - rotdata[0] for S in rotdata]

def QTM2OR(R, q, U, s):
	d = 1 #1 for just rotation, -1 for rotation & reflection
	D = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]]) #reflection matrix
	return R.dot(D).dot(q + U.dot(s)) # changed from - to +
 
def R_from_quaternion(q):
	"""returns rotation 3x3-matrix given a quaternion of the form [a, b, c, d] ~ a + bi + cj + dk"""
	q = q/np.linalg.norm(q) #only normed quaternions are physically meaningful
	return np.array([[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3]            , 2*q[1]*q[3] + 2*q[0]*q[2]], \
				 	 [2*q[1]*q[2] + 2*q[0]*q[3]            , q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]], \
				 	 [2*q[1]*q[3] - 2*q[0]*q[2]            , 2*q[2]*q[3] + 2*q[0]*q[1]            , q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]])

def obj_fun(params): #params = [sx sy sz a b c d]
	s = np.array(params[0:3])
	R = R_from_quaternion(params[3:7])

	sum = 0
	for i,_ in enumerate(r_qtm):
		sum += np.linalg.norm(r_or[i] - QTM2OR(R, r_qtm[i], U[i], s))**2
	return sum/len(r_qtm) #returns RMS-error

s_guess = np.array([-16.72286907, 41.05006416, 1.69847897])*1e-3 # [m]
quaternion_guess = np.array([0.3363792, 0.0404392, 1.99778822, 0.03380634]) # [a b c d] ~ a + bi + cj + dk
guess = np.append(s_guess, quaternion_guess)
res = minimize(obj_fun, guess, method='COBYLA', tol=1e-6)
s = res['x'][0:3]
quaternion = res['x'][3:7]
R = R_from_quaternion(quaternion)

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

	ax.plot3D(r_qtm[:,0], r_qtm[:,1], r_qtm[:,2], color='r', linestyle=':', label='QTM trajectory')
	ax.plot3D(qdata[:,0], qdata[:,1], qdata[:,2], color='r', label='transformed QTM trajectory')
	ax.plot3D(pdata[:,0], pdata[:,1], pdata[:,2], color='b', label='OR trajectory')
	
	plt.xlabel('x', fontsize=24)
	plt.ylabel('y', fontsize=24)
	plt.legend()
	plt.show()

print(res['message'])
print('RMS error = ' + str(round(res['fun']*1000, 2)) + ' [mm]')
print('s = ' + str(s*1000) + ' [mm]')
print('q = ' + str(quaternion)) 
print('t = ' + str(get_t()) + ' [m]')

r_qtm_transformed = np.array([QTM2OR(R, q, U[i], s) for i,q in enumerate(r_qtm)])
plot_trajectories(r_qtm_transformed, r_or)

input_var = input("save calibration? y/n: ")
if (input_var=='y' or input_var=='Y'):
	np.savetxt('data_files/s_from_calibration.txt', s)
	np.savetxt('data_files/q_from_calibration.txt', quaternion)
	np.savetxt('data_files/R_from_calibration.txt', R)
	np.savetxt('data_files/t_from_calibration.txt', t)

	### edit QTM 6DOF body ###
	filepath_xml = filedialog.askopenfilename(initialdir="C:/Users/gussjo/OneDrive - Zenuity/Documents/someProject", \
												title="Locate the 6DOF file used for this recording", \
												filetypes = (("xml files","*.xml"),("all files","*.*")))
	tree = ET.parse(filepath_xml)
	root = tree.getroot()
	for body in root:
		name = body.find('Name')
		if name.text == 'oculus_rift':
			points = body.find('Points')
			for point in points: # take negative s vector since we're moving the points now
				point.attrib['X'] = str(float(point.attrib['X']) + s[2]*1000) # [mm]
				point.attrib['Y'] = str(float(point.attrib['Y']) + -s[0]*1000) # [mm]
				point.attrib['Z'] = str(float(point.attrib['Z']) + -s[1]*1000) # [mm]
	tree.write(filepath_xml)

	print('Calibration data saved as ' + filepath_xml + '.')