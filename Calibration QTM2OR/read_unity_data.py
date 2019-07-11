import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import pandas
from mpl_toolkits import mplot3d
from collections import defaultdict
import sys
path = sys.path[0] + '/data_files/sample_data/'		#if running sample data
#path = sys.path[0] + '/data_files/'				#if running like normal

def translate(x,y,z): #translates a list of (x,y,z) coordinates so that the first one is the origin
	x_translated = np.array([u - x[0] for u in x])
	y_translated = np.array([v - y[0] for v in y])
	z_translated = np.array([w - z[0] for w in z])

	return x_translated,y_translated,z_translated

qtm_file = path + 'QTMtracking_PositionRotationData.txt'
or_file = path + 'ORtracking_PositionRotationData.txt'

### import data as numpy arrays ###
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

### assume data starts at origin, oriented along coordinate axii ###
R0_or = np.array(Rotation.from_quat([qx_or[0], qy_or[0], qz_or[0], qw_or[0]]).as_dcm())
R0_or_inv = R0_or.T #R is orthogonal so transpose is inverse
R_or = [np.array(Rotation.from_quat([qx_or[i], qy_or[i], qz_or[i], qw_or[i]]).as_dcm()).dot(R0_or_inv) for i,_ in enumerate(qx_or)] #initiate list of Rotation matrices

R0_qtm = np.array(Rotation.from_quat([qx_qtm[0], qy_qtm[0], qz_qtm[0], qw_qtm[0]]).as_dcm())
R0_qtm_inv = R0_qtm.T #R is orthogonal so transpose is inverse
R_qtm = [np.array(Rotation.from_quat([qx_qtm[i], qy_qtm[i], qz_qtm[i], qw_qtm[i]]).as_dcm()).dot(R0_qtm_inv) for i,_ in enumerate(qx_qtm)] #initiate list of Rotation matrices

t = np.array([x_qtm[0], y_qtm[0], z_qtm[0]]) - np.array([x_or[0], y_or[0], z_or[0]])
x_qtm, y_qtm, z_qtm = translate(x_qtm,y_qtm,z_qtm)
x_or, y_or, z_or = translate(x_or,y_or,z_or)

def get_qtm_pos_data():
	"""returns list of 3vectors as row vectors"""
	return np.array([x_qtm, y_qtm, z_qtm]).T

def get_or_pos_data():
	return np.array([x_or, y_or, z_or]).T #returns list of 3vectors as row vectors

def get_qtm_quaternion_data(): #q = a + bi + cj + dk
	return np.array([qw_qtm, qx_qtm, qy_qtm, qz_qtm]).T #returns list of quaternions as row vectors

def get_or_quaternion_data(): #q = a + bi + cj + dk
	return np.array([qw_or, qx_or, qy_or, qz_or]).T #returns list of quaternions as row vectors

def get_qtm_orientation_data():
	return R_qtm #returns list of rotation matrices

def get_or_orientation_data():
	return R_or #returns list of rotation matrices

def get_t():
	return t