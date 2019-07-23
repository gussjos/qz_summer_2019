import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import pandas
from mpl_toolkits import mplot3d
from collections import defaultdict
import sys
#path = sys.path[0] + '/data_files/sample_data/'		#if running sample data
path = sys.path[0] + '/data_files/'				#if running like normal

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
end_index = -1000
x_qtm = df_np_qtm[start_index:end_index,0]
y_qtm = df_np_qtm[start_index:end_index,1]
z_qtm = df_np_qtm[start_index:end_index,2]
qx_qtm = df_np_qtm[start_index:end_index,3]
qy_qtm = df_np_qtm[start_index:end_index,4]
qz_qtm = df_np_qtm[start_index:end_index,5]
qw_qtm = df_np_qtm[start_index:end_index,6]
x_or = df_np_or[start_index:end_index,0]
y_or = df_np_or[start_index:end_index,1]
z_or = df_np_or[start_index:end_index,2]
qx_or = df_np_or[start_index:end_index,3]
qy_or = df_np_or[start_index:end_index,4]
qz_or = df_np_or[start_index:end_index,5]
qw_or = df_np_or[start_index:end_index,6]

R_or = [np.array(Rotation.from_quat([qx_or[i], qy_or[i], qz_or[i], qw_or[i]]).as_dcm()) for i,_ in enumerate(qx_or)] #initiate list of Rotation matrices
R_qtm = [np.array(Rotation.from_quat([qx_qtm[i], qy_qtm[i], qz_qtm[i], qw_qtm[i]]).as_dcm()) for i,_ in enumerate(qx_qtm)] #initiate list of Rotation matrices

t = np.array([x_qtm[0], y_qtm[0], z_qtm[0]]) - np.array([x_or[0], y_or[0], z_or[0]])
x_qtm, y_qtm, z_qtm = translate(x_qtm,y_qtm,z_qtm)
x_or, y_or, z_or = translate(x_or,y_or,z_or)

def get_qtm_pos_data():
	"""returns list of 3vectors as row vectors"""
	return np.array([x_qtm, y_qtm, z_qtm]).T

def get_or_pos_data():
	"""returns list of 3vectors as row vectors"""
	return np.array([x_or, y_or, z_or]).T

def get_qtm_quaternion_data():
	"""returns list of quaternions as row vectors [a b c d] ~ a + bi + cj + dk"""
	return np.array([qw_qtm, qx_qtm, qy_qtm, qz_qtm]).T

def get_or_quaternion_data():
	"""returns list of quaternions as row vectors [a b c d] ~ a + bi + cj + dk"""
	return np.array([qw_or, qx_or, qy_or, qz_or]).T

def get_qtm_orientation_data():
	"""returns list of rotation matrices"""
	return R_qtm

def get_or_orientation_data():
	"""returns list of rotation matrices"""
	return R_or

def get_t():
	return t