import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas
from mpl_toolkits import mplot3d

from read_and_plot_trajectories import plot_trajectories
from read_and_plot_trajectories import get_qtm_rift_data

import sys
path = sys.path[0] + '/data_files/'

### Get rift and qtm data ###
qtm_file = path + 'QTMtracking_PositionRotationData.txt'
or_file = path + 'ORtracking_PositionRotationData.txt'

df_qtm = pandas.read_csv(qtm_file)
df_np_qtm = df_qtm.to_numpy()
df_rift = pandas.read_csv(or_file)
df_np_rift = df_rift.to_numpy()

qx_qtm = df_np_qtm[:, 3]/df_np_qtm[1, 3]
qy_qtm = df_np_qtm[:, 4]/df_np_qtm[1, 4]
qz_qtm = df_np_qtm[:, 5]/df_np_qtm[1, 5]
qw_qtm = df_np_qtm[:, 6]/df_np_qtm[1, 6]
qx_rift = df_np_rift[:, 3]/df_np_qtm[1, 3]
qy_rift = df_np_rift[:, 4]/df_np_qtm[1, 4]
qz_rift = df_np_rift[:, 5]/df_np_qtm[1, 5]
qw_rift = df_np_rift[:, 6]/df_np_qtm[1, 6]

print(qw_rift = df_np_rift[:, 6])

def plot_quaternion_trajectories():

    ### 3d-plot ###
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    plt.title('qtm and rift quaternion trajectories')

    ax.plot3D(qx_qtm*qw_qtm, qy_qtm*qw_qtm, qz_qtm*qw_qtm, color='r', label='QTM')
    ax.plot3D(qx_rift*qw_rift, qy_rift*qw_rift, qz_rift*qw_rift, color='b', label='rift')
    # ax.plot3D(zed_traj[0], zed_traj[1], zed_traj[2], color='g', label='ZED')
    plt.xlabel('x', fontsize=24)
    plt.ylabel('y', fontsize=24)
    plt.legend()
    plt.show()

plot_quaternion_trajectories()