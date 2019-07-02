import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas
from mpl_toolkits import mplot3d
from read_and_plot_trajectories import get_qtm_data, get_rift_data #BODGE
from find_first_and_last_frames import return_indices

def handpalaggning(qtm_traj_var, or_traj_var): #BODGE
	plt.figure()
	plt.plot(qtm_traj_var[:,0] - qtm_traj_var[0][0], 'b', label='x(t) QTM')
	plt.plot(qtm_traj_var[:,1] - qtm_traj_var[0][1], 'g', label='y(t) QTM')
	plt.plot(qtm_traj_var[:,2] - qtm_traj_var[0][2], 'r', label='z(t) QTM')
	plt.plot(or_traj_var[:,0] - or_traj_var[0][0], 'b:', label='x(t) OR')
	plt.plot(or_traj_var[:,1] - or_traj_var[0][1], 'g:', label='y(t) OR')
	plt.plot(or_traj_var[:,2] - or_traj_var[0][2], 'r:', label='z(t) OR')
	plt.legend(fontsize=16)
	plt.show()

#handpalaggning(get_qtm_data(), get_rift_data()) #BODGE

def match_trajectories(qtm_traj, or_traj):#qtm/or_traj should be array with 3d vectors as rows
	qtm_traj = np.array(qtm_traj).T #we need column vectors for this specific method
	or_traj = np.array(or_traj).T

	### BODGE ### TODO: find a way to do this using derivatives

	## importera return_indices

	tol = 0.1 # change if weird indices are found
	I_qtm_first, I_qtm_last = return_indices(qtm_traj,tol)
	print('qtm indices: ' + str(I_qtm_first) + ", " + str(I_qtm_last))

	I_or_first, I_or_last = return_indices(or_traj,tol)
	print('rift indices: ' + str(I_or_first) + ", " + str(I_or_last))

	### remove data before and after actual trajectory ###
	qtm_traj = qtm_traj[:,I_qtm_first:I_qtm_last]
	or_traj = or_traj[:,I_or_first:I_or_last]

	### match x-axii ###
	qtm_traj_matched = [[], [] ,[]] #initiate
	or_traj_matched = [[], [] ,[]] #initiate
	t_index_qtm = np.arange(I_qtm_first,I_qtm_last)
	t_index_or = np.arange(I_or_first,I_or_last)
	t_common = np.linspace(0.01,0.99, num=int(1e4)) #num is the 
	for i in range(0,3):
		f_qtm = interp1d(t_index_qtm, qtm_traj[i], kind='cubic')
		f_or = interp1d(t_index_or, or_traj[i], kind='cubic')

		qtm_traj_matched[i] = f_qtm(t_common*(I_qtm_last-I_qtm_first)+I_qtm_first)
		or_traj_matched[i] = f_or(t_common*(I_or_last-I_or_first)+I_or_first)

	return np.array(qtm_traj_matched).T, np.array(or_traj_matched).T #return list of row vectors

#qtm_data_matched, or_data_matched = match_trajectories(get_qtm_data(), get_rift_data()) #BODGE
#handpalaggning(qtm_data_matched, or_data_matched) #BODGE





