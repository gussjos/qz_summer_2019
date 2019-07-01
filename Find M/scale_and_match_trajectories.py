import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pandas
from mpl_toolkits import mplot3d
from collections import defaultdict
import sys
sys.path.insert(0, sys.path[0] + '/../smaple QTM OR ZED data')

from read_and_plot_trajectories import get_qtm_data
from read_and_plot_trajectories import get_rift_data

#### BODGE ###
#qtm_traj = get_qtm_data()
#or_traj = get_rift_data()

def handpalaggning(qtm_traj_var, or_traj_var): #BODGE
	plt.figure()
	plt.plot(qtm_traj_var[0] - qtm_traj_var[0][0], 'b', label='x(t) QTM')
	plt.plot(qtm_traj_var[1] - qtm_traj_var[1][0], 'g', label='y(t) QTM')
	plt.plot(qtm_traj_var[2] - qtm_traj_var[2][2], 'r', label='z(t) QTM')
	plt.plot(or_traj_var[0] - or_traj_var[0][0], 'b:', label='x(t) OR')
	plt.plot(or_traj_var[1] - or_traj_var[1][0], 'g:', label='y(t) OR')
	plt.plot(or_traj_var[2] - or_traj_var[2][0], 'r:', label='z(t) OR')
	plt.legend(fontsize=22)
	plt.show()


#handpalaggning(qtm_traj, or_traj)

def match_trajectories(qtm_traj, or_traj):#qtm/or_traj should be array with 3d vectors as rows
	qtm_traj = np.array(qtm_traj)
	or_traj = np.array(or_traj)

	### BODGE ### TODO: find a way to do this using derivatives
	I_qtm_first = 1000
	I_qtm_last = 1665
	I_or_first = 1419
	I_or_last = 4376
	

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

	return qtm_traj_matched.transpose(), or_traj_matched.transpose()



qtm_traj_matched, or_traj_matched = match_trajectories(get_qtm_data(), get_rift_data())
handpalaggning(qtm_traj_matched, or_traj_matched)

#def object_fun(M): #pseudocode
#	sum = 0 #initiate
#	sum += np.trapz(abs((r_OR - M r_QTM)^2))
#	return sum

#print('p1 = ' + str(qtm_traj_matched[0][4000]) + ', ' + str(qtm_traj_matched[1][4000]) + ', ' + str(qtm_traj_matched[2][4000]))
#print('q1 = ' + str(or_traj_matched[0][4000]) + ', ' + str(or_traj_matched[1][4000]) + ', ' + str(or_traj_matched[2][4000]))

#print('p2 = ' + str(qtm_traj_matched[0][6000]) + ', ' + str(qtm_traj_matched[1][6000]) + ', ' + str(qtm_traj_matched[2][6000]))
#print('q2 = ' + str(or_traj_matched[0][6000]) + ', ' + str(or_traj_matched[1][6000]) + ', ' + str(or_traj_matched[2][6000]))

#print('p3 = ' + str(qtm_traj_matched[0][8000]) + ', ' + str(qtm_traj_matched[1][8000]) + ', ' + str(qtm_traj_matched[2][8000]))
#print('q3 = ' + str(or_traj_matched[0][8000]) + ', ' + str(or_traj_matched[1][8000]) + ', ' + str(or_traj_matched[2][8000]))









