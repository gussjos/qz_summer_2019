import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

from read_and_plot_trajectories import get_qtm_data
from read_and_plot_trajectories import get_rift_data

first_index = 0
last_index = 0

def plot_unscaled_trajectory(trajectory,first_index):
	plt.figure()
	plt.plot(trajectory[0] - trajectory[0][0], 'b', label='x(t)')
	plt.plot(trajectory[1] - trajectory[1][0], 'g', label='y(t)')
	plt.plot(trajectory[2] - trajectory[2][2], 'r', label='z(t)')
	plt.legend(fontsize=16)
	if first_index == 1:
		plt.title('Please choose starting index',fontsize = 22)
	elif first_index == 0:
		plt.title('Please choose final index',fontsize = 22)
	return plt.ginput(1)

	plt.show()

def find_start_index_for_search_by_ocular_inspection(trajectory):


    choose_first_index = 1 # for title text

    chosen_tuple = plot_unscaled_trajectory(trajectory,choose_first_index)
    start_index_for_search = int(chosen_tuple[0][0])

    return start_index_for_search

def find_last_index_for_search_by_ocular_inspection(trajectory):

    choose_first_index = 0 # for title text

    chosen_tuple = plot_unscaled_trajectory(trajectory, choose_first_index)
    start_index_for_search = int(chosen_tuple[0][0])

    return start_index_for_search

def find_index(trajectory, start_index_for_search, tol):

    for i in range(start_index_for_search, len(trajectory[0])-1):

        central_diff = (trajectory[0][i+1] + trajectory[1][i+1] + trajectory[2][i+1]
                        - trajectory[0][i-1] - trajectory[1][i-1] - trajectory[2][i-1]) / 6*i

        if central_diff > tol:
            return i

def return_indices(trajectory, tol):

    start_index_for_search = find_start_index_for_search_by_ocular_inspection(trajectory)
    final_index_for_search = find_last_index_for_search_by_ocular_inspection(trajectory)

    first_index = find_index(trajectory, start_index_for_search, tol)
    last_index = find_index(trajectory, final_index_for_search, tol)

    return first_index, last_index