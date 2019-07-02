import numpy as np
from numpy import linalg

from read_and_plot_trajectories import get_qtm_data
from read_and_plot_trajectories import get_rift_data
from scale_and_match_trajectories import plot_unscaled_trajectory

first_index = 0
last_index = 0

def find_start_index_for_search_by_ocular_inspection(trajectory):

    first_index = 1

    chosen_tuple = plot_unscaled_trajectory(trajectory,first_index)
    start_index_for_search = int(chosen_tuple[0][0])

    return start_index_for_search


def find_last_index_for_search_by_ocular_inspection(trajectory):

    first_index = 0

    chosen_tuple = plot_unscaled_trajectory(trajectory, first_index)
    start_index_for_search = int(chosen_tuple[0][0])

    return start_index_for_search

def find_index(trajectory, start_index_for_search, tol):

    for i in range(start_index_for_search, len(trajectory[0])-1):

        central_diff = (trajectory[0][i+1] + trajectory[0][i-1]) / 2*i

        if central_diff > tol:
            return i


qtm_trajectory = get_qtm_data()
rift_trajectory = get_rift_data()
tol = 10 # located derivative is greater than this tolerance value

### QTM ###

start_index_for_search = find_start_index_for_search_by_ocular_inspection(qtm_trajectory)
final_index_for_search = find_last_index_for_search_by_ocular_inspection(qtm_trajectory)

first_index_qtm = find_index(qtm_trajectory, start_index_for_search, tol)
last_index_qtm = find_index(qtm_trajectory, final_index_for_search, tol)

print('first_index_qtm ' + str(first_index_qtm))
print('last_index_qtm' + str(last_index_qtm))

### Rift ###

#first_index_rift = find_first_index(rift_trajectory,tol)
#last_index_rift = find_last_index(rift_trajectory,tol)


