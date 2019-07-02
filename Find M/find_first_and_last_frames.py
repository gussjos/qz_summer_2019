import numpy as np
from numpy import linalg

from read_and_plot_trajectories import get_qtm_data
from read_and_plot_trajectories import get_rift_data

first_index = 0
last_index = 0

def find_first_index(trajectory, tol):

    for i in range(3,len(trajectory[0])): # start later than 0 because derivative needs one data point backwards

        central_diff = (trajectory[0][i+1] + trajectory[0][i-1]) / 2*i

        if central_diff > tol:
            print('first index = ' + str(i))
            return i

def find_last_index(trajectory, tol):

    print('ok')


    for i in reversed(range(10,len(trajectory[0])-10)): # -3 to start later than last index

        central_diff = - (trajectory[0][i+1] + trajectory[0][i-1]) / 2*i
        print(central_diff)
        # notice minus sign because we're running backwards

        if central_diff > tol:
            print('last index = ' + str(i))
            return i

qtm_data = get_qtm_data()
rift_data = get_rift_data()
tol = 1

first_index_qtm = find_first_index(qtm_data,tol)
last_index_qtm = find_last_index(qtm_data,tol)

first_index_rift = find_first_index(first_index_rift,tol)
last_index_rift = find_last_index(last_index_rift,tol)


