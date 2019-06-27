import matplotlib.pyplot as plt
import numpy as np
import pandas
from mpl_toolkits import mplot3d
from collections import defaultdict
import sys
sys.path.insert(0, sys.path[0] + '/../smaple QTM OR ZED data')

from read_and_plot_trajectories import get_qtm_data
from read_and_plot_trajectories import get_rift_data

asd = get_qtm_data
asdf = get_rift_data
