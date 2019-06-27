import matplotlib.pyplot as plt
import numpy as np
import pandas
from mpl_toolkits import mplot3d

def get_rift_data():

	file = 'ORtracking_PositionAccelerationRotationData.txt'

	df = pandas.read_csv(file)
	df_np = df.to_numpy()

	#print(df_np)

	x = df_np[1200:-1,0]
	y = df_np[1200:-1,1]
	z = df_np[1200:-1,2]
	qx = df_np[:,3]
	qy = df_np[:,4]
	qz = df_np[:,5]
	qw = df_np[:,6]

	return x,y,z

#### 3d-plot ###
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#
#max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
#mid_x = (x.max()+x.min()) * 0.5
#mid_y = (y.max()+y.min()) * 0.5
#mid_z = (z.max()+z.min()) * 0.5
#ax.set_xlim(mid_x - max_range, mid_x + max_range)
#ax.set_ylim(mid_y - max_range, mid_y + max_range)
#ax.set_zlim(mid_z - max_range, mid_z + max_range)
#plt.title('OR trajectory')
#
#ax.plot3D(x,y,z)
#plt.show()
