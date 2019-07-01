import numpy as np
from numpy import linalg

p1 = np.array([0.013849792670141492, 0.037808234669644704, 1.0084932462336589])
q1 = np.array([0.08720056619036007, 1.0228535481550693, 0.005660490767391328])
p2 = np.array([-0.11619744404668322, 0.8338395577844054, 0.9208249874837462])
q2 = np.array([0.4166173894577574, 0.9041515981380953, 0.7412018000928503])
p3 = np.array([-0.07347052396756322, 0.08244689438131277, 1.0256986393318832])
q3 = np.array([0.040464792337801966, 1.048070735271747, 0.09690475540942207])


#p1 = np.array([1, 2])
#p2 = np.array([2, 1])
#p3 = np.array([1, 1])

#p_data = [p1, p2, p3]

#t = np.array([100, 0])
#M = np.array([[0, 1], [-1, 0]])
#q_data = [[], [], []] #initiate
#for i,p in enumeratye(p_data):
#	q = M.dot(p) + t
#	q_data[i] = q


#define centroids
centroid_p = (p1 + p2 + p3)/3
centroid_q = (q1 + q2 + q3)/3

#transform centroids to origin
p1 = p1 - centroid_p
p2 = p2 - centroid_p
p3 = p3 - centroid_p
q1 = q1 - centroid_q
q2 = q2 - centroid_q
q3 = q3 - centroid_q


H = np.outer(p1,q1) + np.outer(p2,q2) + np.outer(p3,q3)

U, S, V = np.linalg.svd(H)


R = V.dot(U.transpose())
t = -R.dot(centroid_p) + centroid_q

print(R)
print(t)

