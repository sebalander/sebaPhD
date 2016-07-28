# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 20:21:33 2016

generate the camera's pose  conditions by hand

@author: sebalander
"""
# %%
import cv2
import numpy as np
import numpy.linalg as lin
from scipy.linalg import sqrtm, inv
import matplotlib.pyplot as plt

# %% 
tVecFile = "PTZsheetTvecInitial.npy"
rVecFile = "PTZsheetRvecInitial.npy"

# %% Initial TRASLATION VECTOR
tVec = np.array([[0], [0], [2.5]]) 

# %% ROTATION MATRIX
# center of image points to grid point:
center = np.array([3*0.21, 5*0.297, 0])
z = center - tVec[:,0]
z /= lin.norm(z)

# la tercera coordenada no la se, la dejo en cero
x = np.array([6*21, -1*29.7, 0])
y = np.array([-1*21, -7*29.7, 0])

# hacer que x,y sean perp a z, agregar la tercera componente 
x = x - z * np.dot(x,z) # hago perpendicular a z
x /= lin.norm(x) 

y = y - z * np.dot(y,z) # hago perpendicular a z
y /= lin.norm(y)

# %% test ortogonal
np.dot(x,z)
np.dot(y,z)
np.dot(x,y) # ok if not perfectly 0

# %% make into versor matrix
rMatrix = np.array([x,y,z])

# find nearest ortogonal matrix
# http://stackoverflow.com/questions/13940056/orthogonalize-matrix-numpy
rMatrix = rMatrix.dot(inv(sqrtm(rMatrix.T.dot(rMatrix))))

# %% SAVE PARAMETERS

# convert to rodrigues vector
rVec, _ = cv2.Rodrigues(rMatrix)

np.save(tVecFile, tVec)
np.save(rVecFile, rVec)

# %% PLOT VECTORS
[x,y,z] = rMatrix # get from ortogonal matrix
tvec = tVec[:,0]

fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D

ax = fig.gca(projection='3d')

ax.plot([0, tvec[0]], 
        [0, tvec[1]],
        [0, tvec[2]])
        
ax.plot([tvec[0], tvec[0] + x[0]],
        [tvec[1], tvec[1] + x[1]],
        [tvec[2], tvec[2] + x[2]])

ax.plot([tvec[0], tvec[0] + y[0]],
        [tvec[1], tvec[1] + y[1]],
        [tvec[2], tvec[2] + y[2]])

ax.plot([tvec[0], tvec[0] + z[0]],
        [tvec[1], tvec[1] + z[1]],
        [tvec[2], tvec[2] + z[2]])

#ax.legend()
#ax.set_xlim3d(0, 1)
#ax.set_ylim3d(0, 1)
#ax.set_zlim3d(0, 1)

plt.show()