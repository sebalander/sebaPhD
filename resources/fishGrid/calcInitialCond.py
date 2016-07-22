# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 20:21:33 2016

generate the camera's pose initial conditions by hand

@author: sebalander
"""
import cv2
import numpy as np
import numpy.linalg as lin
from scipy.linalg import sqrtm, inv

# %% 
tVecFile = "./resources/fishGrid/FEsheetTvecInitial.npy"
rVecFile = "./resources/fishGrid/FEsheetRvecInitial.npy"

# %% INITIAL TRASLATION VECTOR
tVec = np.array([[0], [0], [2.7]]) 

# %% ROTATION MATRIX
z = np.array([0, 1, -2.7])
z /= lin.norm(z)

# la tercera coordenada no la se, la dejo en cero
x = np.array([-8*21, 5*29.7, 0])
y = np.array([5*21, 5*29.7, 0])

# hacer que x,y sean perp a z, agregar la tercera componente 
x = x - z * np.dot(x,z) # hago perpendicular a z
x /= lin.norm(x) 

y = y - z * np.dot(y,z) # hago perpendicular a z
y /= lin.norm(y)

# %% test ortogonal
np.dot(x,z)
np.dot(x,y)
np.dot(y,z)

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

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot([0, tvecInitial[0]], 
        [0, tvecInitial[1]],
        [0, tvecInitial[2]])
        
ax.plot([tvecInitial[0], tvecInitial[0] + x[0]],
        [tvecInitial[1], tvecInitial[1] + x[1]],
        [tvecInitial[2], tvecInitial[2] + x[2]])

ax.plot([tvecInitial[0], tvecInitial[0] + y[0]],
        [tvecInitial[1], tvecInitial[1] + y[1]],
        [tvecInitial[2], tvecInitial[2] + y[2]])

ax.plot([tvecInitial[0], tvecInitial[0] + z[0]],
        [tvecInitial[1], tvecInitial[1] + z[1]],
        [tvecInitial[2], tvecInitial[2] + z[2]])

#ax.legend()
#ax.set_xlim3d(0, 1)
#ax.set_ylim3d(0, 1)
#ax.set_zlim3d(0, 1)

plt.show()