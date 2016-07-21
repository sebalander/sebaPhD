# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 20:21:33 2016

generate the camera's pose initial conditions by hand

@author: sebalander
"""
import cv2
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, inv

# %%
tvecInitial = np.array([0,0,2.7])
x = np.array([-8*21, 5*29.7, 0])
y = np.array([5*21, 5*29.7, 0])
z = np.array([0, 1, -2.7])

x /= lin.norm(x)
y /= lin.norm(y)
z /= lin.norm(z)

# make into versor matrix
rMatrix = np.array([x,y,z])

# find nearest 
# http://stackoverflow.com/questions/13940056/orthogonalize-matrix-numpy

rMatrix.dot(inv(sqrtm(rMatrix.T.dot(rMatrix))))

# %%
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.linspace(0, 1, 100)
y = np.sin(x * 2 * np.pi) / 2 + 0.5
ax.plot(x, y, zs=0, zdir='z', label='zs=0, zdir=z')

colors = ('r', 'g', 'b', 'k')
for c in colors:
    x = np.random.sample(20)
    y = np.random.sample(20)
    ax.scatter(x, y, 0, zdir='y', c=c)

ax.legend()
ax.set_xlim3d(0, 1)
ax.set_ylim3d(0, 1)
ax.set_zlim3d(0, 1)

plt.show()