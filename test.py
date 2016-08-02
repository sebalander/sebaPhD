# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:30:28 2016

test stuff

test inverse rational function

@author: sebalander
"""
# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

from inverseRational import inverseRational

# %% DATA FILES
imageFile = "./resources/PTZgrid/ptz_(0.850278, -0.014444, 0.0).jpg"
cornersFile = "./resources/PTZgrid/ptzCorners.npy"
patternFile = "./resources/PTZgrid/ptzGridPattern.npy"

distCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
linearCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"

rvecOptimFile = "./resources/PTZchessboard/zoom 0.0/ptzSheetRvecOptim.npy"
tVecOptimFile = "./resources/PTZchessboard/zoom 0.0/ptzSheettVecOptim.npy"

# %% LOAD DATA
img = cv2.imread(imageFile)
corners = np.load(cornersFile)
objectPoints = np.load(patternFile)

distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)

rVec = np.load(rvecOptimFile)
rotMatrix, _ = cv2.Rodrigues(rVec)
tVec = np.load(tVecOptimFile)

# %% test distortion
r = np.linspace(0,10,100)

def distortRadius(r, k):
    '''
    returns distorted radius
    '''
    r2 = r**2
    r4 = r2**2
    r6 = r2*r4
    # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    rd = r * (1 + k[0,0]*r2 + k[1,0]*r4 + k[4,0]*r6) / \
        (1 + k[5,0]*r2 + k[6,0]*r4 + k[7,0]*r6)
    return rd

rd = distortRadius(r, distCoeffs)

plt.plot(r,rd)

# %%
u,v = corners[10,0]

XYZ = inverseRational(corners, cameraMatrix, distCoeffs, rotMatrix, tVec)

# %% PLOT 3D SCENE

[x,y,z] = rotMatrix # get from ortogonal matrix

fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D

ax = fig.gca(projection='3d')

ax.plot([0, tVec[0,0]],
        [0, tVec[1,0]],
        [0, tVec[2,0]])
        
ax.plot([tVec[0,0], tVec[0,0] + x[0]],
        [tVec[1,0], tVec[1,0] + x[1]],
        [tVec[2,0], tVec[2,0] + x[2]])

ax.plot([tVec[0,0], tVec[0,0] + y[0]],
        [tVec[1,0], tVec[1,0] + y[1]],
        [tVec[2,0], tVec[2,0] + y[2]])

ax.plot([tVec[0,0], tVec[0,0] + z[0]],
        [tVec[1,0], tVec[1,0] + z[1]],
        [tVec[2,0], tVec[2,0] + z[2]])

# ax.scatter(XYZ[:,0,0], XYZ[:,0,1], XYZ[:,0,2]) # da muy mal!!!!

ax.scatter(objectPoints[0,:,0],
           objectPoints[0,:,1],
           objectPoints[0,:,2])


#ax.legend()
#ax.set_xlim3d(0, 1)
#ax.set_ylim3d(0, 1)
#ax.set_zlim3d(0, 1)

plt.show()