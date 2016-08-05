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
from mpl_toolkits.mplot3d import Axes3D

from rational import inverseRational
from rational import directRational

# % DATA FILES
imageFile = "./resources/PTZgrid/ptz_(0.850278, -0.014444, 0.0).jpg"
cornersFile = "./resources/PTZgrid/ptzCorners.npy"
patternFile = "./resources/PTZgrid/ptzGridPattern.npy"

distCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
linearCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"

rvecOptimFile = "./resources/PTZchessboard/zoom 0.0/ptzSheetRvecOptim.npy"
tVecOptimFile = "./resources/PTZchessboard/zoom 0.0/ptzSheetTvecOptim.npy"

# % LOAD DATA
img = cv2.imread(imageFile)
corners = np.load(cornersFile)
objectPoints = np.load(patternFile)

distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)

rVec = np.load(rvecOptimFile)
tVec = np.load(tVecOptimFile)

# %% PLOT LOADED DATA
plt.imshow(img)
plt.scatter(corners[:,0,0], corners[:,0,1])
plt.show()

[x,y,z], _ = cv2.Rodrigues(rVec) # get from ortogonal matrix

fig = plt.figure()
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
ax.scatter(objectPoints[0,:,0],
           objectPoints[0,:,1],
           objectPoints[0,:,2])
plt.show()


# %% CALCULATE PROJECTIONS

rotMatrix, _ = cv2.Rodrigues(rVec)

cornersProjected = directRational(objectPoints,
                                  rotMatrix,
                                  tVec,
                                  cameraMatrix,
                                  distCoeffs,
                                  plot=True,
                                  img=img,
                                  corners=corners)

cornersProjectedOpenCV, _ = cv2.projectPoints(objectPoints,
                                              rVec,
                                              tVec,
                                              cameraMatrix,
                                              distCoeffs)

objectPointsProjected = inverseRational(corners,
                                        rotMatrix,
                                        tVec,
                                        cameraMatrix,
                                        distCoeffs)


# %% IN IMAGE CHECK DIRECT MAPPING

plt.imshow(img)
plt.plot(corners[:,0,0],
         corners[:,0,1],'o',label='corners')
plt.plot(cornersProjected[:,0,0],
         cornersProjected[:,0,1],'x',label='projected manual')
plt.plot(cornersProjectedOpenCV[:,0,0],
         cornersProjectedOpenCV[:,0,1],'*',label='projected OpenCV')
plt.legend()
plt.show()

# %% PLOT 3D SCENE CHECK INVERSE MAPPING

[x,y,z] = rotMatrix # get from ortogonal matrix

fig = plt.figure()
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

ax.scatter(objectPointsProjected[:,0,0],
           objectPointsProjected[:,0,1],
           objectPointsProjected[:,0,2])

ax.scatter(objectPoints[0,:,0],
           objectPoints[0,:,1],
           objectPoints[0,:,2])

# %%
plt.scatter(objectPoints[0,:,0],objectPoints[0,:,1])
plt.plot(objectPointsProjected[:,0,0],objectPointsProjected[:,0,1],'x')

for i in range(objectPoints.shape[1]):
    plt.plot([objectPoints[0,i,0],objectPointsProjected[i,0,0]],
             [objectPoints[0,i,1],objectPointsProjected[i,0,1]],
             '-k')#,headwidth=0,headlength=0,headaxislength=0)

rError = np.sqrt(np.sum((objectPoints[0,:]-objectPointsProjected[:,0])**2,1))

# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.bar(objectPoints[0,:,0],rError,zs=objectPoints[0,:,1],width=0.1,zdir='y')
#ax.plot_surface(objectPoints[0,:,0],objectPoints[0,:,1],rError)