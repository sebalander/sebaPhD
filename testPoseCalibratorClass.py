# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:57:48 2016

test calibrator class

@author: sebalander
"""


# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import poseCalibratorClass as pc
#from lmfit import minimize, Parameters


# % FILES
# input files
imageFile = "./resources/PTZgrid/ptz_(0.850278, -0.014444, 0.0).jpg"
cornersFile = "./resources/PTZgrid/ptzCorners.npy"
patternFile = "./resources/PTZgrid/ptzGridPattern.npy"
rvecInitialFile = "./resources/PTZgrid/PTZsheetRvecInitial.npy"
tvecInitialFile = "./resources/PTZgrid/PTZsheetTvecInitial.npy"

# intrinsic parameters (input)
distCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
linearCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"

# output files
rvecOptimDirFile = "./resources/PTZchessboard/zoom 0.0/ptzSheetRvecOptimDir.npy"
tvecOptimDirFile = "./resources/PTZchessboard/zoom 0.0/ptzSheetTvecOptimDir.npy"
rvecOptimInvFile = "./resources/PTZchessboard/zoom 0.0/ptzSheetRvecOptimInv.npy"
tvecOptimInvFile = "./resources/PTZchessboard/zoom 0.0/ptzSheetTvecOptimInv.npy"

# % LOAD DATA
img = cv2.imread(imageFile)
imageCorners = np.load(cornersFile)
fiducialPoints = np.load(patternFile)
rVecIni = np.load(rvecInitialFile) # cond inicial de los extrinsecos
tVecIni = np.load(tvecInitialFile)
linearCoeffs = np.load(linearCoeffsFile) # coef intrinsecos
distCoeffs = np.load(distCoeffsFile)

# %%
reload(pc)

# %% test mapping
pc.directRational(fiducialPoints, rVecIni, tVecIni, linearCoeffs, distCoeffs)

# %% format parameters
initialParams = pc.formatParameters(rVecIni, tVecIni, linearCoeffs, distCoeffs)
pc.retrieveParameters(initialParams)

# %%
rvecOpt, tvecOpt = pc.calibrateDirectRational(fiducialPoints, imageCorners, rVecIni, tVecIni, linearCoeffs, distCoeffs)

# %% PLOT FOR INITIAL CONDITIONS
cornersX = corners[:,0,0]
cornersY = corners[:,0,1]
prjtCnesXIni = projectedCornersIni[:,0,0]
prjtCnesYIni = projectedCornersIni[:,0,1]
plt.imshow(img)
plt.plot(cornersX, cornersY, 'xr', markersize=10, label='Corners')
plt.plot(prjtCnesXIni, prjtCnesYIni, '+b', markersize=10, label='Proyectados')
plt.legend()
plt.show()


scenePntsX = objectPoints[0,:,0]
scenePntsY = objectPoints[0,:,1]
prjtPntsXIni = projectedPointsIni[0,:,0]
prjtPntsYIni = projectedPointsIni[0,:,1]
plt.plot(scenePntsX, scenePntsY, 'xr', markersize=10, label='pnts Calibracion')
plt.plot(prjtPntsXIni, prjtPntsYIni, '+b', markersize=10, label='Proyectados')
plt.legend()
plt.show()



# %% map with OPTIMAL conditions
projectedCornersOut, jaco = cv2.projectPoints(objectPoints,
                                          rVecOptDir,
                                          tVecOptDir,
                                          linearCoeffs,
                                          distCoeffs)

projectedPointsOut = inverseRational(corners,
                                     rVecOptInv,
                                     tVecOptInv,
                                     linearCoeffs,
                                     distCoeffs)

# %% PLOT FOR FINAL CONDITIONS
prjtCnesXOut = projectedCornersOut[:,0,0]
prjtCnesYOut = projectedCornersOut[:,0,1]
plt.imshow(img)
plt.plot(cornersX, cornersY, 'xr', markersize=10, label='Corners')
plt.plot(prjtCnesXIni, prjtCnesYIni, '+b', markersize=10, label='Proyectados')
plt.plot(prjtCnesXOut, prjtCnesYOut, '+k', markersize=10, label='Optimized proj.')
plt.legend()


prjtPntsXOut = projectedPointsOut[0,:,0]
prjtPntsYOut = projectedPointsOut[0,:,1]
plt.plot(scenePntsX, scenePntsY, 'xr', markersize=10, label='pnts Calibracion')
plt.plot(prjtPntsXIni, prjtPntsYIni, '+b', markersize=10, label='Proyectados')
plt.plot(prjtPntsXOut, prjtPntsYOut, '+k', markersize=10, label='proy, Optimizados')
plt.legend()
plt.show()
