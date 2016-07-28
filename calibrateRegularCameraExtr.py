# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:23:21 2016

adjusts extrinsic calibration parameteres using fiducial points
must provide initial conditions

@author: sebalander
"""

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters

# %%
def reform(params):
    # get params into correct format
    rvec = np.zeros((3,1))
    tvec = np.zeros((3,1))
    for i in range(3):
        rvec[i,0] = params['rvec%d'%i].value
        tvec[i,0] = params['tvec%d'%i].value
    
    cameraMatrix = np.zeros((3,3))  
    cameraMatrix[0,0] = params['cameraMatrix0'].value
    cameraMatrix[1,1] = params['cameraMatrix1'].value
    cameraMatrix[0,2] = params['cameraMatrix2'].value
    cameraMatrix[1,2] = params['cameraMatrix3'].value    
    cameraMatrix[2,2] = 1

    distCoeffs = np.zeros((14,1))
    for i in range(14):
        distCoeffs[i,0] = params['distCoeffs%d'%i].value 

    return rvec, tvec, cameraMatrix, distCoeffs


# %% FILES
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
rvecOptimFile = "./resources/PTZchessboard/zoom 0.0/ptzSheetRvecOptim.npy"
tvecOptimFile = "./resources/PTZchessboard/zoom 0.0/ptzSheetTvecOptim.npy"

# %% LOAD DATA
img = cv2.imread(imageFile)
corners = np.load(cornersFile)
objectPoints = np.load(patternFile)
rVecIni = np.load(rvecInitialFile)
tVecIni = np.load(tvecInitialFile)
distCoeffs = np.load(distCoeffsFile)
linearCoeffs = np.load(linearCoeffsFile)

# %% map wit initial conditions
projectedPointsIni, _ = cv2.projectPoints(objectPoints,
                                          rVecIni,
                                          tVecIni,
                                          linearCoeffs,
                                          distCoeffs)


# %% PLOT FOR INITIAL CONDITIONS
imagePntsX = projectedPointsIni[:,0,0]
imagePntsY = projectedPointsIni[:,0,1]

cornersX = corners[:,0,0]
cornersY = corners[:,0,1]

plt.imshow(img)
plt.plot(cornersX, cornersY, 'xr', markersize=10)
plt.plot(imagePntsX, imagePntsY, '+b', markersize=10)

# %% DEFINE CUADRATIC ERROR ON THE IMAGE
# http://cars9.uchicago.edu/software/python/lmfit/intro.html

def residual(params, objectPoints, corners):
    
    rvec, tvec, cameraMatrix, distCoeffs = reform(params)
    
    # project points
    projectedPoints, _ = cv2.projectPoints(objectPoints,
                                     rvec,
                                     tvec,
                                     cameraMatrix,
                                     distCoeffs)
    
    return corners[:,0,:] - projectedPoints[:,0,:]


# %% PARAMETERS
# problem, cant use a matrix or vector as parameter, will have to bypass this
params = Parameters()
for i in range(3):
    params.add('rvec%d'%i, value=rVecIni[i,0], vary=True)
    params.add('tvec%d'%i, value=tVecIni[i,0], vary=True)


params.add('cameraMatrix0', value=linearCoeffs[0,0], vary=False)
params.add('cameraMatrix1', value=linearCoeffs[1,1], vary=False)
params.add('cameraMatrix2', value=linearCoeffs[0,2], vary=False)
params.add('cameraMatrix3', value=linearCoeffs[1,2], vary=False)

for i in range(14):
    params.add('distCoeffs%d'%i, value=distCoeffs[i,0], vary=False)


# %% OPTIMIZE
out = minimize(residual, params, args=(objectPoints, corners))

# %% OPTIMIZED PARAMETERS
rVecOpt, tVecOpt, _ , _ = reform(out.params)

# %% SAVE OPTIM PARAMETERS
np.save(rvecOptimFile, rVecOpt)
np.save(tvecOptimFile, tVecOpt)

# %% map with OPTIMAL conditions
projectedPointsOut, _ = cv2.projectPoints(objectPoints,
                                          rVecOpt,
                                          tVecOpt,
                                          linearCoeffs,
                                          distCoeffs)

# %% PLOT FOR FINAL CONDITIONS
imagePntsX = projectedPointsOut[:,0,0]
imagePntsY = projectedPointsOut[:,0,1]

cornersX = corners[:,0,0]
cornersY = corners[:,0,1]

plt.imshow(img)
plt.plot(cornersX, cornersY, 'xr', markersize=10, label='Corners')
plt.plot(imagePntsX, imagePntsY, '+b', markersize=10, label='Optimized proj.')
plt.legend()