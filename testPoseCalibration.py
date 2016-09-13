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
import poseCalibration as pc
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

# %% LOAD DATA
img = cv2.imread(imageFile)
imageCorners = np.load(cornersFile)
fiducialPoints = np.load(patternFile)
rVecIni = np.load(rvecInitialFile) # cond inicial de los extrinsecos
tVecIni = np.load(tvecInitialFile)
linearCoeffs = np.load(linearCoeffsFile) # coef intrinsecos
distCoeffs = np.load(distCoeffsFile)

# %%
reload(pc)

# %% DIRECT RATIONAL CALIBRATION
# test format parameters
initialParams = pc.rational.formatParametersRational(rVecIni, tVecIni, linearCoeffs, distCoeffs)
# test retrieve parameters
pc.rational.retrieveParametersRational(initialParams)

# test mapping with initial conditions
pc.rational.directRational(fiducialPoints, rVecIni, tVecIni, linearCoeffs, distCoeffs)

# calculate initial residual
initialRes = np.sum(pc.rational.residualDirectRational(initialParams, fiducialPoints, imageCorners)**2)

# optimise rVec, tVec
rVecOpt, tVecOpt, optParams = pc.rational.calibrateDirectRational(fiducialPoints, imageCorners, rVecIni, tVecIni, linearCoeffs, distCoeffs)

# test mapping with optimised conditions
pc.rational.directRational(fiducialPoints, rVecOpt, tVecOpt, linearCoeffs, distCoeffs)

# residual after optimisation
optRes = np.sum(pc.rational.residualDirectRational(optParams, fiducialPoints, imageCorners)**2)

# %% INVERSE RATIONAL CALIBRATION
# test format parameters
initialParams = pc.rational.formatParametersRational(rVecIni, tVecIni, linearCoeffs, distCoeffs)
# test retrieve parameters
pc.rational.retrieveParametersRational(initialParams)

# test mapping with initial conditions
pc.rational.inverseRational(imageCorners, rVecIni, tVecIni, linearCoeffs, distCoeffs)

# calculate initial residual
initialRes = np.sum(pc.rational.residualInverseRational(initialParams, fiducialPoints, imageCorners)**2)

# optimise rVec, tVec
rVecOpt, tVecOpt, optParams = pc.rational.calibrateInverseRational(fiducialPoints, imageCorners, rVecIni, tVecIni, linearCoeffs, distCoeffs)

# residual after optimisation
optRes = np.sum(pc.rational.residualInverseRational(optParams, fiducialPoints, imageCorners)**2)


# %%
reload(pc)

# %% DIRECT STEREOGRAPHIC CALCULATION
linearCoeffs = np.array([1920,1920])/2
distCoeffs = 952.16 # k calculated by stanganelli

# test format parameters
initialParams = pc.stereographic.formatParametersStereographic(rVecIni, tVecIni, linearCoeffs, distCoeffs)
# test retrieve parameters
pc.stereographic.retrieveParametersStereographic(initialParams)

# test mapping with initial conditions
pc.stereographic.directStereographic(fiducialPoints, rVecIni, tVecIni, linearCoeffs, distCoeffs)

# calculate initial residual
initialRes = np.sum(pc.stereographic.residualDirectStereographic(initialParams, fiducialPoints, imageCorners)**2)

# optimise rVec, tVec
rVecOpt, tVecOpt, optParams = pc.stereographic.calibrateDirectStereographic(fiducialPoints, imageCorners, rVecIni, tVecIni, linearCoeffs, distCoeffs)

# residual after optimisation
optRes = np.sum(pc.stereographic.residualDirectStereographic(optParams, fiducialPoints, imageCorners)**2)

# %%
reload(pc)

# %% INVERSE STEREOGRAPHIC CALCULATION
linearCoeffs = np.array([1920,1920])/2
distCoeffs = 952.16 # k calculated by stanganelli

# test format parameters
initialParams = pc.stereographic.formatParametersStereographic(rVecIni, tVecIni, linearCoeffs, distCoeffs)
# test retrieve parameters
pc.stereographic.retrieveParametersStereographic(initialParams)

# test mapping with initial conditions
pc.stereographic.inverseStereographic(imageCorners, rVecIni, tVecIni, linearCoeffs, distCoeffs)

initialRes = np.sum(pc.stereographic.residualInverseStereographic(initialParams, fiducialPoints, imageCorners)**2)

# optimise rVec, tVec
rVecOpt, tVecOpt, optParams = pc.stereographic.calibrateInverseStereographic(fiducialPoints, imageCorners, rVecIni, tVecIni, linearCoeffs, distCoeffs)

# residual after optimisation
optRes = np.sum(pc.stereographic.residualInverseStereographic(optParams, fiducialPoints, imageCorners)**2)




# %%
reload(pc)

# %% DIRECT STEREOGRAPHIC CALCULATION
linearCoeffs = np.array([1920,1920])/2
distCoeffs = np.array([1, 952]) # l,m calculated by stanganelli?

# test format parameters
initialParams = pc.unified.formatParametersUnified(rVecIni, tVecIni, linearCoeffs, distCoeffs)
# test retrieve parameters
pc.unified.retrieveParametersUnified(initialParams)

# test mapping with initial conditions
pc.unified.directUnified(fiducialPoints, rVecIni, tVecIni, linearCoeffs, distCoeffs)

# calculate initial residual
initialRes = np.sum(pc.unified.residualDirectUnified(initialParams, fiducialPoints, imageCorners)**2)

# optimise rVec, tVec
rVecOpt, tVecOpt, optParams = pc.unified.calibrateDirectUnified(fiducialPoints, imageCorners, rVecIni, tVecIni, linearCoeffs, distCoeffs)

# residual after optimisation
optRes = np.sum(pc.unified.residualDirectUnified(optParams, fiducialPoints, imageCorners)**2)

# %%
reload(pc)

# %% INVERSE STEREOGRAPHIC CALCULATION
linearCoeffs = np.array([1920,1920])/2
distCoeffs = np.array([1, 952]) # l,m calculated by stanganelli?

# test format parameters
initialParams = pc.unified.formatParametersUnified(rVecIni, tVecIni, linearCoeffs, distCoeffs)
# test retrieve parameters
pc.unified.retrieveParametersUnified(initialParams)

# test mapping with initial conditions
pc.unified.inverseUnified(imageCorners, rVecIni, tVecIni, linearCoeffs, distCoeffs)

initialRes = np.sum(pc.unified.residualInverseUnified(initialParams, fiducialPoints, imageCorners)**2)

# optimise rVec, tVec
rVecOpt, tVecOpt, optParams = pc.unified.calibrateInverseUnified(fiducialPoints, imageCorners, rVecIni, tVecIni, linearCoeffs, distCoeffs)

# residual after optimisation
optRes = np.sum(pc.unified.residualInverseUnified(optParams, fiducialPoints, imageCorners)**2)














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
