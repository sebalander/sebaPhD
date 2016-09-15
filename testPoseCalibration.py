# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:57:48 2016

test calibrator class

@author: sebalander
"""


# %%
import cv2
import numpy as np
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

## output files
#rvecOptimDirFile = "./resources/PTZchessboard/zoom 0.0/ptzSheetRvecOptimDir.npy"
#tvecOptimDirFile = "./resources/PTZchessboard/zoom 0.0/ptzSheetTvecOptimDir.npy"
#rvecOptimInvFile = "./resources/PTZchessboard/zoom 0.0/ptzSheetRvecOptimInv.npy"
#tvecOptimInvFile = "./resources/PTZchessboard/zoom 0.0/ptzSheetTvecOptimInv.npy"

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

# %% STEREOGRAPHIC params
linearCoeffs = np.array([1920,1920])/2
distCoeffs = np.array(952) # k calculated by stanganelli?
#
## %% FISHEYE params
#linearCoeffs = np.load(linearCoeffsFile) # coef intrinsecos
#distCoeffs = np.array([[1.1],[2.2],[3.3],[4.4]]) # k1, k2, k3, k4

# %%
model= 'rational' # 'stereographic' 'fisheye', 'unified', 'stereographic'

# %% DIRECT GENERIC CALIBRATION

# direct mapping with initial conditions
cornersProjectedIni = pc.direct(fiducialPoints, rVecIni, tVecIni, linearCoeffs, distCoeffs, model)

# plot corners in image
pc.cornerComparison(img, imageCorners, cornersProjectedIni)

# format parameters
initialParams = pc.formatParameters(rVecIni, tVecIni, linearCoeffs, distCoeffs,model)

# calculate initial residual
initialRes = np.sum(pc.residualDirect(initialParams, fiducialPoints, imageCorners,model)**2)

# optimise rVec, tVec
rVecOpt, tVecOpt, optParams = pc.calibrateDirect(fiducialPoints, imageCorners, rVecIni, tVecIni, linearCoeffs, distCoeffs, model)

# test mapping with optimised conditions
cornersProjectedOpt = pc.direct(fiducialPoints, rVecOpt, tVecOpt, linearCoeffs, distCoeffs, model)
pc.cornerComparison(img, imageCorners, cornersProjectedOpt)

# residual after optimisation
optRes = np.sum(pc.residualDirect(optParams, fiducialPoints, imageCorners,model)**2)

# %% INVERSE GENERIC CALIBRATION

# test mapping with initial conditions
fiducialProjectedIni = pc.inverse(imageCorners, rVecIni, tVecIni, linearCoeffs, distCoeffs,model)

# plot
pc.fiducialComparison(fiducialPoints, fiducialProjectedIni)
pc.fiducialComparison3D(rVecIni, tVecIni, fiducialPoints, fiducialProjectedIni, label1 = 'Fiducial points', label2 = 'Projected points')

# calculate initial residual
initialRes = np.sum(pc.residualInverse(initialParams, fiducialPoints, imageCorners,model)**2)
# plot

# optimise rVec, tVec
rVecOpt, tVecOpt, optParams = pc.calibrateInverse(fiducialPoints, imageCorners, rVecIni, tVecIni, linearCoeffs, distCoeffs,model)

fiducialProjectedOpt = pc.inverse(imageCorners, rVecOpt, tVecOpt, linearCoeffs, distCoeffs,model)
# plot
pc.fiducialComparison(fiducialPoints, fiducialProjectedOpt)
pc.fiducialComparison3D(rVecOpt, tVecOpt, fiducialPoints, fiducialProjectedOpt, label1 = 'Fiducial points', label2 = 'Optimised points')

# residual after optimisation
optRes = np.sum(pc.residualInverse(optParams, fiducialPoints, imageCorners,model)**2)




