# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:57:48 2016

tutorail/examples on poseCalibrator

There are two possible mappings: inverse and direct
  - Direct mapping: converts the scene 3D points to the image 2D pixel points.
        This mapping *distorts* (i.e. "applies the dostortion function") to the
        coordinates.
  - Inverse mapping: takes the 2D distorted coordinates from the image and maps
        to the undistorted scene 3D coordinates. This *un-distorts* the points.

Here we deal with three sets of rototranslation, pose, paramters: user
provided, optimised via direct mapping and optimised via inverse mapping.
  - Initial paramters: proposed by user, wherever he's gotten them from...
  - Direct-optimised parameters: are obtained minimising the cuadratic error
        between the direct-mapping of some 3D fiducial points into the image
        and some 2D corner points that correspond to said fiducial points.
  - Inverse.optimised paramters: The same as direct-optimised parameters, only
        backwards. Are obtained minimising the cuadratic error between the
        inverse-mapping of some 2D corner points into the scene and some 3D
        fiducial points that correspond to said corner points.


In this tutorial we compare all possible combination:

|                   | Direct | Inverse|
|-------------------|--------|--------|
|     Initial       | case 1 | case 4 |
| Direct-optimised  | case 2 | case 5 |
| Inverse-optinised | case 3 | case 6 |


@author: sebalander
"""


# %%
from cv2 import imread
from numpy import load, sum
import poseCalibration as pc

# %% FILES
# input files
dataFolder = "/home/sebalander/Code/sebaPhD/resources/"

imageFile = dataFolder + "PTZgrid/ptz_(0.850278, -0.014444, 0.0).jpg"
cornersFile = dataFolder + "PTZgrid/ptzCorners.npy"
patternFile = dataFolder + "PTZgrid/ptzGridPattern.npy"
rvecInitialFile = dataFolder + "PTZgrid/PTZsheetRvecInitial.npy"
tvecInitialFile = dataFolder + "PTZgrid/PTZsheetTvecInitial.npy"

# intrinsic parameters (input)
distCoeffsFile = dataFolder + "PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
linearCoeffsFile = dataFolder + "PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"

# %% LOAD DATA
img = imread(imageFile)
imageCorners = load(cornersFile)
fiducialPoints = load(patternFile)
rVecIni = load(rvecInitialFile) # cond inicial de los extrinsecos
tVecIni = load(tvecInitialFile)
linearCoeffs = load(linearCoeffsFile) # coef intrinsecos
distCoeffs = load(distCoeffsFile)

# this example works fine with theintrinsic parameters available
# others: 'stereographic' 'fisheye', 'unified'
model= 'rational'

# %% PARAMTER HANDLING
# giving paramters the appropiate format for the optimisation function
paramsIni = pc.formatParameters(rVecIni, tVecIni,
                                linearCoeffs, distCoeffs, model)
# also, retrieving the numerical values
pc.retrieveParameters(paramsIni, model)

# %% OPTIMISATION
# optimise rVec, tVec using direct (distorting) mapping
rVecOptDir, tVecOptDir, paramsOptDir = pc.calibrateDirect(fiducialPoints,
                                                          imageCorners,
                                                          rVecIni,
                                                          tVecIni,
                                                          linearCoeffs,
                                                          distCoeffs,
                                                          model)

# optimise rVec, tVec using inverse (un-distorting) mapping
rVecOptInv, tVecOptInv, paramsOptInv = pc.calibrateInverse(fiducialPoints,
                                                           imageCorners,
                                                           rVecIni,
                                                           tVecIni,
                                                           linearCoeffs,
                                                           distCoeffs,
                                                           model)


# %% CASE 1: use direct mapping to test initial parameters
# project
cornersCase1 = pc.direct(fiducialPoints, rVecIni, tVecIni,
                         linearCoeffs,distCoeffs, model)
# plot to compare
pc.cornerComparison(img, imageCorners, cornersCase1)
# calculate residual
sum(pc.residualDirect(paramsIni, fiducialPoints, imageCorners, model)**2)

# %% CASE 2: use direct mapping to test direct-optimised parameters
# project
cornersCase2 = pc.direct(fiducialPoints, rVecOptDir, tVecOptDir,
                         linearCoeffs, distCoeffs, model)
# plot to compare
pc.cornerComparison(img, imageCorners, cornersCase2)
# calculate residual
sum(pc.residualDirect(paramsOptDir, fiducialPoints, imageCorners, model)**2)

# %% CASE 3: use direct mapping to test inverse-optimised parameters
# project
cornersCase3 = pc.direct(fiducialPoints, rVecOptInv, tVecOptInv,
                         linearCoeffs, distCoeffs, model)
# plot to compare
pc.cornerComparison(img, imageCorners, cornersCase3)
# calculate residual
sum(pc.residualDirect(paramsOptInv, fiducialPoints, imageCorners, model)**2)

# %% CASE 4: use inverse mapping to test initial parameters
# project
fiducialCase4 = pc.inverse(imageCorners, rVecIni, tVecIni,
                           linearCoeffs, distCoeffs, model)
# plot to compare
pc.fiducialComparison(fiducialPoints, fiducialCase4)
pc.fiducialComparison3D(rVecIni, tVecIni,
                        fiducialPoints, fiducialCase4,
                        label1='Fiducial points', label2='Projected points')
# calculate residual
sum(pc.residualInverse(paramsIni, fiducialPoints, imageCorners, model)**2)

# %% CASE 5: use inverse mapping to test direct-optimised parameters
# project
fiducialCase5 = pc.inverse(imageCorners, rVecOptDir, tVecOptDir,
                           linearCoeffs, distCoeffs,model)
# plot to compare
pc.fiducialComparison(fiducialPoints, fiducialCase5)
pc.fiducialComparison3D(rVecOptDir, tVecOptDir,
                        fiducialPoints, fiducialCase5,
                        label1='Fiducial points', label2='Projected points')
# calculate residual
sum(pc.residualInverse(paramsOptDir, fiducialPoints, imageCorners, model)**2)

# %% CASE 6: use inverse mapping to test inverse-optimised parameters
# project
fiducialCase6 = pc.inverse(imageCorners, rVecOptInv, tVecOptInv,
                           linearCoeffs, distCoeffs,model)
# plot to compare
pc.fiducialComparison(fiducialPoints, fiducialCase6)
pc.fiducialComparison3D(rVecOptInv, tVecOptInv,
                        fiducialPoints, fiducialCase6,
                        label1='Fiducial points', label2='Projected points')
# calculate residual
sum(pc.residualInverse(paramsOptInv, fiducialPoints, imageCorners, model)**2)

