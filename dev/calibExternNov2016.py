#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:16:43 2017

hacer la calibracion de los datos tomados en nov 2016

@author: sebalander
"""


# %%
import cv2
from copy import deepcopy as dc
from calibration import calibrator as cl
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload



# %% LOAD DATA
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
model = 'rational'

# model files
modelFile = "./resources/intrinsicCalib/" + camera + "/"
distCoeffsFile =   modelFile + camera + model + "DistCoeffs.npy"
cameraMatrixFile = modelFile + camera + model + "LinearCoeffs.npy"
imgShapeFile =     modelFile + camera + "Shape.npy"

# load data
cameraMatrix = np.load(cameraMatrixFile) # coef intrinsecos
distCoeffs = np.load(distCoeffsFile)
imgShape = np.load(imgShapeFile)

# %%
# data files
rawDataFile = '/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/calibrExtr/'
imgFile = rawDataFile + 'vcaSnapShot.png'
dawCalibTxt = rawDataFile + 'puntosCalibracion.txt'

ptsCalib = np.loadtxt(dawCalibTxt)
img = cv2.imread(imgFile)

# corners in image must have shape (N,1,2)
imagePoints = ptsCalib[:, :2].reshape((-1,2))
# pongo longitud como X y latitud como Y
# points in 3D wolrd must have shape
objectPoints = np.concatenate((ptsCalib[:, 3:1:-1],
                               np.zeros((len(ptsCalib),1)) ),
                               axis=1).reshape((-1,3))

# hago deepcopy para que no chille solvePnP
# http://answers.opencv.org/question/1073/what-format-does-cv2solvepnp-use-for-points-in-python/
imagePoints = dc(imagePoints)
objectPoints = dc(objectPoints)
N = imagePoints.shape[0]  # cantidad de puntos

## %% uso la funcion dada por opencv
#
#retval, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints,
#                                  cameraMatrix, distCoeffs)
#
## %%
#objectPointsProjected = cl.inverse(imagePoints, rVec, tVec, cameraMatrix,
#                                   distCoeffs, model)
#
#cl.fiducialComparison3D(rVec, tVec, objectPoints, objectPointsProjected)
#cl.fiducialComparison(objectPoints, objectPointsProjected)
#
#
#
#
## %% USAR OPENCV PERO CON COND INICIALES CONOCIDAS
#
#retval, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints,
#                                  cameraMatrix, distCoeffs,
#                                  rVecIni, tVecIni, useExtrinsicGuess = True)
## NOT WORKING. TRY ANOTHER ALGORITHM
#
## %%
##cv2.SOLVEPNP_EPNP
##cv2.SOLVEPNP_DLS
##cv2.SOLVEPNP_UPNP
#retval, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints,
#                                  cameraMatrix, distCoeffs,
#                                  flags=cv2.SOLVEPNP_EPNP)
#
#retval, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints,
#                                  cameraMatrix, distCoeffs,
#                                  flags=cv2.SOLVEPNP_DLS)
#
#retval, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints,
#                                  cameraMatrix, distCoeffs,
#                                  flags=cv2.SOLVEPNP_UPNP)



## %%
#objectPointsProjected = cl.inverse(imagePoints, rVec, tVec, cameraMatrix,
#                                   distCoeffs, model)
#
#cl.fiducialComparison3D(rVec, tVec, objectPoints, objectPointsProjected)
#cl.fiducialComparison(objectPoints, objectPointsProjected)





## %% PARAMTER HANDLING
## tweaking linear params so we get correct units
#linearCoeffs = load(linearCoeffsFile) # coef intrinsecos
#distCoeffs = load(distCoeffsFile)
#
#
## giving paramters the appropiate format for the optimisation function
#paramsIni = pc.formatParameters(rVecIni, tVecIni,
#                                linearCoeffs, distCoeffs, model)
## also, retrieving the numerical values
#pc.retrieveParameters(paramsIni, model)
#
## %% CASE 1: use direct mapping to test initial parameters
## project
#cornersCase1 = pc.direct(fiducialPoints, rVecIni, tVecIni,
#                         linearCoeffs, distCoeffs, model)
## plot to compare
#pc.cornerComparison(img, imageCorners, cornersCase1)
## calculate residual
#sum(pc.residualDirect(paramsIni, fiducialPoints, imageCorners, model)**2)

