#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 20:52:00 2017

@author: sebalander
"""



# %%
import cv2
from copy import deepcopy as dc
from calibration import calibrator as cl
from calibration import RationalCalibration as rational

import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from numpy import sqrt, array, isreal, roots
xypToZplane = cl.xypToZplane
formatParameters = rational.formatParameters
retrieveParameters = rational.retrieveParameters

from lmfit import minimize, Parameters


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

# data files
rawDataFile = '/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/calibrExtr/'
imgFile = rawDataFile + 'vcaSnapShot.png'
dawCalibTxt = rawDataFile + 'puntosCalibracion.txt'

# initil pose
tVecIniFile = rawDataFile + 'tVecIni.npy'
rVecIniFile = rawDataFile + 'rVecIni.npy'

# %% load data
ptsCalib = np.loadtxt(dawCalibTxt)
img = cv2.imread(imgFile)

tVecIni = np.load(tVecIniFile)
rVecIni = np.load(rVecIniFile)

# corners in image must have shape (N,1,2)
imagePoints = ptsCalib[:, :2].reshape((-1,2))
# pongo longitud como X y latitud como Y
# points in 3D wolrd must have shape
objectPoints = np.concatenate((ptsCalib[:, 3:1:-1],
                               np.zeros((len(ptsCalib),1)) ),
                               axis=1).reshape((-1,3))

# %% iteratively correct positions to nearest "correct" given the parameters


cl.direct()









