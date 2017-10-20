#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 01:12:17 2017

@author: sebalander
"""


# %%
#import glob
import numpy as np
import scipy.linalg as ln
import matplotlib.pyplot as plt
from importlib import reload
from copy import deepcopy as dc

from dev import bayesLib as bl

from multiprocess import Process, Queue
# https://github.com/uqfoundation/multiprocess/tree/master/py3.6/examples

# %% LOAD DATA
# input
plotCorners = False
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye']
model = modelos[2]

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# model data files
distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# model data files
intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
intrinsicParamsOutFile = intrinsicParamsOutFile + "0.5.npy"

# calibration points
calibptsFile = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/calibrExtr/puntosCalibracion.txt"

# load model specific data
distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)

# load intrinsic calib uncertainty
resultsML = np.load(intrinsicParamsOutFile).all()  # load all objects
Nmuestras = resultsML["Nsamples"]
muDist = resultsML['paramsMU']
covarDist = resultsML['paramsVAR']
muCovar = resultsML['paramsMUvar']
covarCovar = resultsML['paramsVARvar']
Ns = resultsML['Ns']
cameraMatrixOut, distCoeffsOut = bl.flat2int(muDist, Ns)

# Calibration points
calibPts = np.loadtxt(calibptsFile)

# according to google earth, a good a priori position is
# -34.629344, -58.370350
y0, x0 = -34.629344, -58.370350
z0 = 15.7 # metros, as measured by oliva



