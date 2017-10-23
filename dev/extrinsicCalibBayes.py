#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 01:12:17 2017

@author: sebalander
"""


# %%
#import glob
import numpy as np
#import scipy.linalg as ln
import matplotlib.pyplot as plt
#from importlib import reload
#from copy import deepcopy as dc
#import pyproj as pj

from dev import bayesLib as bl
from calibration import lla2eceflib as llef

#from multiprocess import Process, Queue
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
lats = calibPts[:,2]
lons = calibPts[:,3]

# according to google earth, a good a priori position is
# -34.629344, -58.370350
y0, x0 = -34.629344, -58.370350
z0 = 15.7 # metros, as measured by oliva

# EGM96 sobre WGS84: https://geographiclib.sourceforge.io/cgi-bin/GeoidEval?input=-34.629344%2C+-58.370350&option=Submit
geoideval = 16.1066
elev = 7 # altura msnm en parque lezama segun google earth
h = elev + geoideval # altura del terreno sobre WGS84

# lat lon alt of the data points
hvec = np.zeros_like(calibPts[:,3]) + h
LLA = np.concatenate((calibPts[:,2:], hvec.reshape((-1,1))), axis=1)
LLA0 = np.array([y0, x0, h]) # frame of reference origin
LLAcam = np.array([y0, x0, h + z0]) # a priori position of camera


# %%
xyzPts = llef.lla2ltp(LLA, LLA0)
xyzCam = llef.lla2ltp(LLAcam, LLA0)

plt.plot(xyzPts[:,0], xyzPts[:,1], '+')
plt.plot(xyzCam[0], xyzCam[1], 'x')


