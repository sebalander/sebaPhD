#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 20:55:46 2017

quantitavely check uncertainty propagation adn error calculation

@author: sebalander
"""

# %%
#import time
#import timeit

import numpy as np
import numdifftools as ndf
from calibration import calibrator as cl
import matplotlib.pyplot as plt

from importlib import reload


# %% LOAD DATA
# input
plotCorners = False
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye']
model = modelos[2]

imagesFolder = "/home/sebalander/Desktop/Code/sebaPhD/resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# %% load data
imagePoints = np.load(cornersFile)
n = len(imagePoints)  # cantidad de imagenes
#indexes = np.arange(n)
#
#np.random.shuffle(indexes)
#indexes = indexes
#
#imagePoints = imagePoints[indexes]
#n = len(imagePoints)  # cantidad de imagenes

chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
#images = glob.glob(imagesFolder+'*.png')

# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)
m = chessboardModel.shape[1]  # cantidad de puntos


# model data files
distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"
# load model specific data
distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)
rVecs = np.load(rVecsFile)#[indexes]
tVecs = np.load(tVecsFile)#[indexes]


# %% poniendo ruido

nSampl = int(1e3)  # cantidad de samples MC

ep = 1e-3 # relative standard deviation in parameters
stdIm = 1e-1 # error in image


# %% test CCD2HOM
# no hay grados de libertad porque es lineal
impts = np.array([[0.0, 0.0]])
Cccd = np.eye(2) * stdIm**2
Cf = np.diag(cameraMatrix[[0,1,0,1],[0,1,2,2]] * ep)**2

# jacobianos


xpp, ypp, Cpp = cl.ccd2hom(impts, cameraMatrix, Cccd=Cccd, Cf=Cf)





















