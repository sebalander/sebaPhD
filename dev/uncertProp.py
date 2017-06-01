#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:46:50 2017

@author: sebalander
"""
# %%
import numpy as np
import glob
from calibration import calibrator as cl
import matplotlib.pyplot as plt
from importlib import reload
import scipy.linalg as ln
from numpy import sqrt, cos, sin


# %% LOAD DATA
# input
plotCorners = False
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye']
model = modelos[1]

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# load data
imagePoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')

nIm, _, nPts, _ = imagePoints.shape  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*nIm)

distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# load model specific data
distCoeffs = np.load(distCoeffsFile).reshape(-1)
cameraMatrix = np.load(linearCoeffsFile)
rVecs = np.load(rVecsFile)
tVecs = np.load(tVecsFile)

# covarianza de los parametros intrinsecos de la camara
Cf = np.eye(4)*(0.1)**2
Ck = np.diag((distCoeffs[[0,1,4,5,6,7]]*0.001)**2)  # 0.1% error distorsion

# %% choose image
j = 2
print('\t imagen', j)
imagePoints = np.load(cornersFile)
imagePoints = imagePoints[j,0]

img = plt.imread(images[j])

# covariances
Cccd = (0.5**2) * np.array([np.eye(2)]*imagePoints.shape[0])

# plot initial uncertanties
fig = plt.figure()
ax = fig.gca()
ax.imshow(img)
for i in range(nPts):
    x, y = imagePoints[i]
    cl.plotEllipse(ax, Cccd[i], x, y, 'b')

#

# %% propagate to homogemous

xpp, ypp, Cpp = cl.ccd2hom(imagePoints, cameraMatrix, Cccd=Cccd, Cf=Cf)

fig = plt.figure()
ax = fig.gca()
for i in range(nPts):
    cl.plotEllipse(ax, Cpp[i], xpp[i], ypp[i], 'b')

# 

# %% go to undistorted homogenous

xp, yp, Cp = cl.homDist2homUndist(xpp, ypp, distCoeffs, model, Cpp=Cpp, Ck=Ck)

fig = plt.figure()
ax = fig.gca()
for i in range(nPts):
    cl.plotEllipse(ax, Cp[i], xp[i], yp[i], 'b')
#

# %% project to map
# cargo la rototraslacion
rV = rVecs[j].reshape(-1)
tV = tVecs[j].reshape(-1)

# invento unas covarianzas
Cr = np.diag((rV*0.001)**2)
Ct = np.diag((tV*0.001)**2)
Crt = [Cr, Ct]


xm, ym, Cm = cl.xypToZplane(xp, yp, rV, tV, Cp=Cp, Crt=[Cr, Ct])

fig = plt.figure()
ax = fig.gca()
ax.plot(xm, ym, '+', markersize=2)
for i in range(nPts):
    cl.plotEllipse(ax, Cm[i], xm[i], ym[i], 'b')



# %% in one line

cl.inverse(imagePoints, rV, tV, cameraMatrix, distCoeffs, model, Cccd, Cf, Ck, Crt)

fig = plt.figure()
ax = fig.gca()
ax.plot(xm, ym, '+', markersize=2)
for i in range(nPts):
    cl.plotEllipse(ax, Cm[i], xm[i], ym[i], 'b')