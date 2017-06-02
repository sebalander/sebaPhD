#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 20:11:26 2017

test uncertanty propagation wrt montecarlo

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
model = modelos[2]

model

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
if model is 'poly':
    Ck = np.diag((distCoeffs[[0,1,4]]*0.001)**2)  # 0.1% error distorsion
if model is 'rational':
    Ck = np.diag((distCoeffs[[0,1,4,5,6,7]]*0.001)**2)  # 0.1% error distorsion
if model is 'fisheye':
    Ck = np.diag((distCoeffs*0.001)**2)  # 0.1% error distorsion

# %% choose image
j = 2
print('\t imagen', j)
imagePoints = np.load(cornersFile)
imagePoints = imagePoints[j,0]
npts = imagePoints.shape[0]

img = plt.imread(images[j])

# cargo la rototraslacion
rV = rVecs[j].reshape(-1)
tV = tVecs[j].reshape(-1)

# invento unas covarianzas
Cccd = (0.5**2) * np.array([np.eye(2)]*imagePoints.shape[0])

Cr = np.diag((rV*0.001)**2)
Ct = np.diag((tV*0.001)**2)
Crt = [Cr, Ct]


# plot initial uncertanties
fig = plt.figure()
ax = fig.gca()
ax.imshow(img)
for i in range(nPts):
    x, y = imagePoints[i]
    cl.plotEllipse(ax, Cccd[i], x, y, 'b')

# %% in one line, analytical propagation
xm, ym, Cm = cl.inverse(imagePoints, rV, tV, cameraMatrix, distCoeffs, model, Cccd, Cf, Ck, Crt)


# %% simulate many
N = 200

# por suerte todas las pdfs son indep
xI = np.random.randn(N,npts,2).dot(np.sqrt(Cccd[0])) + imagePoints
rots = np.random.randn(N,3).dot(np.sqrt(Cr)) + rV
tras = np.random.randn(N,3).dot(np.sqrt(Ct)) + tV
kD = np.random.randn(N,Ck.shape[0]).dot(np.sqrt(Ck)) + distCoeffs  # disorsion
kL = np.zeros((N, 3, 3), dtype=float)  # lineal
kL[:,2,2] = 1
kL[:,:2,2] = np.random.randn(N,2).dot(np.sqrt(Cf[2:,2:])) + cameraMatrix[:2,2]
kL[:,[0,1],[0,1]] = np.random.randn(N,2).dot(np.sqrt(Cf[:2,:2]))
kL[:,[0,1],[0,1]] += cameraMatrix[[0,1],[0,1]]

xM = np.empty((N,npts), dtype=float)
yM = np.empty_like(xM)

for i in range(N):
    xM[i], yM[i] = cl.inverse(xI[i], rots[i], tras[i], kL[i], kD[i], model)

fig = plt.figure()
ax = fig.gca()
ax.plot(xm, ym, '+', markersize=2)
for i in range(nPts):
    cl.plotEllipse(ax, Cm[i], xm[i], ym[i], 'b')

ax.plot(xM,yM,'.k', markersize=0.5)


