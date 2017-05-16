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


# %%
from scipy.special import chdtriv
fi = np.linspace(0,2*np.pi,20)
Xcirc = np.array([np.cos(fi), np.sin(fi)]) * chdtriv(0.1, 2)

def plotEllipse(ax, c, mux, muy, col):
    '''
    se grafican una elipse asociada a la covarianza c, centrada en mux, muy
    '''
    
    l, v = ln.eig(ln.inv(c))
    
    D = v.T*np.sqrt(l.real) # queda con los autovectores como filas
    
    xeli, yeli = np.dot(ln.inv(D), Xcirc)
    
    ax.plot(xeli+mux, yeli+muy, c=col, lw=0.5)

#


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

n = len(imagePoints)  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)

distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# load model specific data
distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)
rVecs = np.load(rVecsFile)
tVecs = np.load(tVecsFile)

# %% choose image
j = 18
print('\t imagen', j)

# %% testeo cada paso de propagacion
imagePoints[j,0].shape

Cccd = np.array([np.eye(2)]*imagePoints[j,0].shape[0])

reload(cl)

xp, yp, Cp = cl.ccd2homUndistorted(imagePoints[j].reshape(-1,2), cameraMatrix,  distCoeffs, model, cov=Cccd)

fig = plt.figure()
ax = fig.gca()

for i in range(len(xp)):
    print(Cp[i], xp[i], yp[i])
    plotEllipse(ax, Cp[i], xp[i], yp[i], 'k')

# %%
rvec = rVecs[j]
tvec = tVecs[j]

imagePointsProjected = cl.direct(chessboardModel, rvec, tvec,
                                 cameraMatrix, distCoeffs, model)
imagePointsProjected = imagePointsProjected.reshape((-1,2))

objectPointsProjected = cl.inverse(imagePoints[j,0], rvec, tvec,
                                    cameraMatrix, distCoeffs, model)
objectPointsProjected = objectPointsProjected.reshape((-1,3))

if plotCorners:
    imagePntsX = imagePoints[j, 0, :, 0]
    imagePntsY = imagePoints[j, 0, :, 1]

    xPos = imagePointsProjected[:, 0]
    yPos = imagePointsProjected[:, 1]

    plt.figure(j)
    im = plt.imread(images[j])
    plt.imshow(im)
    plt.plot(imagePntsX, imagePntsY, 'xr', markersize=10)
    plt.plot(xPos, yPos, '+b', markersize=10)
    #fig.savefig("distortedPoints3.png")

# calculate distorted radius
xpp, ypp = cl.ccd2hom(imagePoints[j,0], cameraMatrix)
RPP[model].append(ln.norm([xpp,ypp], axis=0))

# calculate undistorted homogenous radius from 3D rototraslation
xyp = cl.rotoTrasHomog(chessboardModel, rVecs[j], tVecs[j])
RP[model].append(ln.norm(xyp, axis=1))





