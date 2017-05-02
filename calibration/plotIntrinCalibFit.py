#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 17:26:31 2017

test intrinsic calibration paramters

@author: sebalander
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:30:53 2016

calibrates intrinsic with diff distortion model

@author: sebalander
"""

# %%
import glob
import numpy as np
import scipy.linalg as ln
from calibration import calibrator as cl
import matplotlib.pyplot as plt


# %% LOAD DATA
# input
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye']
model = modelos[2]
plotCorners = False

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# load data
imagePoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')

n = len(imagePoints)  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)

distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)
tVecs = np.load(tVecsFile)
rVecs = np.load(rVecsFile)

# %% MAP TO HOMOGENOUS PLANE TO GET RADIUS
# pruebo con la imagen j-esima
n=len(imagePoints)

RP = []
RPP = []


for j in range(n):
    if plotCorners:
        imagePntsX = imagePoints[j, 0, :, 0]
        imagePntsY = imagePoints[j, 0, :, 1]
    
        rvec = rVecs[j]
        tvec = tVecs[j]
    
        imagePointsProjected = cl.direct(chessboardModel, rvec, tvec,
                                         cameraMatrix, distCoeffs, model)
        imagePointsProjected = imagePointsProjected.reshape((-1,2))
    
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
    RPP.append(ln.norm([xpp,ypp], axis=0))
    
    # calculate undistorted radius
    xyp = cl.rotoTrasHomog(chessboardModel, rVecs[j], tVecs[j])
    RP.append(ln.norm(xyp, axis=1))
    

0  # si no pongo este cero spyder no sale solo del bucle en la consola

# %%

RP = np.array(RP).flatten()
RPP = np.array(RPP).flatten()

rp0 = np.linspace(0,np.max(RP)*1.2)

# %%
plt.figure(n)
plt.plot(RP,RPP, '.', markersize=3)
plt.xlim([0, rp0[-1]])
plt.ylim([0,np.max(RPP)*1.2])

rpp0 = cl.distort[model](rp0, distCoeffs)
plt.plot(rp0, rpp0, '-', lw=1, label=model)

plt.legend()



