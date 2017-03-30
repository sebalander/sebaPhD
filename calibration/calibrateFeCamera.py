# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:30:53 2016

calibrates using fisheye distortion model (polynomial in theta)

@author: sebalander
"""

# %%
import cv2
import glob
import numpy as np
from calibration import calibrator as cl
import matplotlib.pyplot as plt


# %% LOAD DATA
# input
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'ptz'
# puede ser ['rational', fisheye]
model = 'rational'

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# output
distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# load data
imgpoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')

n = len(imgpoints)  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)


# %% OPTIMIZAR
rms, K, D, rVecs, tVecs = cl.calibrateIntrinsic(objpoints, imgpoints, imgSize, model)


# %% plot fiducial 
#points and corners to ensure the calibration data is ok

for i in range(n): # [9,15]:
    rVec = rVecs[i]
    tVec = tVecs[i]
    fiducial1 = chessboardModel
    
    cl.fiducialComparison3D(rVec, tVec, fiducial1)

0

# %% TEST MAPPING (DISTORTION MODEL)
# pruebo con la imagen j-esima


for j in range(n):  # range(len(imgpoints)):
    imagePntsX = imgpoints[j, 0, :, 0]
    imagePntsY = imgpoints[j, 0, :, 1]
    
    rvec = rVecs[j]
    tvec = tVecs[j]
    
    imagePointsProjected = cl.direct(chessboardModel, rvec, tvec, K, D, model)
    
    if model == 'rational':
        xPos = np.array(imagePointsProjected[:, 0, 0])
        yPos = np.array(imagePointsProjected[:, 0, 1])
    
    if model == 'fisheye':
        xPos = np.array(imagePointsProjected[0, :, 0])
        yPos = np.array(imagePointsProjected[0, :, 1])
        
    plt.figure()
    im = plt.imread(images[j])
    plt.imshow(im)
    plt.plot(imagePntsX, imagePntsY, 'xr', markersize=10)
    plt.plot(xPos, yPos, '+b', markersize=10)
    #fig.savefig("distortedPoints3.png")

0

# %% SAVE CALIBRATION
np.save(distCoeffsFile, D)
np.save(linearCoeffsFile, K)
np.save(tVecsFile, tVecs)
np.save(rVecsFile, rVecs)