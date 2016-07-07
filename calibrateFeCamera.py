# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:30:53 2016

calibrates using fisheye distortion model (polynomial in theta)

@author: sebalander
"""

# %%
import cv2
import numpy as np
#import matplotlib.pyplot as plt
import cv2.fisheye as fe
import glob

# %% LOAD DATA
imagesFolder = ".resources/fishWideChessboardImg/"
cornersFile = "./resources/fishWideCorners.npy"
patternFile = "./resources/fishWidePattern.npy"
imgShapeFile = "./resources/fishWideShape.npy"

imgpoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')


# %%
n = len(imgpoints)
# Parametros de entrada/salida de la calibracion
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n)]
K = np.zeros((3, 3))
D = np.zeros((1, 4))
objpoints = [chessboardModel]*n

calibrationFlags = 0 # se obtiene el mismo resultado que con el de arriba

calibrationCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, # termination criteria type
            1000, # max number of iterations
            0.0001) # min accuracy

# %% OPTIMIZAR
rms, K, D, rvecs, tvecs = fe.calibrate(objpoints,
                               imgpoints,
                               imgSize,
                               K,
                               D,
                               rvecs,
                               tvecs,
                               calibrationFlags,
                               calibrationCriteria)

# %% TEST MAPPING (DISTORTION MODEL)

# pruebo con la imagen j-esima
for j in range(2):  # range(len(imgpoints)):

    imagePntsX = imgpoints[j,0,:,0]
    imagePntsY = imgpoints[j,0,:,1]

    rvec = rvecs[j][0,0]
    tvec = tvecs[j][0,0]

    imagePointsProjected = []
    imagePointsProjected = fe.projectPoints(chessboardModel,
                            	      	 rvec,
                                		 tvec,
                                		 K,
                                		 D)

    xPos = np.array(imagePointsProjected[0][0,:,0])
    yPos = np.array(imagePointsProjected[0][0,:,1])

    fig, ax = plt.subplots(1)
    im = plt.imread(images[j])
    ax.imshow(im)
    ax.plot(imagePntsX, imagePntsY, 'xr', markersize=10)
    ax.plot(xPos, yPos, '+b', markersize=10)
    #fig.savefig("distortedPoints3.png")


# %% to try and solve the inverse problem
coeffs = [D[3,0], 0, D[2,0], 0, D[1,0], 0, D[0,0], 0, 1, -100]
theta = np.roots(coeffs)
theta = np.real(theta[~ np.iscomplex(theta)])  # extraigo el que no es complejo