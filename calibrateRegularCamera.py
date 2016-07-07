# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:30:53 2016

calibrates using fisheye distortion model (polynomial in theta)

help in
http://docs.opencv.org/ref/master/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d&gsc.tab=0

@author: sebalander
"""

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# %% LOAD DATA
imagesFolder = "./resources/fishWideChessboardImg/"
cornersFile = "./resources/fishWideCorners.npy"
patternFile = "./resources/fishWidePattern.npy"
imgShapeFile = "./resources/fishWideShape.npy"
distCoeffsFile = "./resources/fishWideFEDistCoeffs.npy"
linearCoeffsFile = "./resources/fishWideFELinearCoeffs.npy"

imgpoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')


# %%
n = len(imgpoints)  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
rvecs = ()
tvecs = ()
cameraMatrix = ()
distCoeffs = ()

# enable possible coefficients
flags = cv2.CALIB_RATIONAL_MODEL  # \
#        + cv2.CALIB_THIN_PRISM_MODEL \
#        + cv2.CALIB_TILTED_MODEL

objpoints = [chessboardModel]*n
#calibrationFlags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
calibrationFlags = 0 # se obtiene el mismo resultado que con el de arriba
calibrationCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, # termination criteria type
            1000, # max number of iterations
            0.0001) # min accuracy

# %% OPTIMIZAR 
rms, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                                  imgpoints,
                                                                  imgSize,
                                                                  cameraMatrix,
                                                                  distCoeffs,
                                                                  rvecs,
                                                                  tvecs,
                                                                  flags)

# %% SAVE INTRINSIC PARAMETERS
np.save(distCoeffsFile, distCoeffs)
np.save(linearCoeffsFile, cameraMatrix)


# %% TEST MAPPING (DISTORTION MODEL)

# pruebo con la imagen j-esima
imagePointsProjected = chessboardModel[0,:,0:2]
for j in range(n):  # range(len(imgpoints)):
    
    imagePntsX = imgpoints[j,0,:,0]
    imagePntsY = imgpoints[j,0,:,1]
    
    rvec = rvecs[j][:,0]
    tvec = tvecs[j][:,0]
    
    imagePointsProjected, _ = cv2.projectPoints(chessboardModel,
                              rvec,
                              tvec,
                              cameraMatrix,
                              distCoeffs,
                              imagePointsProjected)
    
    xPos = np.array(imagePointsProjected[:,0,0])
    yPos = np.array(imagePointsProjected[:,0,1])
    
    fig, ax = plt.subplots(1)
    im = plt.imread(images[j])
    ax.imshow(im)
    ax.plot(imagePntsX, imagePntsY, 'xr', markersize=10)
    ax.plot(xPos, yPos, '+b', markersize=10)
    #fig.savefig("distortedPoints3.png")