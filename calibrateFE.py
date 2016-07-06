# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:30:53 2016

calibrates using fisheye distortion model (polynomial in theta)

@author: sebalander
"""

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2.fisheye as fe
import glob

# %% LOAD DATA
imagesFolder = "/home/sebalander/code/sebaPhD/resources/fishWideChessboardImg/"
cornersFile = "/home/sebalander/code/sebaPhD/resources/fishWideCorners.npy"
patternFile = "/home/sebalander/code/sebaPhD/resources/fishWidePattern.npy"
imgShapeFile = "/home/sebalander/code/sebaPhD/resources/fishWideShape.npy"

imgpoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgShape = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')


# %%
n = len(imgpoints)
# Parametros de entrada/salida de la calibracion
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n)]
K = np.zeros((3, 3))
D = np.zeros((1, 4))
objpoints = [chessboardModel]*n
#calibrationFlags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
calibrationFlags = 0 # se obtiene el mismo resultado que con el de arriba
calibrationCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, # termination criteria type
            1000, # max number of iterations
            0.0001) # min accuracy

# %% OPTIMIZAR 
rms, _, _, _, _ = fe.calibrate(objpoints, 
                               imgpoints, 
                               imgShape, 
                               K,
                               D, 
                               rvecs, 
                               tvecs, 
                               calibrationFlags, 
                               calibrationCriteria)

# %% RE COMPUTE EXTRINSIC
calibrationFlags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC \
                   + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS

rms, _, _, _, _ = fe.calibrate(objpoints, 
                               imgpoints, 
                               imgShape, 
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
    ax.scatter(xPos, yPos)
    ax.scatter(imagePntsX, imagePntsY)
    #fig.savefig("distortedPoints3.png")