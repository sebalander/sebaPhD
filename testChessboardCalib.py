# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:15:12 2016

visualy inspect that chessboard calibration was succesfull

@author: sebalander
"""
# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


# ============================================================================
# ============================ FE WIDE LENS TEST =============================
# %% LOAD DATA
imagesFolder = "./resources/fishChessboardImg/"
cornersFile = "./resources/fishCorners.npy"
patternFile = "./resources/fishWidePattern.npy"
imgShapeFile = "./resources/fishShape.npy"
distCoeffsFile = "./resources/fishDistCoeffs.npy"
linearCoeffsFile = "./resources/fishLinearCoeffs.npy"
rvecsFile = "./resources/fishRvecs.npy"
tvecsFile = "./resources/fishTvecs.npy"

imgpoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')

distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)

rvecs = np.load(rvecsFile)
tvecs = np.load(tvecsFile)

# %% TEST MAPPING (DISTORTION MODEL)

# pruebo con la imagen j-esima
imagePointsProjected = chessboardModel[0,:,0:2]
for j in range(2):  # range(len(imgpoints)):
    
    imagePntsX = imgpoints[j,0,:,0]
    imagePntsY = imgpoints[j,0,:,1]
    
    rvec = rvecs[j][:,0]
    tvec = tvecs[j][:,0]
    
    # project points to image plane with distortion
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