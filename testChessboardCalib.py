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
#imagesFolder = "./resources/fishChessboard/"
#cornersFile = "./resources/fishChessboard/fishCorners.npy"
#patternFile = "./resources/chessPattern.npy"
#imgShapeFile = "./resources/fishImgShape.npy"
#distCoeffsFile = "./resources/fishDistCoeffs.npy"
#linearCoeffsFile = "./resources/fishLinearCoeffs.npy"
#rvecsFile = "./resources/fishChessboard/fishRvecs.npy"
#tvecsFile = "./resources/fishChessboard/fishTvecs.npy"

imagesFolder = "./resources/PTZchessboard/zoom 0.0/"
cornersFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzCorners.npy"
patternFile = "/home/sebalander/Code/sebaPhD/resources/chessPattern.npy"
imgShapeFile = "/home/sebalander/Code/sebaPhD/resources/ptzImgShape.npy"
distCoeffsFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
linearCoeffsFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"
rvecsFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzRvecs.npy"
tvecsFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzTvecs.npy"


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

for j in [2,5,7,10]:  # range(len(imgpoints)):
    
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