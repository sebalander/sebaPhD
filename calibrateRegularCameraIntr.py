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
import glob

# %% LOAD DATA
#imagesFolder = "./resources/fishChessboardImg/"
#cornersFile = "/home/sebalander/code/sebaPhD/resources/fishCorners.npy"
#patternFile = "/home/sebalander/code/sebaPhD/resources/chessPattern.npy"
#imgShapeFile = "/home/sebalander/code/sebaPhD/resources/fishShape.npy"

imagesFolder = "./resources/PTZchessboard/zoom 0.0/"
cornersFile = "./resources/PTZchessboard/zoom 0.0/ptzCorners.npy"
patternFile = "./resources/chessPattern.npy"
imgShapeFile = "./resources/ptzImgShape.npy"

imgpoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')

# output files
distCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
linearCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"
rvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzRvecs.npy"
tvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzTvecs.npy"

# %%
n = len(imgpoints)  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
rvecs = ()
tvecs = ()
cameraMatrix = ()
distCoeffs = ()

# enable possible coefficients
flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_RATIONAL_MODEL

objpoints = [chessboardModel]*n

calibrationCriteria = (cv2.TERM_CRITERIA_EPS
                       + cv2.TERM_CRITERIA_MAX_ITER,
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

# (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])


# %% SAVE PARAMETERS, INTRINSIC AND EXTRINSIC
np.save(distCoeffsFile, distCoeffs)
np.save(linearCoeffsFile, cameraMatrix)
np.save(rvecsFile, rvecs)
np.save(tvecsFile, tvecs)