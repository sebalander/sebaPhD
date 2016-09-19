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
import matplotlib.pyplot as plt

# %% LOAD DATA
#imagesFolder = "./resources/fishChessboardImg/"
#cornersFile = "/home/sebalander/code/sebaPhD/resources/fishCorners.npy"
#patternFile = "/home/sebalander/code/sebaPhD/resources/chessPattern.npy"
#imgShapeFile = "/home/sebalander/code/sebaPhD/resources/fishShape.npy"

imagesFolder = "./resources/PTZchessboard/zoom 0.0/"
cornersFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzCorners.npy"
patternFile = "/home/sebalander/Code/sebaPhD/resources/chessPattern.npy"
imgShapeFile = "/home/sebalander/Code/sebaPhD/resources/ptzImgShape.npy"

corners = np.load(cornersFile)
fiducialPoints = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')

# output files
distCoeffsFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
linearCoeffsFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"
rvecsFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzRvecs.npy"
tvecsFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzTvecs.npy"

# %% LINEAR APPROACH TO CAMERA CALIBRATION
n = corners.shape[0] # number of images
m = corners.shape[2] # points per image

#i = 3
#j = 31
#x = corners[i,0,j]
#p = fiducialPoints[0,j]
cer = np.zeros(4)

# los 1s se agregan porque tiene que se en coordenadas homogéneas
def Pelement(x,p):
    a = np.concatenate((p, [1], cer, -x[0]*p, [-x[0]]))
    b = np.concatenate((cer, p, [1], -x[1]*p, [-x[1]]))
    return np.concatenate(([a] ,[b] ),0)


PP = np.array([[Pelement(corners[i,0,j],fiducialPoints[0,j]) for j in range(m)] for i in range(n)])

PP = np.reshape(np.array(PP),(2*n*m,12),order='F')

# check that there are more rows than columns (12)
# make square
PTP = PP.T.dot(PP)
l, v = np.linalg.eig(PTP)

# %% WHAT FOLLOWS IS OBSOLETE
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