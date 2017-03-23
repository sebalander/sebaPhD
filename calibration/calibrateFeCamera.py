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
# input
# cam puede ser ['fish', 'fishWide', 'ptz']
cam = 'fish'

if cam=='fish':
    imagesFolder = "./resources/fishChessboard/"
    cornersFile =  "./resources/fishChessboard/fishCorners.npy"
    patternFile =  "./resources/fishChessboard/fishPattern.npy"
    imgShapeFile = "./resources/fishChessboard/fishShape.npy"
    
    # output
    distCoeffsFile =   "./resources/fishChessboard/fishDistCoeffs.npy"
    linearCoeffsFile = "./resources/fishChessboard/fishLinearCoeffs.npy"
elif cam=='fishWide':
    imagesFolder = "./resources/fishWideChessboard/"
    cornersFile =  "./resources/fishWideChessboard/fishCorners.npy"
    patternFile =  "./resources/fishWideChessboard/fishPattern.npy"
    imgShapeFile = "./resources/fishWideChessboard/fishShape.npy"
    
    # output
    distCoeffsFile =   "./resources/fishDistCoeffs.npy"
    linearCoeffsFile = "./resources/fishLinearCoeffs.npy"
elif cam=='ptz':
    imagesFolder = "./resources/fishWideChessboard/"
    cornersFile =  "./resources/fishWideChessboard/fishCorners.npy"
    patternFile =  "./resources/fishWideChessboard/fishPattern.npy"
    imgShapeFile = "./resources/fishWideChessboard/fishShape.npy"
    
    # output
    distCoeffsFile =   "./resources/fishDistCoeffs.npy"
    linearCoeffsFile = "./resources/fishLinearCoeffs.npy"

# load data
imgpoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')

# %%
n = len(imgpoints)  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)

rvecs = np.zeros((n, 1, 1, 3))
# [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n)]
tvecs = np.zeros((n, 1, 1, 3))
# [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n)]
K0 = np.eye(3)
D = np.zeros((1, 4))

K0[0, 2] = imgSize[1]/2
K0[1, 2] = imgSize[0]/2
K0[0, 0] = K0[1, 1] = 500.0

# CALIB_USE_INTRINSIC_GUESS cameraMatrix contains valid initial values of fx,
# fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to
# the image center ( imageSize is used), and focal distances are computed in a
# least-squares fashion.
#
# CALIB_RECOMPUTE_EXTRINSIC Extrinsic will be recomputed after each iteration of
# intrinsic optimization.
#
# CALIB_CHECK_COND The functions will check validity of condition number.
#
# CALIB_FIX_SKEW Skew coefficient (alpha) is set to zero and stay zero.
#
# CALIB_FIX_K1..fisheye::CALIB_FIX_K4 Selected distortion coefficients are set
# to zeros and stay zero.
#
# CALIB_FIX_PRINCIPAL_POINT The principal point is not changed during the global
# optimization. It stays at the center or at a different location specified when
# CALIB_USE_INTRINSIC_GUESS is set too.
#
#.CALIB_FIX_SKEW CALIB_USE_INTRINSIC_GUESS CALIB_FIX_PRINCIPAL_POINT
flags = 1 + 8 + 512

# terminaion criteria
criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 10, 0.1)
#
# %% OPTIMIZAR
rvecs = np.zeros((n, 1, 1, 3))
tvecs = np.zeros((n, 1, 1, 3))
K 
rms, K, D, rvecs, tvecs = fe.calibrate(objpoints, imgpoints, imgSize, K0, D,
                    rvecs, tvecs, flags=flags, criteria=criteria)

# %% SAVE CALIBRATION
np.save(distCoeffsFile, D)
np.save(linearCoeffsFile, K)

# %% TEST MAPPING (DISTORTION MODEL)
# pruebo con la imagen j-esima
for j in range(2):  # range(len(imgpoints)):

    imagePntsX = imgpoints[j, 0, :, 0]
    imagePntsY = imgpoints[j, 0, :, 1]

    rvec = rvecs[j]
    tvec = tvecs[j]

    imagePointsProjected = []
    imagePointsProjected = fe.projectPoints(chessboardModel,
                                            rvec,
                                            tvec,
                                            K,
                                            D)

    xPos = np.array(imagePointsProjected[0][0, :, 0])
    yPos = np.array(imagePointsProjected[0][0, :, 1])

    plt.figure(j)
    im = plt.imread(images[j])
    plt.imshow(im)
    plt.plot(imagePntsX, imagePntsY, 'xr', markersize=10)
    plt.plot(xPos, yPos, '+b', markersize=10)
    #fig.savefig("distortedPoints3.png")

# %% to try and solve the inverse problem
coeffs = [D[3,0], 0, D[2,0], 0, D[1,0], 0, D[0,0], 0, 1, -100]
theta = np.roots(coeffs)
theta = np.real(theta[np.isreal(theta)])  # extraigo el que no es complejo