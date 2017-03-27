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
from calibration import poseCalibration as pc


# %% LOAD DATA
# input
# cam puede ser ['fish', 'fishWide', 'ptz']
cam = 'fish'
# puede ser ['rational', fisheye]
model = 'rational'

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

'''
rVecs and tVecs is be a list of n elements each a vector of shape (3,1)
'''

#rVecs = [np.array([[np.pi], [0], [0]])] * n
## [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n)]
#tVecs = [np.array([[0], [0], [1]])] * n
## [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n)]
K0 = np.eye(3)

de = {
    'rational' : np.zeros((1, 5)),
    'fisheye' : np.zeros((4))
    }

D = de[model]

K0[0, 2] = imgSize[1]/2
K0[1, 2] = imgSize[0]/2
K0[0, 0] = K0[1, 1] = 600.0

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

#.CALIB_FIX_SKEW CALIB_USE_INTRINSIC_GUESS 
# CALIB_RECOMPUTE_EXTRINSIC CALIB_FIX_PRINCIPAL_POINT 
flags = 1 + 2 + 8 + 512

#.CALIB_FIX_SKEW CALIB_FIX_PRINCIPAL_POINT 
flags = 1 + 512

# terminaion criteria
criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, int(1e6), 1e-4)

# %% OPTIMIZAR

switcherOpt = {
'rational' : cv2.calibrateCamera,
'fisheye' : fe.calibrate
}

#rms, K, D, rVecs, tVecs = switcher[model](objpoints, imgpoints, imgSize, K0, D,
#                    rVecs, tVecs, flags=flags, criteria=criteria)

rms, K, D, rVecs, tVecs = switcherOpt[model](objpoints, imgpoints,
                                  imgSize, K0, D,
                                  flags=flags) #, criteria=criteria)
#
##.CALIB_FIX_SKEW CALIB_USE_INTRINSIC_GUESS 
## CALIB_RECOMPUTE_EXTRINSIC CALIB_FIX_PRINCIPAL_POINT 
#flags = 1 + 2 + 8 + 512
#
#rms, K, D, rVecs, tVecs = switcherOpt[model](objpoints, imgpoints,
#                                  imgSize, K0, D,
#                                  np.array(rVecs).reshape((3,n)),
#                                  np.array(tVecs).reshape((3,n)),
#                                  flags=flags, criteria=criteria)

# %% plot fiducial points and corners to ensure the calibration data is ok

for i in range(n): # [9,15]:
    rVec = rVecs[i]
    tVec = tVecs[i]
    fiducial1 = chessboardModel
    
    pc.fiducialComparison3D(rVec, tVec, fiducial1)



# %% TEST MAPPING (DISTORTION MODEL)
# pruebo con la imagen j-esima

switcherProj = {
    'rational' : cv2.projectPoints,
    'fisheye' : fe.projectPoints
    }


for j in range(n):  # range(len(imgpoints)):

    imagePntsX = imgpoints[j, 0, :, 0]
    imagePntsY = imgpoints[j, 0, :, 1]

    rvec = rVecs[j]
    tvec = tVecs[j]

    imagePointsProjected = []
    imagePointsProjected = switcherProj[model](chessboardModel,
                                            rvec,
                                            tvec,
                                            K,
                                            D)

    if model == 'rational':
        xPos = np.array(imagePointsProjected[0][:, 0, 0])
        yPos = np.array(imagePointsProjected[0][:, 0, 1])
    if model == 'fisheye':
        xPos = np.array(imagePointsProjected[0][0, :, 0])
        yPos = np.array(imagePointsProjected[0][0, :, 1])

    plt.figure()
    im = plt.imread(images[j])
    plt.imshow(im)
    plt.plot(imagePntsX, imagePntsY, 'xr', markersize=10)
    plt.plot(xPos, yPos, '+b', markersize=10)
    #fig.savefig("distortedPoints3.png")

## %% SAVE CALIBRATION
#np.save(distCoeffsFile, D)
#np.save(linearCoeffsFile, K)
