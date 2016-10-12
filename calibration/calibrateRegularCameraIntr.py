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
from numpy import zeros
import glob
import matplotlib.pyplot as plt
from scipy import linalg
import poseCalibration as pc
from lmfit import minimize, Parameters
import poseRationalCalibration as rational

# %%
reload(pc)


# %% LOAD DATA
#imagesFolder = "./resources/fishChessboardImg/"
#cornersFile = "/home/sebalander/code/sebaPhD/resources/fishCorners.npy"
#patternFile = "/home/sebalander/code/sebaPhD/resources/chessPattern.npy"
#imgShapeFile = "/home/sebalander/code/sebaPhD/resources/fishShape.npy"

imagesFolder = "./resources/PTZchessboard/zoom 0.0/"
cornersFile = "./resources/PTZchessboard/zoom 0.0/ptzCorners.npy"
patternFile = "./resources/chessPattern.npy"
imgShapeFile = "./resources/ptzImgShape.npy"

corners = np.load(cornersFile).transpose((0,2,1,3))
fiducialPoints = np.load(patternFile)
imgSize = np.load(imgShapeFile)
images = glob.glob(imagesFolder+'*.jpg')

# output files
distCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
linearCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"
rvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzRvecs.npy"
tvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzTvecs.npy"


# %% # %% from testHomography.py
## use real data
#f = 5e2 # proposal of f, can't be estimated from homography
#
#rVecs, tVecs, Hs = pc.estimateInitialPose(fiducialPoints, corners, f, imgSize)
#
#pc.plotHomographyToMatch(fiducialPoints, corners[1:3], f, imgSize, images[1:3])
#
#pc.plotForwardHomography(fiducialPoints, corners[1:3], f, imgSize, Hs[1:3], images[1:3])
#
#pc.plotBackwardHomography(fiducialPoints, corners[1:3], f, imgSize, Hs[1:3])

# %%
n = corners.shape[0] # number of images
m = corners.shape[1] # points per image

model= 'rational'

f = 5e2  # importa el orden de magnitud aunque no demaisado.
         # entre 1e2 y 1e3 anda?

# %% ========== ========== RATIONAL PARAMETER HANDLING ========== ==========
def formatParametersChessIntrs(rVecs, tVecs, linearCoeffs, distCoeffs):
    '''
    set to vary all parameetrs
    '''
    params = Parameters()
    
    for j in range(len(rVecs)):
        for i in range(3):
            params.add('rvec%d%d'%(j,i),
                       value=rVecs[j,i,0], vary=True)
            params.add('tvec%d%d'%(j,i),
                       value=tVecs[j,i,0], vary=True)
    
    params.add('cameraMatrix0',
               value=linearCoeffs[0,0], vary=True)
    params.add('cameraMatrix1',
               value=linearCoeffs[1,1], vary=True)
    params.add('cameraMatrix2',
               value=linearCoeffs[0,2], vary=True)
    params.add('cameraMatrix3',
               value=linearCoeffs[1,2], vary=True)
    
    # polynomial coeffs, grade 7
    # # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    for i in [2,3,8,9,10,11,12,13]:
        params.add('distCoeffs%d'%i,
                   value=distCoeffs[i,0], vary=False)
    
    for i in [0,1,4,5,6,7]:
        params.add('distCoeffs%d'%i,
                   value=distCoeffs[i,0], vary=True)
    
    return params

# %%

def retrieveParametersChess(params, n):
    rvec = zeros((n,3,1))
    tvec = zeros((n,3,1))
    
    for j in range(n):
        for i in range(3):
            rvec[j,i,0] = params['rvec%d%d'%(j,i)].value
            tvec[j,i,0] = params['tvec%d%d'%(j,i)].value
    
    cameraMatrix = zeros((3,3))
    cameraMatrix[0,0] = params['cameraMatrix0'].value
    cameraMatrix[1,1] = params['cameraMatrix1'].value
    cameraMatrix[0,2] = params['cameraMatrix2'].value
    cameraMatrix[1,2] = params['cameraMatrix3'].value
    cameraMatrix[2,2] = 1
    
    distCoeffs = zeros((14,1))
    for i in range(14):
        distCoeffs[i,0] = params['distCoeffs%d'%i].value
    
    return rvec, tvec, cameraMatrix, distCoeffs

# %% intrinsic parameters initial conditions
linearCoeffs = np.array([[f, 0, imgSize[0]/2], [0, f, imgSize[0]/2], [0, 0, 1]])
distCoeffs = np.zeros((14, 1))  # despues hacer generico para todos los modelos

# %% extrinsic parameters initial conditions, from estimated homography
rVecs, tVecs, Hs = pc.estimateInitialPose(fiducialPoints, corners, f, imgSize)

# %%
# format parameters

initialParams = formatParametersChessIntrs(rVecs, tVecs, linearCoeffs, distCoeffs)

# test retrieving 
# n=10
# retrieveParametersChess(initialParams,n)


# %% residual

def residualDirectChessRatio(params, fiducialPoints, imageCorners):
    '''
    '''
    n = len(imageCorners)
    rVecs, tVecs, linearCoeffs, distCoeffs = retrieveParametersChess(params, n)
    E = list()
    
    for j in range(n):
        projectedCorners = rational.direct(fiducialPoints,
                                          rVecs[j],
                                          tVecs[j],
                                          linearCoeffs,
                                          distCoeffs)
        err = imageCorners[j,:,0,:] - projectedCorners[:,0,:]
        E.append(err)
    
    return np.reshape(E,(n*len(fiducialPoints[0]),2))

# %%
def calibrateDirectChessRatio(fiducialPoints, corners, rVecs, tVecs, linearCoeffs, distCoeffs):
    initialParams = formatParametersChessIntrs(rVecs, tVecs, linearCoeffs, distCoeffs) # generate Parameters obj
    
    out = minimize(residualDirectChessRatio,
                   initialParams,
                   args=(fiducialPoints,
                         corners))
    
    return out

# %%
# E = residualDirectChessRatio(initialParams, fiducialPoints, corners)

out = calibrateDirectChessRatio(fiducialPoints, corners, rVecs, tVecs, linearCoeffs, distCoeffs)

rVecsOpt, tVecsOpt, cameraMatrixOpt, distCoeffsOpt = retrieveParametersChess(out.params, len(corners))


# %%
i=9
img = plt.imread(images[i])
imageCorners = corners[i]
rVecOpt = rVecsOpt[i]
tVecOpt = tVecsOpt[i]

cornersProjectedOpt = pc.direct(fiducialPoints, rVecOpt, tVecOpt, linearCoeffs, distCoeffs, model)

pc.cornerComparison(img, imageCorners, cornersProjectedOpt)

# %%

















# %% ==========================================================================
'''
hacer optimizacion para terminar de ajustar el pinhole y asi sacar la pose para
cada imagen. usar las funciones de las librerias extrinsecas
'''
# es para cada imagen por separado
linearCoeffs = np.array([[f, 0, imgSize[0]/2],
                         [0, f, imgSize[1]/2],
                         [0, 0, 1]], dtype='float32')

distCoeffs = np.zeros((14,1)) # no hay distorsion

# %% test initial calibration INVERSE

rVecsIni, tVecsIni = initialPoses2(fiducialPoints, corners, imgSize, f)


# %%
i = 9


fiducialProjectedIni = pc.inverse(corners[i],
                                  rVecsIni[i], tVecsIni[i],
                                  linearCoeffs, distCoeffs, model)

# plot
# pc.fiducialComparison(fiducialPoints, fiducialProjectedIni)
pc.fiducialComparison3D(rVecsIni[i], tVecsIni[i], fiducialPoints, fiducialProjectedIni, label1 = 'Fiducial points', label2 = 'Projected points')


# %% optimise rVec, tVec
rVecOpt, tVecOpt, optParams = pc.calibrateInverse(fiducialPoints, corners[i], rVecsIni[i], tVecsIni[i], linearCoeffs, distCoeffs,model)

fiducialProjectedOpt = pc.inverse(corners[i], rVecOpt, tVecOpt, linearCoeffs, distCoeffs,model)
# plot
pc.fiducialComparison(fiducialPoints, fiducialProjectedOpt)
pc.fiducialComparison3D(rVecOpt, tVecOpt, fiducialPoints, fiducialProjectedOpt, label1 = 'Fiducial points', label2 = 'Optimised points')


# %% test initial calibration DIRECT

cornersProjectedIni = pc.direct(fiducialPoints,
                                rVecsIni[i], tVecsIni[i],
                                linearCoeffs, distCoeffs, model)

# plot corners in image
img = cv2.imread(images[i])
pc.cornerComparison(img, corners[i], cornersProjectedIni)

# %% optimise rVec, tVec
rVecOpt, tVecOpt, optParams = pc.calibrateDirect(fiducialPoints, corners[i], rVecsIni[i], tVecsIni[i], linearCoeffs, distCoeffs, model)

# test mapping with optimised conditions
cornersProjectedOpt = pc.direct(fiducialPoints, rVecOpt, tVecOpt, linearCoeffs, distCoeffs, model)
pc.cornerComparison(img, corners[i], cornersProjectedOpt)


# %%
# para todas las imagenes juntas dejar la pose fija y optimizar los parametros de distorsion 

# %% optimizar todo junto a la vez tomando lo anterior como condiciones iniciales.