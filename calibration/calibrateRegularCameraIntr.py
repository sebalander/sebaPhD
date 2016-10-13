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
#from scipy import linalg
import poseCalibration as pc
from lmfit import minimize, Parameters
import poseRationalCalibration as rational


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
                         corners),
                   xtol=1e-7, # Relative error in the approximate solution
                   ftol=1e-7) # Relative error in the desired sum of squares
    
    return out


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

# calibration using chessboard method opencv
distCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
linearCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"
rvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzRvecs.npy"
tvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzTvecs.npy"

distCoeffsTrue = np.load(distCoeffsFile)
linearCoeffsTrue = np.load(linearCoeffsFile)
rVecsTrue = np.load(rvecsFile)
tVecsTrue = np.load(tvecsFile)

flip = False # hacer el flip da muuucho peor y no tiene sentido además

# %% reverse y axis
# flip 'y' coordinates, so it works
# why flip image:
# http://stackoverflow.com/questions/14589642/python-matplotlib-inverted-image
corners = np.load(cornersFile).transpose((0,2,1,3))

if flip:
    corners = np.array([ [0, imgSize[0]] + cor[:,:,:2]*[1,-1] for cor in corners])

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
model= 'rational'

f = 1e3  # importa el orden de magnitud aunque no demaisado.
         # entre 1e2 y 1e3 anda?

# %% intrinsic parameters initial conditions
linearCoeffsIni = np.array([[f, 0, imgSize[1]/2], [0, f, imgSize[0]/2], [0, 0, 1]])
distCoeffsIni = np.zeros((14, 1))  # despues hacer generico para todos los modelos
#distCoeffsIni = distCoeffsTrue
# %% extrinsic parameters initial conditions, from estimated homography
rVecsIni, tVecsIni, Hs = pc.estimateInitialPose(fiducialPoints, corners, f, imgSize)

# %% from testposecalibration DIRECT GENERIC CALIBRATION
i=3
img = plt.imread(images[i])
if flip:
    img = np.flipud(img)
imageCorners = corners[i]
rVec = rVecsIni[i]
tVec = tVecsIni[i]
linearCoeffs = linearCoeffsIni
distCoeffs = distCoeffsIni

# direct mapping with initial conditions
cornersProjected = pc.direct(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs, model)

# plot corners in image
pc.cornerComparison(img, imageCorners, cornersProjected)

# test mapping with initial conditions
fiducialProjected = pc.inverse(imageCorners, rVec, tVec, linearCoeffs, distCoeffs, model)

# plot
pc.fiducialComparison3D(rVec, tVec, fiducialPoints, fiducialProjected, label1 = 'Fiducial points', label2 = 'Projected points')



# %%
# format parameters

initialParams = formatParametersChessIntrs(rVecsIni, tVecsIni, linearCoeffsIni, distCoeffsIni)

# test retrieving parameters
# n=10
# retrieveParametersChess(initialParams,n)



# %%
#E = residualDirectChessRatio(initialParams, fiducialPoints, corners)

out = calibrateDirectChessRatio(fiducialPoints, corners, rVecsIni, tVecsIni, linearCoeffsIni, distCoeffsIni)

out.nfev
out.message
out.lmdif_message
(out.residual**2).sum()


rVecsOpt, tVecsOpt, cameraMatrixOpt, distCoeffsOpt = retrieveParametersChess(out.params, len(corners))
rVecsOpt 


# %%
img = plt.imread(images[i])
if flip:
    img = np.flipud(img)
imageCorners = corners[i]
rVec = rVecsOpt[i]
tVec = tVecsOpt[i]
linearCoeffs = cameraMatrixOpt
distCoeffs = distCoeffsOpt

# direct mapping with initial conditions
cornersProjected = pc.direct(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs, model)

# plot corners in image
pc.cornerComparison(img, imageCorners, cornersProjected)


# test mapping with initial conditions
fiducialProjected = pc.inverse(imageCorners, rVec, tVec, linearCoeffs, distCoeffs, model)

# plot
pc.fiducialComparison3D(rVec, tVec, fiducialPoints, fiducialProjected, label1 = 'Fiducial points', label2 = 'Projected points')

# %% comparar fiteos. true corresponde a lo que da chessboard
distCoeffsTrue
distCoeffsOpt
pc.plotRationalDist(distCoeffsTrue,imgSize, linearCoeffsTrue)
pc.plotRationalDist(distCoeffsOpt,imgSize, cameraMatrixOpt)

linearCoeffsTrue
cameraMatrixOpt

rVecsTrue[i]
rVecsOpt[i] # da muy parecido pero con los signos al reves!??

tVecsTrue[i]
tVecsOpt[i] # muy parecido

np.linalg.norm(rVecsTrue[9])
np.linalg.norm(rVecsOpt[9])


# %% ver porque el signo cambiado en rVecs, que significa?
r1, r2 = rVecsTrue[1,:,0],rVecsOpt[1,:,0]
r1
r2
np.linalg.norm(r1), np.linalg.norm(r2)

distance = np.array([np.linalg.norm(rVecsTrue[j,:,0] - rVecsOpt[j,:,0]) for j in range(len(rVecsTrue))]) / np.pi / 2
# es por la periodicidad en 2pi
plt.figure()
plt.plot(distance)
'''
pero no quiere decir que la camara apunta hacia donde debe. pero da igual
que opencv al menos
'''
