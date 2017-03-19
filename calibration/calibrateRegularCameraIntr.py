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
    
    params.add('fX', value=linearCoeffs[0,0], vary=True)
    params.add('fY', value=linearCoeffs[1,1], vary=True)
    params.add('cX', value=linearCoeffs[0,2], vary=True)
    params.add('cY', value=linearCoeffs[1,2], vary=True)
    
    # polynomial coeffs, grade 7
    # # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    #for i in [2,3,8,9,10,11,12,13]:
    #    params.add('distCoeffs%d'%i,
    #               value=distCoeffs[i,0], vary=False)
    
    params.add('numDist0', value=distCoeffs[0,0], vary=True)
    params.add('numDist1', value=distCoeffs[1,0], vary=True)
    params.add('numDist2', value=distCoeffs[4,0], vary=True)
    
    params.add('denomDist0', value=distCoeffs[5,0], vary=True)
    params.add('denomDist1', value=distCoeffs[6,0], vary=True)
    params.add('denomDist2', value=distCoeffs[7,0], vary=True)
    
    return params

# %%

def retrieveParametersChess(params):
    
    n = len([0 for x in params.iterkeys()])/6 - 3
    
    rvec = zeros((n,3,1))
    tvec = zeros((n,3,1))
    
    for j in range(n):
        for i in range(3):
            rvec[j,i,0] = params['rvec%d%d'%(j,i)].value
            tvec[j,i,0] = params['tvec%d%d'%(j,i)].value
    
    cameraMatrix = zeros((3,3))
    cameraMatrix[0,0] = params['fX'].value
    cameraMatrix[1,1] = params['fY'].value
    cameraMatrix[0,2] = params['cX'].value
    cameraMatrix[1,2] = params['cY'].value
    cameraMatrix[2,2] = 1
    
    distCoeffs = zeros((14,1))
    
    distCoeffs[0] = params['numDist0'].value
    distCoeffs[1] = params['numDist1'].value
    distCoeffs[4] = params['numDist2'].value
    
    distCoeffs[5] = params['denomDist0'].value
    distCoeffs[6] = params['denomDist1'].value
    distCoeffs[7] = params['denomDist2'].value
    
    
    return rvec, tvec, cameraMatrix, distCoeffs

# %% change state of paramters

def setDistortionParams(params, state):
    
    for i in [0,1,4,5,6,7]:
        params['distCoeffs%d'%i].vary=state

def setLinearParams(params, state):
    params['cameraMatrix0'].value = state
    params['cameraMatrix1'].value = state
    params['cameraMatrix2'].value = state
    params['cameraMatrix3'].value = state

def setExtrinsicParams(params, state):
    
    n = len([0 for x in params.iterkeys()])/6 - 3
    
    for j in range(n):
        for i in range(3):
            params['rvec%d%d'%(j,i)].vary = state
            params['tvec%d%d'%(j,i)].vary = state

# %% residual

def residualDirectChessRatio(params, fiducialPoints, corners):
    '''
    '''
    n = len(corners)
    rVecs, tVecs, linearCoeffs, distCoeffs = retrieveParametersChess(params)
    E = list()
    
    for j in range(n):
        projectedCorners = rational.direct(fiducialPoints,
                                           rVecs[j],
                                           tVecs[j],
                                           linearCoeffs,
                                           distCoeffs)
        err = projectedCorners[:,0,:] - corners[j,:,0,:]
        E.append(err)
    
    return np.reshape(E,(n*len(fiducialPoints[0]),2))

# %%
def calibrateDirectChessRatio(fiducialPoints, corners, rVecs, tVecs, linearCoeffs, distCoeffs):
    '''
    parece que si no se hace por etapas hay inestabilidad numerica. lo veo
    con la camara plana que en ppio no deberia tener problemas para ajustar y 
    en mucha mayor medida con la fisheye.
    quiza valga la pena iterar ete ciclo la cantidad de veces que sea necesario
    hasta que converja. roguemos que converja
    '''
    params = formatParametersChessIntrs(rVecs, tVecs, linearCoeffs, distCoeffs) # generate Parameters obj
    
    
    setDistortionParams(params,False)
    setLinearParams(params,True)
    setExtrinsicParams(params,True)
    
    out = minimize(residualDirectChessRatio,
                   params,
                   args=(fiducialPoints,
                         corners),
                   xtol=1e-5, # Relative error in the approximate solution
                   ftol=1e-5, # Relative error in the desired sum of squares
                   maxfev=int(1e3))
    '''
    params = out.params
    # setDistortionParams(params,False)
    setLinearParams(params,True)
    setExtrinsicParams(params,False)
    
    out = minimize(residualDirectChessRatio,
                   params,
                   args=(fiducialPoints,
                         corners),
                   xtol=1e-5, # Relative error in the approximate solution
                   ftol=1e-5, # Relative error in the desired sum of squares
                   maxfev=int(1e3))
    
    params = out.params
    setDistortionParams(params,True)
    setLinearParams(params,False)
    # setExtrinsicParams(params,False)
    
    out = minimize(residualDirectChessRatio,
                   params,
                   args=(fiducialPoints,
                         corners),
                   xtol=1e-5, # Relative error in the approximate solution
                   ftol=1e-5, # Relative error in the desired sum of squares
                   maxfev=int(1e3))
    '''
    return out


# %%
reload(pc)

# %% LOAD DATA

### fisheye data
imagesFolder = "./resources/fishChessboard/"
extension = "*.png"
cornersFile = "./resources/fishChessboard/fishCorners.npy"
patternFile = "./resources/chessPattern.npy"
imgShapeFile = "./resources/fishImgShape.npy"

distCoeffsFile = "./resources/fishDistCoeffs.npy"
linearCoeffsFile = "./resources/fishLinearCoeffs.npy"
rvecsFile = "./resources/fishChessboard/fishRvecs.npy"
tvecsFile = "./resources/fishChessboard/fishTvecs.npy"

### ptz data
#imagesFolder = "./resources/PTZchessboard/zoom 0.0/"
#extension = "*.jpg"
#cornersFile = "./resources/PTZchessboard/zoom 0.0/ptzCorners.npy"
#patternFile = "./resources/chessPattern.npy"
#imgShapeFile = "./resources/ptzImgShape.npy"
#
#distCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
#linearCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"
#rvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzRvecs.npy"
#tvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzTvecs.npy"

corners = np.load(cornersFile).transpose((0,2,1,3))
fiducialPoints = np.load(patternFile)
imgSize = np.load(imgShapeFile)
images = glob.glob(imagesFolder+extension)

distCoeffsTrue = np.load(distCoeffsFile)
linearCoeffsTrue = np.load(linearCoeffsFile)
rVecsTrue = np.load(rvecsFile)
tVecsTrue = np.load(tvecsFile)

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
#distCoeffsIni = np.zeros((14, 1))  # despues hacer generico para todos los modelos
#k = 10 # factor en que escalear la distancia focal
#linearCoeffsIni = linearCoeffsTrue * [k,k,1]
distCoeffsIni = distCoeffsTrue
# %% extrinsic parameters initial conditions, from estimated homography
rVecsIni, tVecsIni, Hs = pc.estimateInitialPose(fiducialPoints, corners, linearCoeffsIni)
#rVecsIni = rVecsTrue
#tVecsIni = tVecsTrue
# %% from testposecalibration DIRECT GENERIC CALIBRATION
i=0
img = plt.imread(images[i])

imageCorners = corners[i]
rVec = rVecsIni[i]
tVec = tVecsIni[i]
linearCoeffs = linearCoeffsIni
distCoeffs = distCoeffsIni

# direct mapping with initial conditions
cornersProjected = pc.direct(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs, model)

# plot corners in image
pc.cornerComparison(img, imageCorners, cornersProjected)


# %%
# format parameters

initialParams = formatParametersChessIntrs(rVecsIni, tVecsIni, linearCoeffsIni, distCoeffsIni)

# test retrieving parameters
# n=10
# retrieveParametersChess(initialParams)



# %%
#E = residualDirectChessRatio(initialParams, fiducialPoints, corners)


out = calibrateDirectChessRatio(fiducialPoints, corners, rVecsIni, tVecsIni, linearCoeffsIni, distCoeffsIni)

out.nfev
out.message
out.lmdif_message
(out.residual**2).sum()

# %%
rVecsOpt, tVecsOpt, cameraMatrixOpt, distCoeffsOpt = retrieveParametersChess(out.params)

# %%cameraMatrix0
img = plt.imread(images[i])

imageCorners = corners[i]
rVec = rVecsOpt[i]
tVec = tVecsOpt[i]
linearCoeffs = cameraMatrixOpt
distCoeffs = distCoeffsOpt

# direct mapping with initial conditions
cornersProjected = pc.direct(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs, model)

# plot corners in image
pc.cornerComparison(img, imageCorners, cornersProjected)


# %% comparar fiteos. true corresponde a lo que da chessboard
distCoeffsTrue
distCoeffsOpt
pc.plotRationalDist(distCoeffsTrue,imgSize, linearCoeffsTrue)
pc.plotRationalDist(distCoeffsOpt,imgSize, cameraMatrixOpt)

linearCoeffsTrue
cameraMatrixOpt

rVecsTrue[i]
rVecsOpt[i] 

tVecsTrue[i]
tVecsOpt[i]

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
