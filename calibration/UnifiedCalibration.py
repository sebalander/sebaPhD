# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:14:17 2016

@author: sebalander
"""
from numpy import zeros, sqrt, array, prod
from cv2 import  Rodrigues
from lmfit import minimize, Parameters
from calibration import calibrator
xypToZplane = calibrator.xypToZplane

# %% ========== ==========  PARAMETER HANDLING ========== ==========
def formatParameters(rVec, tVec, linearCoeffs, distCoeffs):
    params = Parameters()
    
    if prod(rVec.shape) == 9:
        rVec = Rodrigues(rVec)[0]
    
    rVec = rVec.reshape(3)
    
    for i in range(3):
        params.add('rvec%d'%i,
                   value=rVec[i], vary=True)
        params.add('tvec%d'%i,
                   value=tVec[i], vary=True)
    
    # image center
    params.add('cameraMatrix0',
               value=linearCoeffs[0], vary=False)
    params.add('cameraMatrix1',
               value=linearCoeffs[1], vary=False)
    
    # l and m
    params.add('distCoeffs0',
               value=distCoeffs[0], vary=False)
    params.add('distCoeffs1',
               value=distCoeffs[1], vary=False)
    return params



def retrieveParameters(params):
    rvec = zeros((3,1))
    tvec = zeros((3,1))
    for i in range(3):
        rvec[i,0] = params['rvec%d'%i].value
        tvec[i,0] = params['tvec%d'%i].value
    
    cameraMatrix = zeros(2)
    cameraMatrix[0] = params['cameraMatrix0'].value
    cameraMatrix[1] = params['cameraMatrix1'].value
    
    distCoeffs = zeros(2)
    distCoeffs[0] = params['distCoeffs0'].value
    distCoeffs[1] = params['distCoeffs1'].value
    return rvec, tvec, cameraMatrix, distCoeffs


# %% ========== ========== DIRECT  ========== ==========
# we asume that intrinsic distortion paramters is just a scalar: distCoeffs=k
def direct(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs):
    # format as matrix
    try:
        rVec.reshape(3)
        rVec = Rodrigues(rVec)[0]
    except:
        pass
    
    xyz = rVec.dot(fiducialPoints[0].T)+tVec
    
    xp = xyz[0]/xyz[2]
    yp = xyz[1]/xyz[2]
    
    rp2 = xp**2 + yp**2
    
    aux = (distCoeffs[0]+distCoeffs[1]) / (1 + distCoeffs[0] * sqrt(1+rp2))
    
    xpp = xp*aux
    ypp = yp*aux
    
    u = xpp + linearCoeffs[0]
    v = ypp + linearCoeffs[1]
    
    return array([u,v]).reshape((fiducialPoints.shape[1],1,2))

def residualDirect(params, fiducialPoints, imageCorners):
    rVec, tVec, linearCoeffs, distCoeffs = retrieveParameters(params)
    
    projectedCorners = direct(fiducialPoints,
                                      rVec,
                                      tVec,
                                      linearCoeffs,
                                      distCoeffs)
    
    return imageCorners[:,0,:] - projectedCorners[:,0,:]

def calibrateDirect(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    initialParams = formatParameters(rVec, tVec, linearCoeffs, distCoeffs) # generate Parameters obj
    
    out = minimize(residualDirect,
                   initialParams,
                   args=(fiducialPoints,
                         imageCorners))
    
    rvecOpt, tvecOpt, _, _ = retrieveParameters(out.params)
    
    return rvecOpt, tvecOpt, out.params


# %% ========== ========== INVERSE  ========== ==========
def inverse(imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    
    xpp = imageCorners[:,0,0] - linearCoeffs[0]
    ypp = imageCorners[:,0,1] - linearCoeffs[1]
    
    lm = distCoeffs[0] * (distCoeffs[0] + distCoeffs[1])
    
    aux1 = (lm/xpp - distCoeffs[0])**2
    aux2 = (lm/ypp - distCoeffs[0])**2
    denom = (aux1 - 1)*(aux2 - 1) - 1
    
    xp = xpp * aux1 / denom
    yp = ypp * aux2 / denom
    
    # project to z=0 plane. perhaps calculate faster with homography function?
    XYZ = xypToZplane(xp, yp, rVec, tVec)
    
    return XYZ


def residualInverse(params, fiducialPoints, imageCorners):
    rVec, tVec, linearCoeffs, distCoeffs = retrieveParameters(params)
    
    projectedFiducialPoints = inverse(imageCorners,
                                              rVec,
                                              tVec,
                                              linearCoeffs,
                                              distCoeffs)
    
    return fiducialPoints[0,:,:2] - projectedFiducialPoints[0,:,:2]

def calibrateInverse(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    initialParams = formatParameters(rVec, tVec, linearCoeffs, distCoeffs) # generate Parameters obj
    
    out = minimize(residualInverse,
                   initialParams,
                   args=(fiducialPoints,
                         imageCorners))
    
    rvecOpt, tvecOpt, _, _ = retrieveParameters(out.params)
    
    return rvecOpt, tvecOpt, out.params
