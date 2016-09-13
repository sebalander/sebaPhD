# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:14:17 2016

@author: sebalander
"""
from numpy import zeros, sqrt, array, reshape, tan, arctan
from cv2 import  Rodrigues
from lmfit import minimize, Parameters

# %% ========== ========== Unified PARAMETER HANDLING ========== ==========
def formatParametersUnified(rVec, tVec, linearCoeffs, distCoeffs):
    '''
    get params into correct format for lmfit optimisation
    '''
    params = Parameters()
    for i in range(3):
        params.add('rvec%d'%i,
                   value=rVec[i,0], vary=True)
        params.add('tvec%d'%i,
                   value=tVec[i,0], vary=True)
    
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

def retrieveParametersUnified(params):
    '''
    
    '''
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


# %% ========== ========== DIRECT Unified ========== ==========
# we asume that intrinsic distortion paramters is just a scalar: distCoeffs=k
def directUnified(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    '''
    # format as matrix
    if rVec.shape != (3,3):
        rVec, _ = Rodrigues(rVec)
    
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

def residualDirectUnified(params, fiducialPoints, imageCorners):
    '''
    '''
    rVec, tVec, linearCoeffs, distCoeffs = retrieveParametersUnified(params)
    
    projectedCorners = directUnified(fiducialPoints,
                                      rVec,
                                      tVec,
                                      linearCoeffs,
                                      distCoeffs)
    
    return imageCorners[:,0,:] - projectedCorners[:,0,:]

def calibrateDirectUnified(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    returns optimised rvecOpt, tvecOpt
    '''
    initialParams = formatParametersUnified(rVec, tVec, linearCoeffs, distCoeffs) # generate Parameters obj
    
    out = minimize(residualDirectUnified,
                   initialParams,
                   args=(fiducialPoints,
                         imageCorners))
    
    rvecOpt, tvecOpt, _, _ = retrieveParametersUnified(out.params)
    
    return rvecOpt, tvecOpt, out.params


# %% ========== ========== INVERSE Unified ========== ==========
def inverseUnified(imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    inverseRational(objPoints, rVec/rotMatrix, tVec, cameraMatrix,
                    distCoeffs)-> objPoints
    takes corners in image and returns coordinates in scene
    corners must be of size (n,1,2)
    objPoints has size (1,n,3)
    ignores tangential and tilt distortions
    '''
    # format as matrix
    if rVec.shape != (3,3):
        rVec, _ = Rodrigues(rVec)
    
    xpp = imageCorners[:,0,0] - linearCoeffs[0]
    ypp = imageCorners[:,0,1] - linearCoeffs[1]
    
    lm = distCoeffs[0] * (distCoeffs[0] + distCoeffs[1])
    
    aux1 = (lm/xpp - distCoeffs[0])**2
    aux2 = (lm/ypp - distCoeffs[0])**2
    denom = (aux1 - 1)*(aux2 - 1) - 1
    
    xp = xpp * aux1 / denom
    yp = ypp * aux2 / denom
    
    # auxiliar calculations
    a = rVec[0,0] - rVec[2,0] * xp
    b = rVec[0,1] - rVec[2,1] * xp
    c = tVec[0,0] - tVec[2,0] * xp
    d = rVec[1,0] - rVec[2,0] * yp
    e = rVec[1,1] - rVec[2,1] * yp
    f = tVec[1,0] - tVec[2,0] * yp
    q = a*e-d*b
    
    X = -(c*e - f*b)/q # check why wrong sign, why must put '-' in front?
    Y = -(f*a - c*d)/q
    
    shape = (1,imageCorners.shape[0],3)
    XYZ = array([X, Y, zeros(shape[1])]).T
    XYZ = reshape(XYZ, shape)
    
    return XYZ


def residualInverseUnified(params, fiducialPoints, imageCorners):
    '''
    '''
    rVec, tVec, linearCoeffs, distCoeffs = retrieveParametersUnified(params)
    
    projectedFiducialPoints = inverseUnified(imageCorners,
                                              rVec,
                                              tVec,
                                              linearCoeffs,
                                              distCoeffs)
    
    return fiducialPoints[0,:,:2] - projectedFiducialPoints[0,:,:2]

def calibrateInverseUnified(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    returns optimised rvecOpt, tvecOpt
    
    '''
    initialParams = formatParametersUnified(rVec, tVec, linearCoeffs, distCoeffs) # generate Parameters obj
    
    out = minimize(residualInverseUnified,
                   initialParams,
                   args=(fiducialPoints,
                         imageCorners))
    
    rvecOpt, tvecOpt, _, _ = retrieveParametersUnified(out.params)
    
    return rvecOpt, tvecOpt, out.params
