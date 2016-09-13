# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:00:40 2016

@author: sebalander
"""
from numpy import zeros, sqrt, array, reshape, tan, arctan
from cv2 import  Rodrigues
from lmfit import minimize, Parameters

# %% ========== ========== STEREOGRAPHIC PARAMETER HANDLING ========== ==========
def formatParametersStereographic(rVec, tVec, linearCoeffs, distCoeffs):
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
    
    # k
    params.add('distCoeffs',
               value=distCoeffs, vary=False)
    
    return params

def retrieveParametersStereographic(params):
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
    
    distCoeffs = params['distCoeffs'].value
    
    return rvec, tvec, cameraMatrix, distCoeffs


# %% ========== ========== DIRECT STEREOGRAPHIC ========== ==========
# we asume that intrinsic distortion paramters is just a scalar: distCoeffs=k
def directStereographic(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    '''
    # format as matrix
    if rVec.shape != (3,3):
        rVec, _ = Rodrigues(rVec)
    
    xyz = rVec.dot(fiducialPoints[0].T)+tVec
    
    xp = xyz[0]/xyz[2]
    yp = xyz[1]/xyz[2]
    
    rp = sqrt(xp**2 + yp**2)
    thetap = arctan(rp)
    
    rpp = distCoeffs*tan(thetap/2)
    
    rpp_rp = rpp/rp
    
    xpp = xp*rpp_rp
    ypp = yp*rpp_rp
    
    u = xpp + linearCoeffs[0]
    v = ypp + linearCoeffs[1]
    
    return array([u,v]).reshape((fiducialPoints.shape[1],1,2))

def residualDirectStereographic(params, fiducialPoints, imageCorners):
    '''
    '''
    rVec, tVec, linearCoeffs, distCoeffs = retrieveParametersStereographic(params)
    
    projectedCorners = directStereographic(fiducialPoints,
                                      rVec,
                                      tVec,
                                      linearCoeffs,
                                      distCoeffs)
    
    return imageCorners[:,0,:] - projectedCorners[:,0,:]

def calibrateDirectStereographic(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    returns optimised rvecOpt, tvecOpt
    '''
    initialParams = formatParametersStereographic(rVec, tVec, linearCoeffs, distCoeffs) # generate Parameters obj
    
    out = minimize(residualDirectStereographic,
                   initialParams,
                   args=(fiducialPoints,
                         imageCorners))
    
    rvecOpt, tvecOpt, _, _ = retrieveParametersStereographic(out.params)
    
    return rvecOpt, tvecOpt, out.params


# %% ========== ========== INVERSE STEREOGRAPHIC ========== ==========
def inverseStereographic(imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
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
    
    xpp = imageCorners[:,0,0]-linearCoeffs[0]
    ypp = imageCorners[:,0,1]-linearCoeffs[1]
    rpp = sqrt(xpp**2 + ypp**2)
    
    thetap = 2*arctan(rpp/distCoeffs)
    
    rp = tan(thetap)
    
    rp_rpp = rp/rpp
    
    xp = xpp * rp_rpp
    yp = ypp * rp_rpp
    
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


def residualInverseStereographic(params, fiducialPoints, imageCorners):
    '''
    '''
    rVec, tVec, linearCoeffs, distCoeffs = retrieveParametersStereographic(params)
    
    projectedFiducialPoints = inverseStereographic(imageCorners,
                                              rVec,
                                              tVec,
                                              linearCoeffs,
                                              distCoeffs)
    
    return fiducialPoints[0,:,:2] - projectedFiducialPoints[0,:,:2]

def calibrateInverseStereographic(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    returns optimised rvecOpt, tvecOpt
    
    '''
    initialParams = formatParametersStereographic(rVec, tVec, linearCoeffs, distCoeffs) # generate Parameters obj
    
    out = minimize(residualInverseStereographic,
                   initialParams,
                   args=(fiducialPoints,
                         imageCorners))
    
    rvecOpt, tvecOpt, _, _ = retrieveParametersStereographic(out.params)
    
    return rvecOpt, tvecOpt, out.params
