# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:47:18 2016

defines the class poseCalibrator that recieves extrinsic calibration data:
- fiducial points
- corners found on image that correspond to the fiducial points
- camera intrinsic parameters
- initial pose guess for pose
and calculates:
- optimum pose minimizing error in image
- optimum pose minimizing error in scene

also perhaps plots things and in the future calculates error.
long term: help user define initial pose guess, help find corners in image

following tutorial in http://www.tutorialspoint.com/python/python_classes_objects.htm

@author: sebalander
"""
# %% IMPORTS
from numpy import zeros, sqrt, roots, array, isreal, reshape, tan, arctan
from cv2 import projectPoints, Rodrigues
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


# %% ========== ========== RATIONAL PARAMETER HANDLING ========== ==========
def formatParametersRational(rVec, tVec, linearCoeffs, distCoeffs):
    '''
    get params into correct format for lmfit optimisation
    '''
    params = Parameters()
    for i in range(3):
        params.add('rvec%d'%i,
                   value=rVec[i,0], vary=True)
        params.add('tvec%d'%i,
                   value=tVec[i,0], vary=True)
    
    params.add('cameraMatrix0',
               value=linearCoeffs[0,0], vary=False)
    params.add('cameraMatrix1',
               value=linearCoeffs[1,1], vary=False)
    params.add('cameraMatrix2',
               value=linearCoeffs[0,2], vary=False)
    params.add('cameraMatrix3',
               value=linearCoeffs[1,2], vary=False)
    
    for i in range(14):
        params.add('distCoeffs%d'%i,
                   value=distCoeffs[i,0], vary=False)
    
    return params

def retrieveParametersRational(params):
    '''
    
    '''
    rvec = zeros((3,1))
    tvec = zeros((3,1))
    for i in range(3):
        rvec[i,0] = params['rvec%d'%i].value
        tvec[i,0] = params['tvec%d'%i].value
    
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


# %% ========== ========== DIRECT RATIONAL ========== ==========
def directRational(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    calls opencv's projection function cv2.projectPoints
    '''
    projected, _ = projectPoints(fiducialPoints,
                                        rVec,
                                        tVec,
                                        linearCoeffs,
                                        distCoeffs)
    
    return projected

def residualDirectRational(params, fiducialPoints, imageCorners):
    '''
    '''
    rVec, tVec, linearCoeffs, distCoeffs = retrieveParametersRational(params)
    
    projectedCorners = directRational(fiducialPoints,
                                      rVec,
                                      tVec,
                                      linearCoeffs,
                                      distCoeffs)
    
    return imageCorners[:,0,:] - projectedCorners[:,0,:]


def calibrateDirectRational(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    returns optimised rvecOpt, tvecOpt
    
    '''
    initialParams = formatParametersRational(rVec, tVec, linearCoeffs, distCoeffs) # generate Parameters obj
    
    out = minimize(residualDirectRational,
                   initialParams,
                   args=(fiducialPoints,
                         imageCorners))
    
    rvecOpt, tvecOpt, _, _ = retrieveParametersRational(out.params)
    
    return rvecOpt, tvecOpt, out.params


# %% ========== ========== INVERSE RATIONAL ========== ==========
def inverseRational(imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    inverseRational(objPoints, rVec/rotMatrix, tVec, cameraMatrix,
                    distCoeffs)-> objPoints
    takes corners in image and returns coordinates in scene
    corners must be of size (n,1,2)
    objPoints has size (1,n,3)
    ignores tangential and tilt distortions
    '''
    
    if rVec.shape != (3,3):
        rVec, _ = Rodrigues(rVec)
    
    xpp = ((imageCorners[:,0,0]-linearCoeffs[0,2]) /
            linearCoeffs[0,0])
    ypp = ((imageCorners[:,0,1]-linearCoeffs[1,2]) /
            linearCoeffs[1,1])
    rpp = sqrt(xpp**2 + ypp**2)
    
    # polynomial coeffs
    # # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    poly = [[distCoeffs[4,0],
             -r*distCoeffs[7,0],
             distCoeffs[1,0],
             -r*distCoeffs[6,0],
             distCoeffs[0,0],
             -r*distCoeffs[5,0],
             1,
             -r] for r in rpp]
    
    rootsPoly = [roots(p) for p in poly]
    rp_rpp = array([roo[isreal(roo)].real[0] for roo in rootsPoly]) / rpp
    
    xp = xpp * rp_rpp
    yp = ypp * rp_rpp
    
    # project to z=0 plane. perhaps calculate faster with homography function?
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

def residualInverseRational(params, fiducialPoints, imageCorners):
    '''
    '''
    rVec, tVec, linearCoeffs, distCoeffs = retrieveParametersRational(params)
    
    projectedFiducialPoints = inverseRational(imageCorners,
                                              rVec,
                                              tVec,
                                              linearCoeffs,
                                              distCoeffs)
    
    return fiducialPoints[0,:,:2] - projectedFiducialPoints[0,:,:2]

def calibrateInverseRational(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    returns optimised rvecOpt, tvecOpt
    
    '''
    initialParams = formatParametersRational(rVec, tVec, linearCoeffs, distCoeffs) # generate Parameters obj
    
    out = minimize(residualInverseRational,
                   initialParams,
                   args=(fiducialPoints,
                         imageCorners))
    
    rvecOpt, tvecOpt, _, _ = retrieveParametersRational(out.params)
    
    return rvecOpt, tvecOpt, out.params


# %% ========== ========== DIRECT FISHEYE ========== ==========
def directFisheye():
    '''TODO'''
# %% ========== ========== INVERSE FISHEYE ========== ==========
def inverseFisheye():
    '''TODO'''

