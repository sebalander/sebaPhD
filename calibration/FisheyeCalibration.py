# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:01:57 2016

@author: sebalander
"""
from numpy import zeros, sqrt, roots, array, isreal, tan
from cv2 import Rodrigues
from cv2.fisheye import projectPoints
from lmfit import minimize, Parameters
#from poseCalibration import xypToZplane
from calibration import poseCalibration
xypToZplane = poseCalibration.xypToZplane

# %% ========== ========== Fisheye PARAMETER HANDLING ========== ==========
def formatParameters(rVec, tVec, linearCoeffs, distCoeffs):
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
    
    for i in range(4):
        params.add('distCoeffs%d'%i,
                   value=distCoeffs[i,0], vary=False)
    
    return params

def retrieveParameters(params):
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
    
    distCoeffs = zeros((4,1))
    for i in range(4):
        distCoeffs[i,0] = params['distCoeffs%d'%i].value
    
    return rvec, tvec, cameraMatrix, distCoeffs


# %% ========== ========== DIRECT Fisheye ========== ==========
def direct(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs):
    projected, _ = projectPoints(fiducialPoints,
                                        rVec,
                                        tVec,
                                        linearCoeffs,
                                        distCoeffs)
    
    return projected

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


# %% ========== ========== INVERSE Fisheye ========== ==========
def inverse(imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    
    if rVec.shape != (3,3):
        rVec, _ = Rodrigues(rVec)
    
    xpp = ((imageCorners[:,0,0]-linearCoeffs[0,2]) /
            linearCoeffs[0,0])
    ypp = ((imageCorners[:,0,1]-linearCoeffs[1,2]) /
            linearCoeffs[1,1])
    rpp = sqrt(xpp**2 + ypp**2)
    
    # polynomial coeffs, grade 9
    # # (k1,k2,k3,k4)
    poly = [[distCoeffs[3,0], # k4
             0,
             distCoeffs[2,0], # k3
             0,
             distCoeffs[1,0], # k2
             0,
             distCoeffs[0,0], # k1
             1,
             -r] for r in rpp]
    
    rootsPoly = [roots(p) for p in poly]
    thetap = array([roo[isreal(roo)].real[0] for roo in rootsPoly]) / rpp
    
    rp = tan(thetap)
    rp_rpp = rp/rpp
    
    xp = xpp * rp_rpp
    yp = ypp * rp_rpp
    
    # project to z=0 plane. perhaps calculate faster with homography function?
    XYZ = xypToZplane(xp, yp, rVec, tVec)
    
    return XYZ

def residualInverse(params, fiducialPoints, imageCorners):
    '''
    '''
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


# %% INTRINSIC CALIBRATION CHESSBOARD METHOD
from cv2.fisheye import calibrate as feCal


def calibrateIntrinsic(objpoints, imgpoints, imgSize, K, D,
                       flags, criteria):
    '''
    calibrates using opencvs implemented algorithm using chessboard method.
    if K, D, flags and criteria are not given it uses the default defined by
    me:
        D = zeros((4))
    
    return rms, K, D, rVecs, tVecs
    '''
    if D is None:
        D = zeros((4))
    
    rms, K, D, rVecs, tVecs = feCal(objpoints, imgpoints,
                                      imgSize, K, D,
                                      flags=flags, criteria=criteria)
    
    return rms, K, D, rVecs, tVecs














