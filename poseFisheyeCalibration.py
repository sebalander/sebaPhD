# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:01:57 2016

@author: sebalander
"""
from numpy import zeros, sqrt, roots, array, isreal, reshape, tan
from cv2 import Rodrigues
from cv2.fisheye import projectPoints
from lmfit import minimize, Parameters

# %% ========== ========== Fisheye PARAMETER HANDLING ========== ==========
def formatParametersFisheye(rVec, tVec, linearCoeffs, distCoeffs):
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
    
    for i in range(4):
        params.add('distCoeffs%d'%i,
                   value=distCoeffs[i,0], vary=False)
    
    return params

def retrieveParametersFisheye(params):
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
    
    distCoeffs = zeros((4,1))
    for i in range(4):
        distCoeffs[i,0] = params['distCoeffs%d'%i].value
    
    return rvec, tvec, cameraMatrix, distCoeffs


# %% ========== ========== DIRECT Fisheye ========== ==========
def directFisheye(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    calls opencv's projection function cv2.projectPoints
    '''
    projected, _ = projectPoints(fiducialPoints,
                                        rVec,
                                        tVec,
                                        linearCoeffs,
                                        distCoeffs)
    
    return projected

def residualDirectFisheye(params, fiducialPoints, imageCorners):
    '''
    '''
    rVec, tVec, linearCoeffs, distCoeffs = retrieveParametersFisheye(params)
    
    projectedCorners = directFisheye(fiducialPoints,
                                      rVec,
                                      tVec,
                                      linearCoeffs,
                                      distCoeffs)
    
    return imageCorners[:,0,:] - projectedCorners[:,0,:]


def calibrateDirectFisheye(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    returns optimised rvecOpt, tvecOpt
    
    '''
    initialParams = formatParametersFisheye(rVec, tVec, linearCoeffs, distCoeffs) # generate Parameters obj
    
    out = minimize(residualDirectFisheye,
                   initialParams,
                   args=(fiducialPoints,
                         imageCorners))
    
    rvecOpt, tvecOpt, _, _ = retrieveParametersFisheye(out.params)
    
    return rvecOpt, tvecOpt, out.params


# %% ========== ========== INVERSE Fisheye ========== ==========
def inverseFisheye(imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    inverseFisheye(objPoints, rVec/rotMatrix, tVec, cameraMatrix,
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

def residualInverseFisheye(params, fiducialPoints, imageCorners):
    '''
    '''
    rVec, tVec, linearCoeffs, distCoeffs = retrieveParametersFisheye(params)
    
    projectedFiducialPoints = inverseFisheye(imageCorners,
                                              rVec,
                                              tVec,
                                              linearCoeffs,
                                              distCoeffs)
    
    return fiducialPoints[0,:,:2] - projectedFiducialPoints[0,:,:2]

def calibrateInverseFisheye(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
    '''
    returns optimised rvecOpt, tvecOpt
    
    '''
    initialParams = formatParametersFisheye(rVec, tVec, linearCoeffs, distCoeffs) # generate Parameters obj
    
    out = minimize(residualInverseFisheye,
                   initialParams,
                   args=(fiducialPoints,
                         imageCorners))
    
    rvecOpt, tvecOpt, _, _ = retrieveParametersFisheye(out.params)
    
    return rvecOpt, tvecOpt, out.params
