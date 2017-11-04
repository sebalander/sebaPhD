# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:00:40 2016

@author: sebalander
"""
from numpy import zeros, sqrt, array, tan, arctan, prod, cos
from cv2 import  Rodrigues
from lmfit import minimize, Parameters
#from calibration import calibrator
#xypToZplane = calibrator.xypToZplane
#
## %% ========== ==========  PARAMETER HANDLING ========== ==========
#def formatParameters(rVec, tVec, linearCoeffs, distCoeffs):
#    params = Parameters()
#    
#    if prod(rVec.shape) == 9:
#        rVec = Rodrigues(rVec)[0]
#    
#    rVec = rVec.reshape(3)
#    
#    for i in range(3):
#        params.add('rvec%d'%i,
#                   value=rVec[i], vary=True)
#        params.add('tvec%d'%i,
#                   value=tVec[i], vary=True)
#    
#    # image center
#    params.add('cameraMatrix0',
#               value=linearCoeffs[0], vary=False)
#    params.add('cameraMatrix1',
#               value=linearCoeffs[1], vary=False)
#    
#    # k
#    params.add('distCoeffs',
#               value=distCoeffs, vary=False)
#    
#    return params
#
#def retrieveParameters(params):
#    '''
#    
#    '''
#    rvec = zeros((3,1))
#    tvec = zeros((3,1))
#    for i in range(3):
#        rvec[i,0] = params['rvec%d'%i].value
#        tvec[i,0] = params['tvec%d'%i].value
#    
#    cameraMatrix = zeros(2)
#    cameraMatrix[0] = params['cameraMatrix0'].value
#    cameraMatrix[1] = params['cameraMatrix1'].value
#    
#    distCoeffs = params['distCoeffs'].value
#    
#    return rvec, tvec, cameraMatrix, distCoeffs


# %% ========== ========== DIRECT  ========== ==========
def radialDistort(rp, k, quot=False, der=False):
    '''
    returns distorted radius using distortion coefficient k
    optionally it returns the distortion quotioent rpp = rp * q
    '''
    k.shape = 1
    
    th = arctan(rp)
    tanth = tan(th/2)
    rpp = k * tanth
    
    if der:
        # derivative of quot of polynomials, "Direct" mapping
        # rpp wrt rp
        dPPdP = k / cos(th / 2)**2 / 2 / (1 + rp**2)
        # calculate quotient
        q = rpp / rp
        # q wrt rpp 
        dQdP = ((dPPdP - q) / rp).reshape((1,-1)) # deriv wrt undistorted coords
        
        dQdK = (tanth).reshape((1,-1))
        
        if quot:
            return q, dQdP, dQdK
        else:
            return rpp, dQdP, dQdK
    else:
        if quot:
            return rpp / rp
        else:
            return rpp


## we asume that intrinsic distortion paramters is just a scalar: distCoeffs=k
#def direct(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs):
#    # format as matrix
#    try:
#        rVec.reshape(3)
#        rVec = Rodrigues(rVec)[0]
#    except:
#        pass
#    
#    xyz = rVec.dot(fiducialPoints[0].T)+tVec
#    
#    xp = xyz[0]/xyz[2]
#    yp = xyz[1]/xyz[2]
#    
#    rp = sqrt(xp**2 + yp**2)
#    thetap = arctan(rp)
#    
#    rpp = distCoeffs*tan(thetap/2)
#    
#    rpp_rp = rpp/rp
#    
#    xpp = xp*rpp_rp
#    ypp = yp*rpp_rp
#    
#    u = xpp + linearCoeffs[0]
#    v = ypp + linearCoeffs[1]
#    
#    return array([u,v]).reshape((fiducialPoints.shape[1],1,2))
#
#def residualDirect(params, fiducialPoints, imageCorners):
#    rVec, tVec, linearCoeffs, distCoeffs = retrieveParameters(params)
#    
#    projectedCorners = direct(fiducialPoints,
#                                      rVec,
#                                      tVec,
#                                      linearCoeffs,
#                                      distCoeffs)
#    
#    return imageCorners[:,0,:] - projectedCorners[:,0,:]
#
#def calibrateDirect(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
#    initialParams = formatParameters(rVec, tVec, linearCoeffs, distCoeffs) # generate Parameters obj
#    
#    out = minimize(residualDirect,
#                   initialParams,
#                   args=(fiducialPoints,
#                         imageCorners))
#    
#    rvecOpt, tvecOpt, _, _ = retrieveParameters(out.params)
#    
#    return rvecOpt, tvecOpt, out.params


# %% ========== ========== INVERSE  ========== ==========

def radialUndistort(rpp, k, quot=False, der=False):
    '''
    takes distorted radius and returns the radius undistorted
    optioally it returns the undistortion quotient rpp = rp * q
    '''
    # polynomial coeffs, grade 7
    # # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    k.shape = -1
    
    thetap = 2 * arctan(rpp / k)
    
    rp = tan(thetap)
    
    retVal = True
    
    if der:
        # derivada de la directa
        q, dQdP, dQdK = radialDistort(rp, k, quot, der)
        
        if quot:
            return q, retVal, dQdP, dQdK
        else:
            return rp, retVal, dQdP, dQdK
    else:
        if quot:
            # returns q
            return rpp / rp, retVal
        else:
            return rp, retVal




#def inverse(imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
#    
#    xpp = imageCorners[:,0,0]-linearCoeffs[0]
#    ypp = imageCorners[:,0,1]-linearCoeffs[1]
#    rpp = sqrt(xpp**2 + ypp**2)
#    
#    thetap = 2*arctan(rpp/distCoeffs)
#    
#    rp = tan(thetap)
#    
#    rp_rpp = rp/rpp
#    
#    xp = xpp * rp_rpp
#    yp = ypp * rp_rpp
#    
#    # project to z=0 plane. perhaps calculate faster with homography function?
#    XYZ = xypToZplane(xp, yp, rVec, tVec)
#    
#    return XYZ
#
#
#def residualInverse(params, fiducialPoints, imageCorners):
#    rVec, tVec, linearCoeffs, distCoeffs = retrieveParameters(params)
#    
#    projectedFiducialPoints = inverse(imageCorners,
#                                              rVec,
#                                              tVec,
#                                              linearCoeffs,
#                                              distCoeffs)
#    
#    return fiducialPoints[0,:,:2] - projectedFiducialPoints[0,:,:2]
#
#def calibrateInverse(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
#    initialParams = formatParameters(rVec, tVec, linearCoeffs, distCoeffs) # generate Parameters obj
#    
#    out = minimize(residualInverse,
#                   initialParams,
#                   args=(fiducialPoints,
#                         imageCorners))
#    
#    rvecOpt, tvecOpt, _, _ = retrieveParameters(out.params)
#    
#    return rvecOpt, tvecOpt, out.params
