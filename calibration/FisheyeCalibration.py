# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:01:57 2016

@author: sebalander
"""
from numpy import zeros, roots, array, isreal, tan, prod, arctan
from numpy import any, empty_like, arange, real, pi, abs
from cv2 import Rodrigues
#from cv2.fisheye import projectPoints
from lmfit import  Parameters # , minimize
#from calibration import calibrator
#xypToZplane = calibrator.xypToZplane

# %% ========== ========== Fisheye PARAMETER HANDLING ========== ==========
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
def radialDistort(rp, k, quot=False, der=False):
    '''
    returns distorted radius using distortion coefficients k
    optionally it returns the distortion quotioent rpp = rp * q
    '''
    th = arctan(rp)
    th2 = th*th
    th4 = th2*th2
    th6 = th2*th4
    th8 = th4*th4
    
    k.shape = -1
    rpp = (1 + k[0]*th2 + k[1]*th4 + k[2]*th6 + k[3]*th8) * th
    
    if der:
        # derivative of quot of polynomials, "Direct" mapping
        # rpp wrt rp
        dPPdP = (1 + 3 * k[0]*th2 + 5 * k[1]*th4 + 7 * k[2]*th6
                   + 9 * k[3]*th8) / (1 + rp**2)
        # calculate quotient
        q = rpp / rp
        # q wrt rpp 
        dQdP = (dPPdP - q) / rp # deriv wrt undistorted coords
        
        dQdK = array([th2, th4, th6, th8]) * th
        
        if quot:
            return q, dQdP, dQdK
        else:
            return rpp, dQdP, dQdK
    else:
        if quot:
            return rpp / rp
        else:
            return rpp


# %% ========== ========== INVERSE Fisheye ========== ==========
def radialUndistort(rpp, k, quot=False, der=False):
    '''
    takes distorted radius and returns the radius undistorted
    optionally it returns the distortion quotioent rpp = rp * q
    '''
    # polynomial coeffs, grade 7
    # # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    # polynomial coeffs, grade 7
    # # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    
    k.shape = -1
    poly = [[k[3], 0, k[2], 0, k[1], 0, k[0], 0, 1, -r] for r in rpp]
    
    # calculate roots
    rootsPoly = array([roots(p) for p in poly])
    
    # return flag, True if there is a suitable (real AND positive) solution
    rPRB = isreal(rootsPoly) & (0 <= real(rootsPoly))  # real Positive Real Bool
    retVal = any(rPRB, axis=1)
    
    thetap = empty_like(rpp)
    
    if any(~retVal): # if at least one case of non solution
        # calculate extrema of polyniomial
        thExtrema = roots([9*k[3], 0, 7*k[2], 0, 5*k[1], 0, 3*k[0], 0, 1])
        # select real extrema in positive side, keep smallest
        thRealPos = min(thExtrema.real[isreal(thExtrema) & (0<=thExtrema.real)])
        # assign to problematic values
        thetap[~retVal] = thRealPos
    
    # choose minimum positive roots
    thetap[retVal] = [min(rootsPoly[i, rPRB[i]].real)
                   for i in arange(rpp.shape[0])[retVal]]
    
    rp = abs(tan(thetap))  # correct negative values
    # if theta angle is greater than pi/2, retVal=False
    retVal[thetap >= pi/2] = False
    
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

