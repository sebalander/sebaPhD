# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:52:00 2016

implements rational camera inverse mapping

@author: sebalander
"""
# %%
import numpy as np


# %%
def inverseRational(corners, cameraMatrix, distCoeffs, rotMatrix, tVec):
    '''
    inverseRational(corners, cameraMatrix, distCoeffs) -> X,Y
    takes corners in image and returns (X,Y,0) in scene
    corners must be of size (n,1,2)
    '''
    xpp = (corners[:,0,0]-cameraMatrix[0,2]) / cameraMatrix[0,0]
    ypp = (corners[:,0,1]-cameraMatrix[1,2]) / cameraMatrix[1,1]
    rpp = np.sqrt(xpp**2 + ypp**2)
    
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
    
    # roots = np.roots(p)
    roots = [np.roots(p) for p in poly]
#    # max radious possible
#    rppMax = np.sqrt((cameraMatrix[0,2] / cameraMatrix[0,0])**2 +
#                     (cameraMatrix[1,2] / cameraMatrix[1,1])**2)
    
    # rp_rpp = roots[np.isreal(roots)].real[0] / rpp # asume real positive root, in interval
    rp_rpp = np.array([roo[np.isreal(roo)].real[0] for roo in roots]) / rpp
    
    xp = xpp * rp_rpp
    yp = ypp * rp_rpp
    
    # auxiliar calculations
    a = rotMatrix[0,0] - rotMatrix[2,0] * xp
    b = rotMatrix[0,1] - rotMatrix[2,1] * xp
    c = tVec[0,0] - tVec[2,0] * xp
    d = rotMatrix[1,0] - rotMatrix[2,0] * yp
    e = rotMatrix[1,1] - rotMatrix[2,1] * yp
    f = tVec[1,0] - tVec[2,0] * yp
    q = a*e-d*b
    
    X = (c*e - f*b)/q
    Y = (f*a - c*d)/q
    
    shape = (corners.shape[0],1,3)
    XYZ = np.array([X, Y, np.zeros(shape[0])])
    XYZ = np.reshape(XYZ, shape)
    
    return XYZ