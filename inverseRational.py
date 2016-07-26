# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:52:00 2016

implements rational camera inverse mapping

@author: sebalander
"""
# %%
import numpy as np


# %%
def inverseRational(u, v, cameraMatrix, distCoeffs):
    xpp = (u-cameraMatrix[0,2]) / cameraMatrix[0,0]
    ypp = (v-cameraMatrix[1,2]) / cameraMatrix[1,0]
    rpp = np.sqrt(xpp**2 + ypp**2)
    
    # polynomial coeffs
    p = [distCoeffs[2],
         -rpp*distCoeffs[5],
         distCoeffs[1],
         -rpp*distCoeffs[4],
         distCoeffs[0],
         -rpp*distCoeffs[3],
         1,
         -rpp]
    
    roots = np.roots(p)
    rp_rpp = roots[np.isreal(roots)]/rpp # asume only one real root,
    
    xp = xpp*rp_rpp
    yp = ypp*rp_rpp
    
    # auxiliar calculations
    a = cameraMatrix[0,0] - cameraMatrix[2,0] * xp
    b = cameraMatrix[0,1] - cameraMatrix[2,1] * xp
    c = cameraMatrix[0,3] - cameraMatrix[2,3] * xp
    d = cameraMatrix[1,0] - cameraMatrix[2,0] * yp
    e = cameraMatrix[1,1] - cameraMatrix[2,1] * yp
    f = cameraMatrix[1,3] - cameraMatrix[2,3] * yp
    q = a*e-d*b
    
    X = (c*e - f*b)/q
    Y = (f*a - c*d)/q
    
    return X,Y