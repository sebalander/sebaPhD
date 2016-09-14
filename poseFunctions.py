# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:06:41 2016

@author: sebalander
"""
from numpy import zeros, array, reshape

# %% Z=z PROJECTION

def xypToZplane(xp, yp, rVec, tVec, z=0):
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
    
    shape = (1,X.shape[0],3)
    XYZ = array([X, Y, zeros(shape[1])]).T
    XYZ = reshape(XYZ, shape)
    
    return XYZ