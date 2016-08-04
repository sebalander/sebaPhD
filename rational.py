# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:52:00 2016

implements rational camera inverse mapping

@author: sebalander
"""
# %%
import numpy as np
from cv2 import Rodrigues

# %%
def inverseRational(corners, rotMatrix, tVec, cameraMatrix, distCoeffs):
    '''
    inverseRational(objPoints, rVec/rotMatrix, tVec, cameraMatrix, distCoeffs)
                                                                  -> objPoints
    takes corners in image and returns corrdinates in scene
    corners must be of size (n,1,2)
    objPoints has size (1,n,3)
    '''
    
    if rotMatrix.shape != (3,3):
        rotMatrix, _ = Rodrigues(rotMatrix)
    
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

# %%
def directRational(XYZ, rotMatrix, tVec, cameraMatrix, distCoeffs):
    '''
    directRational(objPoints, rVec/rotMatrix, tVec, cameraMatrix, distCoeffs)
                                                                    -> corners
    takes points in 3D scene coordinates and maps into distorted image
    objPoints must have size (1,n,3)
    corners is of size (n,1,2)
    '''
    
    if rotMatrix.shape != (3,3):
        rotMatrix, _ = Rodrigues(rotMatrix)
    # chequear esta linea a ver si se esta haciendo bien
    xyz = [np.linalg.linalg.dot(rotMatrix,p)+tVec[:,0] for p in XYZ[0]]
    xyz = np.array(xyz)
    
    xp = xyz[:,0]/xyz[:,2]
    yp = xyz[:,1]/xyz[:,2]
    
    rp2 = xp**2 + yp**2
    rp4 = rp2**2
    rp6 = rp2*rp4
    # polynomial coeffs
    # # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    q = (1 + distCoeffs[0,0]*rp2 + distCoeffs[1,0]*rp4 + distCoeffs[4,0]*rp6) / \
        (1 + distCoeffs[5,0]*rp2 + distCoeffs[6,0]*rp4 + distCoeffs[7,0]*rp6)
    
    # for plotting the distortion
    # rp = np.sqrt(rp2)
    # plt.scatter(rp,rp*q)
    # plt.plot([0,0.6],[0,0.6])
    
    xpp = xp * q
    ypp = yp * q
    
    u = cameraMatrix[0,0] * xpp + cameraMatrix[0,2]
    v = cameraMatrix[1,1] * ypp + cameraMatrix[1,2]
    
    distortedProyection = np.array([u,v]).reshape([np.shape(u)[0],1,2])
    return distortedProyection

# %% test distortion
#r = np.linspace(0,10,100)
#
#def distortRadius(r, k):
#    '''
#    returns distorted radius
#    '''
#    r2 = r**2
#    r4 = r2**2
#    r6 = r2*r4
#    # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
#    rd = r * (1 + k[0,0]*r2 + k[1,0]*r4 + k[4,0]*r6) / \
#        (1 + k[5,0]*r2 + k[6,0]*r4 + k[7,0]*r6)
#    return rd
#
#rd = distortRadius(r, distCoeffs)
#
#plt.plot(r,rd)