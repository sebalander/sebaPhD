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
    
    X = -(c*e - f*b)/q # check why wrong sign, why must put '-' in front?
    Y = -(f*a - c*d)/q
    
    shape = (1,corners.shape[0],3)
    XYZ = np.array([X, Y, np.zeros(shape[1])]).T
    XYZ = np.reshape(XYZ, shape)
    
    return XYZ

# %%
def directRational(objectPoints, rotMatrix, tVec, cameraMatrix, distCoeffs,
                   plot=False, img=None, corners=None):
    '''
    directRational(objPoints, rVec/rotMatrix, tVec, cameraMatrix, distCoeffs,
                   plot=False)
                                                                    -> corners
    takes points in 3D scene coordinates and maps into distorted image
    objPoints must have size (1,n,3)
    corners is of size (n,1,2)
    if plot is enabled, it will plot several intermediate results
    '''
    
    if rotMatrix.shape != (3,3):
        rotMatrix, _ = Rodrigues(rotMatrix)
    # chequear esta linea a ver si se esta haciendo bien
    xyz = [np.linalg.linalg.dot(rotMatrix,p)+tVec[:,0] for p in objectPoints[0]]
    xyz = np.array(xyz).T
    
    # plot rototranslation
    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        [x,y,z] = rotMatrix
        fig1 = plt.figure()
        ax1 = fig1.gca(projection='3d')
        ax1.plot([0, tVec[0,0]],
                [0, tVec[1,0]],
                [0, tVec[2,0]],'k')
        ax1.plot([tVec[0,0], tVec[0,0] + x[0]],
                [tVec[1,0], tVec[1,0] + x[1]],
                [tVec[2,0], tVec[2,0] + x[2]],'g')
        ax1.plot([tVec[0,0], tVec[0,0] + y[0]],
                [tVec[1,0], tVec[1,0] + y[1]],
                [tVec[2,0], tVec[2,0] + y[2]],'r')
        ax1.plot([tVec[0,0], tVec[0,0] + z[0]],
                [tVec[1,0], tVec[1,0] + z[1]],
                [tVec[2,0], tVec[2,0] + z[2]],'b')
        ax1.scatter(objectPoints[0,:,0],
                   objectPoints[0,:,1],
                   objectPoints[0,:,2])
        ax1.plot([0,1],[0,0],[0,0],'g')
        ax1.plot([0,0],[0,1],[0,0],'r')
        ax1.plot([0,0],[0,0],[0,1],'b')
        ax1.scatter(xyz[0],xyz[1],xyz[2])
    
    # to homogenous coords
    xpyp = xyz[:2]/xyz[2]
    
    if plot:
        ax1.scatter(xpyp[0],xpyp[1],1)
        fig2 = plt.figure()
        ax2 = fig2.gca()
        ax2.scatter(xpyp[0],xpyp[1]) # plot in z=1 plane, homogenous coordinates
    
    
    rp2 = np.sum(xpyp[:2]**2,0)
    rp4 = rp2**2
    rp6 = rp2*rp4
    # polynomial coeffs
    # # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    q = (1 + distCoeffs[0,0]*rp2 + distCoeffs[1,0]*rp4 + distCoeffs[4,0]*rp6) / \
        (1 + distCoeffs[5,0]*rp2 + distCoeffs[6,0]*rp4 + distCoeffs[7,0]*rp6)
    
    if plot:
        # for plotting the distortion
        rp = np.sqrt(rp2)
        fig3 = plt.figure()
        ax3 = fig3.gca()
        ax3.scatter(rp,rp*q)
        ax3.plot([0,0.6],[0,0.6])
    
    # distort
    xppypp = xpyp * q
    
    if plot:
        ax2.scatter(xppypp[0],xppypp[1])
    
    # linearly project to image
    uv = [cameraMatrix[0,0],cameraMatrix[1,1]]*xppypp.T + cameraMatrix[:2,2]
    
    if plot:
        # plot distorted positions
        fig4 = plt.figure(4)
        ax4 = fig4.gca()
        ax4.imshow(img)
        ax4.scatter(corners[:,0,0],corners[:,0,1])
        ax4.scatter(uv.T[0],uv.T[1])
    
    distortedProyection = uv.reshape([np.shape(uv)[0],1,2])
    
    return distortedProyection

