# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:01:57 2016

@author: sebalander
"""
from numpy import zeros, sqrt, roots, array, isreal, shape, prod, abs, sum
from numpy import reshape, dot, real, any, empty_like, arange
from cv2 import projectPoints, Rodrigues
from lmfit import minimize, Parameters
from calibration import calibrator
xypToZplane = calibrator.xypToZplane

# %% ========== ========== RATIONAL PARAMETER HANDLING ========== ==========
def formatParameters(rVec, tVec, linearCoeffs, distCoeffs):
    '''
    puts all intrinsic and extrinsic parameters in Parameters() format
    if there are several extrinsica paramters they are correctly 
    
    Inputs
    rVec, tVec : arrays of shape (n,3,1) or (3,1)
    distCoeffs must have be reshapable to (5)
    '''
    
    params = Parameters()
    
    if prod(rVec.shape) == 9:
        rVec = Rodrigues(rVec)[0]
    
    rVec = rVec.reshape(3)
    
    for i in range(3):
        params.add('rvec%d'%i,
                   value=rVec[i], vary=False)
        params.add('tvec%d'%i,
                   value=tVec[i], vary=False)
    
    if len(rVec.shape)==3:
        for j in range(len(rVec)):
            for i in range(3):
                params.add('rvec%d%d'%(j,i),
                           value=rVec[j,i,0], vary=False)
                params.add('tvec%d%d'%(j,i),
                           value=tVec[j,i,0], vary=False)
    
    params.add('fX',
               value=linearCoeffs[0,0], vary=False)
    params.add('fY',
               value=linearCoeffs[1,1], vary=False)
    params.add('cX',
               value=linearCoeffs[0,2], vary=False)
    params.add('cY',
               value=linearCoeffs[1,2], vary=False)
    
    # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    distCoeffs = distCoeffs.reshape(5)
    for i in range(5):
        params.add('distCoeffs%d'%i,
                   value=distCoeffs[i], vary=False)
    
    return params

def setParametersVary(params, flag):
    '''
    sets vary=True for the indicated paramters
    1 : extrinsic traslation
    2 : extrinsic rotation
    4 : intrinsic focal distance
    8 : intrinsic image center
    16 : intrinsic distortion numerator
    
    example:
    flag=35 means that extrinsic traslation, extrinsic rotation and instrinsic
    distortion denominator are set vary=True for optimisation. all others are
    set to vary=False.
    
    parece que no se estan usando los extrinsecos
    '''
    # 0 es False, coc es True
    exTr = flag&1
    exRo = flag&2
    inFo = flag&4
    inCe = flag&8
    inNu = flag&16
    # exTr, exRo, inFo, inCe, inNu
    
    for k in params.iterkeys():
        {
        }.get(k[0])
    
    params['fX'].vary=inFo
    params['fY'].vary=inFo
    params['cX'].vary=inCe
    params['cY'].vary=inCe
    
    # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    for i in [0,1,4]:
        params['distCoeffs%d'%i].vary=inNu



def retrieveParameters(params):
    rvec = zeros((3,1))
    tvec = zeros((3,1))
    for i in range(3):
        rvec[i,0] = params['rvec%d'%i].value
        tvec[i,0] = params['tvec%d'%i].value
    
    cameraMatrix = zeros((3,3))  
    cameraMatrix[0,0] = params['fX'].value
    cameraMatrix[1,1] = params['fY'].value
    cameraMatrix[0,2] = params['cX'].value
    cameraMatrix[1,2] = params['cY'].value
    cameraMatrix[2,2] = 1
    
    # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    distCoeffs = zeros(5)
    for i in range(5):
        distCoeffs[i] = params['distCoeffs%d'%i].value
    
    return rvec, tvec, cameraMatrix, distCoeffs


# %% ========== ========== DIRECT RATIONAL ========== ==========
def radialDistort(rp, k, quot=False):
    '''
    returns distorted radius using distortion coefficients k
    optionally it returns the distortion quotioent rpp = rp * q
    '''
    rp2 = rp**2
    rp4 = rp2**2
    rp6 = rp2*rp4
    
    k.shape = -1
    q = 1 + k[0]*rp2 + k[1]*rp4 + k[4]*rp6
    
    if quot:
        return q
    
    return rp * q

#
#def direct(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs):
#    projected, _ = projectPoints(fiducialPoints,
#                                 rVec,
#                                 tVec,
#                                 linearCoeffs,
#                                 distCoeffs)
#    
#    return projected
#
#def residualDirect(params, fiducialPoints, imageCorners):
#    '''
#    '''
#    rVec, tVec, linearCoeffs, distCoeffs = retrieveParameters(params)
#    
#    projectedCorners = direct(fiducialPoints,
#                              rVec,
#                              tVec,
#                              linearCoeffs,
#                              distCoeffs)
#    
#    return imageCorners[:,0,:] - projectedCorners[:,0,:]
#
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


# %% ========== ========== INVERSE RATIONAL ========== ==========
def radialUndistort(rpp, k, quot=False):
    '''
    takes distorted radius and returns the radius undistorted
    optioally it returns the undistortion quotient rp = rpp * q
    '''
    # polynomial coeffs, grade 7
    # # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    # polynomial coeffs, grade 7
    # # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    k.shape = -1
    poly = [[k[4], # k3
             0,
             k[1], # k2
             0,
             k[0], # k1
             0,
             1,
             -r] for r in rpp]
    
    # calculate roots
    rootsPoly = array([roots(p) for p in poly])
    
    # return flag, True if there is a suitable (real AND positive) solution
    rPRB = isreal(rootsPoly) & (0 <= real(rootsPoly))  # real Positive Real Bool
    retVal = any(rPRB, axis=1)
    
    rp = empty_like(rpp)
    
    if any(-retVal): # if at least one case of non solution
        # calculate extrema of polyniomial
        rExtrema = roots([7*k[4], 0, 5*k[1], 0, 3*k[0], 0, 1])
        # select real extrema in positive side, keep smallest
        rRealPos = min(rExtrema.real[isreal(rExtrema) & (0<=rExtrema.real)])
        # assign to problematic values
        rp[-retVal] = rRealPos
    
    # choose minimum positive roots
    rp[retVal] = [min(rootsPoly[i, rPRB[i]].real)
                   for i in arange(rpp.shape[0])[retVal]]
    
    if quot:
        return rp / rpp, retVal
    
    return rp, retVal

#def inverse(imageCorners, rVec, tVec, linearCoeffs, distCoeffs):
#    
#    imageCorners = imageCorners.reshape((-1,2))
#    distCoeffs = distCoeffs.reshape((5))
#    
#    xpp = ((imageCorners[:,0]-linearCoeffs[0,2]) /
#            linearCoeffs[0,0])
#    ypp = ((imageCorners[:,1]-linearCoeffs[1,2]) /
#            linearCoeffs[1,1])
#    rpp = sqrt(xpp**2 + ypp**2)
#    
#    # polynomial coeffs, grade 7
#    # # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
#    poly = [[distCoeffs[4], # k3
#             0,
#             distCoeffs[1], # k2
#             0,
#             distCoeffs[0], # k1
#             0,
#             1,
#             -r] for r in rpp]
#    
#    # choose the maximum of the real roots, hopefully it will be positive
#    rootsPoly = [roots(p) for p in poly]
#    rp_rpp = array([max(roo[isreal(roo)].real) for roo in rootsPoly]) / rpp
#    
#    xp = xpp * rp_rpp
#    yp = ypp * rp_rpp
#    
#    # project to z=0 plane. perhaps calculate faster with homography function?
#    XYZ = xypToZplane(xp, yp, rVec, tVec)
#    
#    return XYZ
#
#def residualInverse(params, fiducialPoints, imageCorners):
#    '''
#    '''
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


# %% programmed own functions to compare with opencv
def inverseRationalOwn(corners, rotMatrix, tVec, cameraMatrix, distCoeffs):
    '''
    inverseRational(objPoints, rVec/rotMatrix, tVec, cameraMatrix, distCoeffs)
                                                                  -> objPoints
    takes corners in image and returns corrdinates in scene
    function programed to compare with opencv
    corners must be of size (n,1,2)
    objPoints has size (1,n,3)
    '''
    
    if rotMatrix.shape != (3,3):
        rotMatrix, _ = Rodrigues(rotMatrix)
    
    xpp = (corners[:,0,0]-cameraMatrix[0,2]) / cameraMatrix[0,0]
    ypp = (corners[:,0,1]-cameraMatrix[1,2]) / cameraMatrix[1,1]
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
    
    # roots = np.roots(p)
    roots = [roots(p) for p in poly]
#    # max radious possible
#    rppMax = np.sqrt((cameraMatrix[0,2] / cameraMatrix[0,0])**2 +
#                     (cameraMatrix[1,2] / cameraMatrix[1,1])**2)
    
    # rp_rpp = roots[np.isreal(roots)].real[0] / rpp # asume real positive root, in interval
    rp_rpp = array([roo[isreal(roo)].real[0] for roo in roots]) / rpp
    
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
    XYZ = array([X, Y, zeros(shape[1])]).T
    XYZ = reshape(XYZ, shape)
    
    return XYZ

# %%
def directRationalOwn(objectPoints, rotMatrix, tVec, cameraMatrix, distCoeffs,
                   plot=False, img=None, corners=None):
    '''
    directRational(objPoints, rVec/rotMatrix, tVec, cameraMatrix, distCoeffs,
                   plot=False)
                                                                    -> corners
    takes points in 3D scene coordinates and maps into distorted image
    function programed to compare with opencv
    objPoints must have size (1,n,3)
    corners is of size (n,1,2)
    if plot is enabled, it will plot several intermediate results
    '''
    
    if rotMatrix.shape != (3,3):
        rotMatrix, _ = Rodrigues(rotMatrix)
    # chequear esta linea a ver si se esta haciendo bien
    xyz = [dot(rotMatrix,p)+tVec[:,0] for p in objectPoints[0]]
    xyz = array(xyz).T
    
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
    
    distortedProyection = uv.reshape([shape(uv)[0],1,2])
    
    return distortedProyection


# %% INTRINSIC CALIBRATION CHESSBOARD METHOD
from cv2 import calibrateCamera as cali

def calibrateIntrinsic(objpoints, imgpoints, imgSize, K, D,
                       flags, criteria):
    '''
    calibrates using opencvs implemented algorithm using chessboard method.
    if K, D, flags and criteria are not given it uses the default defined by
    me:
        
        D = zeros((1, 5))

    return rms, K, D, rVecs, tVecs
    '''
    if D is None:
        D = zeros((1, 5))
    
    rms, K, D, rVecs, tVecs = cali(objpoints, imgpoints,
                                      imgSize, K, D,
                                      flags=flags, criteria=criteria)
    
    return rms, K, D, rVecs, tVecs






