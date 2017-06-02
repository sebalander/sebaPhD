# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:47:18 2016


@author: sebalander
"""
# %% IMPORTS
from matplotlib.pyplot import plot, imshow, legend, show, figure, gcf, imread
from matplotlib.pyplot import xlabel, ylabel, arrow
from cv2 import Rodrigues, findHomography
from numpy import min, max, ndarray, zeros, array, reshape, sqrt, roots
from numpy import sin, cos, cross, ones, concatenate, flipud, dot, isreal
from numpy import linspace, polyval, eye, linalg, mean, prod, vstack
from numpy import empty_like, ones_like, zeros_like, pi
from scipy.linalg import sqrtm, norm, inv
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from copy import deepcopy as dc
from scipy.linalg import eig
from importlib import reload

# %% Z=0 PROJECTION


def euler(al,be,ga):
    '''
    devuelve matriz de rotacion según angulos de euler.
    Craigh, pag 42
    '''
    ca = cos(al)
    sa = sin(al)
    cb = cos(be)
    sb = sin(be)
    cg = cos(ga)
    sg = sin(ga)
    
    rot = array([[ca*cb, ca*sb*sg-sa*cg, ca*sb*cg+sa*sg],
                 [sa*cb, sa*sb*sg+ca*cg, sa*sb*cg+ca*sa],
                 [ -sb ,      cb*sg    ,      cb*cg    ]])
    
    return rot

## %% HOMOGRAPHY
#def pose2homogr(rVec,tVec):
#    '''
#    generates the homography from the pose descriptors
#    '''
#    
#    # calcular las puntas de los versores
#    if rVec.shape == (3,3):
#        [x,y,z] = rVec
#    else:
#        [x,y,z] = Rodrigues(rVec)[0]
#    
#    
#    H = array([x,y,tVec[:,0]]).T
#    
#    return H
#
#
#
#def homogr2pose(H):
#    '''
#    returns pose from the homography matrix
#    '''
#    r1B = H[:,0]
#    r2B = H[:,1]
#    r3B = cross(r1B,r2B)
#    
#    # convert to versors of B described in A ref frame
#    r1 = array([r1B[0],r2B[0],r3B[0]])
#    r2 = array([r1B[1],r2B[1],r3B[1]])
#    r3 = array([r1B[2],r2B[2],r3B[2]])
#    
#    rot = array([r1, r2, r3]).T
#    # make sure is orthonormal
#    rotNorm = rot.dot(inv(sqrtm(rot.T.dot(rot))))
#    rVec = Rodrigues(rotNorm)[0]  # make into rot vector
#    
#    # rotate to get displ redcribed in A
#    # tVec = np.dot(rot, -homography[2]).reshape((3,1))
#    tVec = H[:,2].reshape((3,1))
#    
#    # rescale
#    k = sqrt(norm(r1)*norm(r2))  # geometric mean
#    tVec = tVec / k
#    
#    return [rVec, tVec]
#
#
#
#
#def estimateInitialPose(objectPoints, corners, cameraMatrix):
#    '''
#    estimateInitialPose(objectPoints, corners, f, imgSize) -> [rVecs, tVecs, Hs]
#    
#    Recieves fiducial points and list of corners (one for each image), proposed
#    focal distance 'f' and returns the estimated pose of the camera. asumes
#    pinhole model.
#<<<<<<< HEAD:poseCalibration.py
#=======
#    
#    this function doesn't work very well, use at your own risk
#>>>>>>> 6d680e7d79797264e8842d99b669d04af01ee6e0:calibration/poseCalibration.py
#    '''
#    
#    src = objectPoints[0]+[0,0,1]
#    unos = ones((src.shape[0],1))
#    
#    rVecs = list()
#    tVecs = list()
#    Hs = list()
#    
#    if len(corners.shape) < 4:
#        corners = array([corners])
#    
#    for cor in corners:
#        # flip 'y' coordinates, so it works
#        # why flip image:
#        # http://stackoverflow.com/questions/14589642/python-matplotlib-inverted-image
#        # dst = [0, imgSize[0]] + cor[:,0,:2]*[1,-1]
#        x,y = cor.transpose((2,0,1))
#        # le saque el -1 y el corrimiento y parece que da mejor??
#        
#        # take to homogenous plane asuming intrinsic pinhole
#        dst = concatenate(((x-cameraMatrix[0,2])/cameraMatrix[0,0],
#                           (y-cameraMatrix[1,2])/cameraMatrix[1,1],
#                           unos), axis=1)
#        # fit homography
#        H = findHomography(src[:,:2], dst[:,:2], method=0)[0]
#        Hs.append(H)
#        
#        rVec, tVec = homogr2pose(H)
#        rVecs.append(rVec)
#        tVecs.append(tVec)
#    
#    return [array(rVecs), array(tVecs), array(Hs)]

# %% BASIC ROTOTRASLATION
def rotateRodrigues(x, r):
    '''
    rotates given vector x or list x of vectors as per rodrigues vector r
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    '''
    r.shape = 3
    th = norm(r)
    rn = r / th
    ct = cos(th)
    st = sin(th)
    
    
    try:
        # if x is just one point
        x.shape = 3
        return x * ct + cross(rn, x) * st + rn * dot(x, rn) * (1 - ct)
    except:
        # if x has many points
        x.shape = (-1,3)
        aux1 = x * ct + cross(rn, x) * st
        aux2 = rn.reshape((-1,1)) * dot(x, rn) * (1 - ct)
        return aux1 + aux2.T

def rotoTrasRodri(x, r, t):
    '''
    rototraslates all x points using r and t
    '''
    t.shape = 3
    return rotateRodrigues(x, r) + t


def rotoTrasRodriInverse(x, r, t):
    '''
    rototraslates all x points using r and t inversely
    '''
    t.shape = 3
    return rotateRodrigues(x - t, -r)

def rotoTrasHomog(x, r, t):
    '''
    rototraslates points x and projects to homogenous coordinates
    '''
    x2 = rotoTrasRodri(x, r, t)
    
    return x2[:,:2] / x2[:,2].reshape((-1,1))

# %%
from calibration import StereographicCalibration as stereographic
from calibration import UnifiedCalibration as unified
from calibration import RationalCalibration as rational
from calibration import FisheyeCalibration as fisheye
from calibration import PolyCalibration as poly

reload(stereographic)
reload(unified)
reload(rational)
reload(fisheye)
reload(poly)

# %% PARAMETER HANDLING
def formatParameters(rVec, tVec, cameraMatrix, distCoeffs, model):
    
    switcher = {
    'stereographic' : stereographic.formatParameters,
    'unified' : unified.formatParameters,
    'rational' : rational.formatParameters,
    'poly' : poly.formatParameters,
    'fisheye' : fisheye.formatParameters
    }
    
    return switcher[model](rVec, tVec, cameraMatrix, distCoeffs)

def retrieveParameters(params, model):
    
    switcher = {
    'stereographic' : stereographic.retrieveParameters,
    'unified' : unified.retrieveParameters,
    'rational' : rational.retrieveParameters,
    'poly' : poly.retrieveParameters,
    'fisheye' : fisheye.retrieveParameters
    }
    
    return switcher[model](params)

# %% DIRECT PROJECTION
def hom2ccd(xpp, ypp, cameraMatrix):
    xccd = cameraMatrix[0,0] * xpp + cameraMatrix[0,2]
    yccd = cameraMatrix[1,1] * ypp + cameraMatrix[1,2]
    
    return vstack((xccd, yccd)).T


# switcher for radial distortion
distort = {
'stereographic' : stereographic.radialDistort,
'unified' : unified.radialDistort,
'rational' : rational.radialDistort,
'poly' : poly.radialDistort,
'fisheye' : fisheye.radialDistort
}


def direct(objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model, ocv=False):
    '''
    performs projection form 3D world into image, is the "direct" distortion
    optionally it uses opencv's function if available
    '''
    
    xHomog = rotoTrasHomog(objectPoints, rVec, tVec)
    
    rp = norm(xHomog, axis=1)
    
    q = distort[model](rp, distCoeffs, quot=True)
    # print(xHomog.shape, q.shape)
    xpp, ypp = xHomog.T * q.reshape(1, -1)
    
    # project to ccd
    return hom2ccd(xpp, ypp, cameraMatrix)

def residualDirect(params, objectPoints, imagePoints, model):
    
    switcher = {
    'stereographic' : stereographic.residualDirect,
    'unified' : unified.residualDirect,
    'rational' : rational.residualDirect,
    'poly' : poly.residualDirect,
    'fisheye' : fisheye.residualDirect
    }
    
    return switcher[model](params, objectPoints, imagePoints)


def calibrateDirect(objectPoints, imagePoints, rVec, tVec, cameraMatrix, distCoeffs, model):
    
    switcher = {
    'stereographic' : stereographic.calibrateDirect,
    'unified' : unified.calibrateDirect,
    'rational' : rational.calibrateDirect,
    'poly' : poly.calibrateDirect,
    'fisheye' : fisheye.calibrateDirect
    }
    
    return switcher[model](objectPoints, imagePoints, rVec, tVec, cameraMatrix, distCoeffs)


# %% INVERSE PROJECTION
def ccd2hom(imagePoints, cameraMatrix, Cccd=None, Cf=None):
    '''
    must provide covariances for every point if cov is not None
    '''
    # undo CCD projection, asume diagonal ccd rescale
    xpp = (imagePoints[:,0] - cameraMatrix[0,2]) / cameraMatrix[0,0]
    ypp = (imagePoints[:,1] - cameraMatrix[1,2]) / cameraMatrix[1,1]
    
    if Cccd is None:
        return xpp, ypp
    else:
        Cpp = dc(Cccd)
        Cpp[:,0,0] /= cameraMatrix[0, 0]**2
        Cpp[:,1,1] /= cameraMatrix[1, 1]**2
        fxfy = cameraMatrix[0, 0] * cameraMatrix[1, 1]
        Cpp[:,0,1] /= fxfy
        Cpp[:,1,0] /= fxfy
        
        if Cf is None:
            return xpp, ypp, Cpp
        else:
            unos = ones_like(imagePoints[:,0])
            ceros = zeros_like(unos)
            a = - unos / cameraMatrix[0, 0]
            b = (cameraMatrix[0, 2] - imagePoints[:,0]) * a**2
            c = - unos / cameraMatrix[1, 1]
            d = (cameraMatrix[1, 2] - imagePoints[:,1]) * c**2
            
            J = array([[a, ceros, b, ceros],[ceros, c, ceros, d]])
            J = J.transpose((2,0,1))
            
            for i in range(unos.shape[0]):
                Cpp[i] += J[i].dot(Cf).dot(J[i].T)
            
            return xpp, ypp, Cpp

# switcher for radial un-distortion
undistort = {
'stereographic' : stereographic.radialUndistort,
'unified' : unified.radialUndistort,
'rational' : rational.radialUndistort,
'poly' : poly.radialUndistort,
'fisheye' : fisheye.radialUndistort
}


def homDist2homUndist(xpp, ypp, distCoeffs, model, Cpp=None, Ck=None):
    '''
    takes ccd cordinates and projects to homogenpus coords and undistorts
    '''
    if Cpp is None:
        rpp = norm([xpp, ypp], axis=0)
        
        # calculate ratio of undistortion
        q, _ = undistort[model](rpp, distCoeffs, quot=True)
        
        xp = q * xpp # undistort in homogenous coords
        yp = q * ypp
        
        return xp, yp
    else:
        rpp = norm([xpp, ypp], axis=0)
        
        # calculate ratio of undistorition and it's derivative wrt radius
        q, _, dqI, dqDk = undistort[model](rpp, distCoeffs, quot=True, der=True)
        
        xp = q * xpp # undistort in homogenous coords
        yp = q * ypp
        
        xpp2 = xpp**2
        ypp2 = ypp**2
        xypp = xpp * ypp
        dqIrpp = dqI / rpp
        
        # calculo jacobiano
        J = array([[xpp2, xypp],[xypp, ypp2]])
        J *= dqIrpp.reshape(1,1,-1)
        J[0,0,:] += q
        J[1,1,:] += q
        
        Jk = array([xpp, ypp]).T.reshape(-1,2,1) * dqDk.T.reshape(-1,1,Ck.shape[0])
        
        Cp = empty_like(Cpp)
        
        for i in range(len(xp)):
            Caux = Cpp[i] + Jk[i].dot(Ck).dot(Jk[i].T)
            Cp[i] = J[:,:,i].dot(Caux).dot(J[:,:,i].T)
        
        return xp, yp, Cp


def rototrasCovariance(xp, yp, rV, tV, Cp):
    '''
    propaga la elipse proyectada por el conoide soble el plano del mapa
    solo toma en cuenta la incerteza en xp, yp
    '''
    a, b, _, c = Cp.flatten()
    mx, my = (xp, yp)
    
    r11, r12, r21, r22, r31,r32 = Rodrigues(rV)[0][:,:2].flatten()
    tx, ty, tz = tV.flatten()
    
    # auxiliar alculations
    br11 = b*r11
    bmx = b*mx
    amx = a*mx
    cmy = c*my
    x0, x1, x2, x3, x4, x8, x9, x10, x11, x12, x13, x14 = 2*array([
            br11, amx*r11, bmx*r21, br11*my, cmy*r21, bmx*my, b*r12, amx*r12,
            bmx*r22, b*my*r12, cmy*r22, r31*r32])
    x5, mx2, my2, x15 = array(r31, mx, my, r32)**2
    x6 = a*mx2
    x7 = c*my2
    
    # matrix elements
    C11 = a*r11**2 + c*r21**2 + r21*x0 - r31*x1 - r31*x2 - r31*x3 - r31*x4
    C11 += x5*x6 + x5*x7 + x5*x8
    
    C12 = r12*(2*a*r11) + r21*x9 + r22*x0 + r22*(2*c*r21) - r31*x10 - r31*x11
    C12 += - r31*x12 - r31*x13 - r32*x1 - r32*x2 - r32*x3 - r32*x4 + x14*x6
    C12 += x14*x7 + (4*r31*r32)*(bmx*my)
    
    C22 = a*r12**2 + c*r22**2 + r22*x9 - r32*x10 - r32*x11 - r32*x12 - r32*x13
    C22 += x15*x6 + x15*x7 + x15*x8
    
    # compose covaciance matrix
    Cm = array([[C11, C12], [C12, C22]])
    
    return Cm



def jacobianosHom2Map(xp, yp, rV, tV):
    '''
    returns the jacobians needed to calculate the propagation of uncertainty
    
    '''
    rx, ry, rz = rV
    tx, ty, tz = tV
    x0 = ty - tz*yp
    x1 = rx**2
    x2 = ry**2
    x3 = rz**2
    x4 = x1 + x2 + x3
    x5 = sqrt(x4)
    x6 = sin(x5)
    x7 = x6/x5
    x8 = rx*x7
    x9 = -x8
    x10 = ry*rz
    x11 = 1/x4
    x12 = cos(x5)
    x13 = -x12
    x14 = x13 + 1
    x15 = x11*x14
    x16 = x10*x15
    x17 = -x16 + x9
    x18 = x15*x2
    x19 = x16 + x8
    x20 = x19*yp
    x21 = x12 + x18 - x20
    x22 = x1*x15
    x23 = ry*x7
    x24 = -x23
    x25 = x15*x3
    x26 = x24 + x25
    x27 = x26*xp
    x28 = x12 + x22 - x27
    x29 = rz*x7
    x30 = -x29
    x31 = rx*ry
    x32 = x15*x31
    x33 = -x19*xp + x30 + x32
    x34 = -x26*yp + x29 + x32
    x35 = x21*x28 - x33*x34
    x36 = 1/x35
    x37 = x23 - x25
    x38 = x17*x34 - x21*x37
    x39 = tx - tz*xp
    x40 = x35**(-2)
    x41 = x40*(x0*x33 - x21*x39)
    x42 = -x17*x28 + x33*x37
    x43 = x40*(-x0*x28 + x34*x39)
    x44 = x6/x4**(3/2)
    x45 = x2*x44
    x46 = rx*x45
    x47 = x4**(-2)
    x48 = 2*rx*x14*x47
    x49 = -x2*x48
    x50 = x11*x12
    x51 = x1*x44
    x52 = x31*x44
    x53 = rz*x52
    x54 = 2*rz*x14*x47
    x55 = -x31*x54
    x56 = x53 + x55 + x7
    x57 = x1*x50 - x51 + x56
    x58 = x46 + x49 - x57*yp + x9
    x59 = 2*ry*x14*x47
    x60 = ry*x51 - x1*x59
    x61 = rx*rz
    x62 = x44*x61
    x63 = x50*x61
    x64 = ry*x15
    x65 = -x57*xp + x60 + x62 - x63 + x64
    x66 = rx**3
    x67 = rx*x15
    x68 = 2*x14*x47
    x69 = x31*x50
    x70 = x3*x44
    x71 = rx*x70 - x3*x48 + x52 - x69
    x72 = x44*x66 - x66*x68 + 2*x67 - x71*xp + x9
    x73 = -x62
    x74 = x60 + x63 + x64 - x71*yp + x73
    x75 = -x21*x72 - x28*x58 + x33*x74 + x34*x65
    x76 = ry**3
    x77 = rz*x45 - x2*x54
    x78 = rz*x15
    x79 = -x52 + x69 + x77 + x78
    x80 = x24 + x44*x76 + 2*x64 - x68*x76 - x79*yp
    x81 = x46 + x49 + x67
    x82 = x10*x44
    x83 = x10*x50
    x84 = -x83
    x85 = -x79*xp + x81 + x82 + x84
    x86 = ry*x70 - x3*x59
    x87 = -x7
    x88 = -x2*x50 + x45 + x86 + x87
    x89 = x24 + x60 - x88*xp
    x90 = x81 - x82 + x83 - x88*yp
    x91 = -x21*x89 - x28*x80 + x33*x90 + x34*x85
    x92 = x63 + x64 + x73 + x86
    x93 = x30 + x77 - x92*yp
    x94 = x3*x50
    x95 = x53 + x55 + x70 + x87 - x92*xp - x94
    x96 = rz**3
    x97 = x44*x96 - x68*x96 + 2*x78 + x82 + x84
    x98 = rz*x51 - x1*x54 + x30 - x97*xp
    x99 = x56 - x70 + x94 - x97*yp
    x100 = -x21*x98 - x28*x93 + x33*x99 + x34*x95
    
    # jacobiano de la pos en el mapa con respecto a las posiciones homogeneas
    JXm_Xp = array([[x36*(tz*x21 + x0*x17) + x38*x41,
                     x36*(-tz*x33 - x17*x39) + x41*x42],
                    [x36*(-tz*x34 - x0*x37) + x38*x43,
                     x36*(tz*x28 + x37*x39) + x42*x43]])
    
    # jacobiano respecto al vector de rodriguez
    JXm_rV = array([[x36*(x0*x65 - x39*x58) + x41*x75,
                     x36*(x0*x85 - x39*x80) + x41*x91,
                     x100*x41 + x36*(x0*x95 - x39*x93)],
                    [x36*(-x0*x72 + x39*x74) + x43*x75,
                     x36*(-x0*x89 + x39*x90) + x43*x91,
                     x100*x43 + x36*(-x0*x98 + x39*x99)]])
    
    # jacobiano respecto a la traslacion
    JXm_tV = array([[x36*(x13 - x18 + x20),
                     x33*x36,
                     x36*(x21*xp - x33*yp)],
                     [x34*x36,
                     x36*(x13 - x22 + x27),
                     x36*(x28*yp - x34*xp)]])

    return JXm_Xp, JXm_rV, JXm_tV

#

def xypToZplane(xp, yp, rV, tV, Cp=None, Crt=None):
    '''
    projects a point from homogenous undistorted to 3D asuming z=0
    '''
    if prod(rV.shape) == 3:
        R = Rodrigues(rV)[0]
    
    # auxiliar calculations
    a = R[0,0] - R[2,0] * xp
    b = R[0,1] - R[2,1] * xp
    c = tV[0] - tV[2] * xp
    d = R[1,0] - R[2,0] * yp
    e = R[1,1] - R[2,1] * yp
    f = tV[1] - tV[2] * yp
    q = a*e - d*b
    
    xm = (f*b - c*e) / q
    ym = (c*d - f*a) / q
    
    if Cp is None:
        return xm, ym
    else:
        if Crt is None:
            # calculo con el conoide
            Cm = rototrasCovariance(xp, yp, rV, tV, Cp)
            return xm, ym, Cm
        
        else:
            Cr, Ct = Crt
            # calculo jacobianos
            JXm_Xp, JXm_rV, JXm_tV = jacobianosHom2Map(xp, yp, rV, tV)
            # propago las matrices de incerteza
            Cm = empty_like(Cp)
            for i in range(len(xp)):
                Cm[i] = JXm_Xp[:,:,i].dot(Cp[i]).dot(JXm_Xp[:,:,i].T)
                Cm[i] += JXm_rV[:,:,i].dot(Cr).dot(JXm_rV[:,:,i].T)
                Cm[i] += JXm_tV[:,:,i].dot(Ct).dot(JXm_tV[:,:,i].T)
            
            return xm, ym, Cm



def inverse(imagePoints, rV, tV, cameraMatrix, distCoeffs, model,
            Cccd=None, Cf=None, Ck=None, Crt=None):
    '''
    inverseFisheye(objPoints, rVec/rotMatrix, tVec, cameraMatrix,
                    distCoeffs)-> objPoints
    takes corners in image and returns coordinates in scene
    corners must be of size (n,1,2)
    objPoints has size (1,n,3)
    ignores tangential and tilt distortions
    
    propagates covariance uncertainty
    '''
    
    if Cccd is None:
        # project to homogenous distorted
        xpp, ypp = ccd2hom(imagePoints, cameraMatrix)
        # undistort
        xp, yp = homDist2homUndist(xpp, ypp, distCoeffs, model)
        # project to plane z=0 from homogenous
        xm, ym = xypToZplane(xp, yp, rV, tV)
        return xm, ym
    else:
        # project to homogenous distorted
        xpp, ypp, Cpp = ccd2hom(imagePoints, cameraMatrix, Cccd, Cf)
        # undistort
        xp, yp, Cp = homDist2homUndist(xpp, ypp, distCoeffs, model, Cpp, Ck)
        # project to plane z=0 from homogenous
        xm, ym, Cm = xypToZplane(xp, yp, rV, tV, Cp, Crt)
        return xm, ym, Cm


def residualInverse(params, objectPoints, imagePoints, model):
    
    switcher = {
    'stereographic' : stereographic.residualInverse,
    'unified' : unified.residualInverse,
    'rational' : rational.residualInverse,
    'poly' : poly.residualInverse,
    'fisheye' : fisheye.residualInverse
    }
    
    return switcher[model](params, objectPoints, imagePoints)


def calibrateInverse(objectPoints, imagePoints, rVec, tVec, cameraMatrix, distCoeffs, model):
    
    switcher = {
    'stereographic' : stereographic.calibrateInverse,
    'unified' : unified.calibrateInverse,
    'rational' : rational.calibrateInverse,
    'poly' : poly.calibrateInverse,
    'fisheye' : fisheye.calibrateInverse
    }
    
    return switcher[model](objectPoints, imagePoints, rVec, tVec, cameraMatrix, distCoeffs)

# %% PLOTTING

# plot corners and their projection
def cornerComparison(img, corners1, corners2=None,
                     label1='Corners', label2='Proyectados'):
    '''
    draw on top of image two sets of corners, ideally calibration and direct
    projected
    '''
    # get in more usefull shape
    corners1 = corners1.reshape((-1,2))
    X1 = corners1[:,0]
    Y1 = corners1[:,1]
    
    figure()
    imshow(img)
    plot(X1, Y1, 'xr', markersize=10, label=label1)
    
    if corners2 is not None:
        corners2 = corners2.reshape((-1,2))
        X2 = corners2[:,0]
        Y2 = corners2[:,1]
        
        plot(X2, Y2, '+b', markersize=10, label=label2)
        
        for i in range(len(X1)):  # unir correpsondencias
            plot([X1[i],X2[i]],[Y1[i],Y2[i]],'k-')
    
    legend()
    show()

# compare fiducial points
def fiducialComparison(rVec, tVec, fiducial1, fiducial2=None,
                       label1='Calibration points', label2='Projected points'):
    '''
    Draw on aplane two sets of fiducial points for comparison, ideally
    calibration and direct projected
    '''
    
    
    fiducial1 = fiducial1.reshape(-1,3)
    X1, Y1, _ = fiducial1.T  #[:,0]
    # Y1 = fiducial1[:,1]
    
    figure()
    plot(X1, Y1, 'xr', markersize=10, label=label1)
    
#    
#    # set origin of MAP reference frame
#    xM0 = min(X1)
#    xM1 = max(X1)
#    yM0 = min(Y1)
#    yM1 = max(Y1)
#    
#    # get a characteristic scale of the graph
#    s = max([xM1 - xM0, yM1 - yM0]) / 10
#    
#    t = array(tVec).reshape(3)
#    
#    # calcular las puntas de los versores
#    if rVec.shape == (3,3):
#        [x,y,z] = s * rVec.T
#    else:
#        [x,y,z] = s * Rodrigues(rVec)[0].T
#
#    plot([t[0], t[0] + x[0]], [t[1], t[1] + x[1]], '-r', label='cam X')
#    plot([t[0], t[0] + y[0]], [t[1], t[1] + y[1]], '-b', label='cam Y')

    
    if fiducial2 is not None:
        fiducial2 = fiducial2.reshape(-1,3)
        X2, Y2, _ = fiducial2.T  # [:,0]
        # Y2 = fiducial2[:,1]
        plot(X2, Y2, '+b', markersize=10, label=label2)
        
        for i in range(len(X1)):  # unir correpsondencias
            plot([X1[i],X2[i]],[Y1[i],Y2[i]],'k-')
    
    legend()
    show()

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def fiducialComparison3D(rVec, tVec, fiducial1, fiducial2=None,
                         label1 = 'Fiducial points', label2 = 'Projected points'):
    '''
    draw in 3D the position of the camera and the fiducial points, also can
    draw an extras et of fiducial points (projected). indicates orienteation
    of camera
    
    '''
    
    fiducial1 = fiducial1.reshape(-1,3)
    
    if prod(rVec.shape) == 3:
        rVec = Rodrigues(rVec)[0]
    
    # convert to camera coords by roto traslation
    fiducialCam = dot(rVec, fiducial1.T)
    fiducialCam += tVec.reshape((3,1))
    
    # get a characteristic scale of the graph
    s = mean(linalg.norm(fiducialCam, axis=0)) / 3
    
    
    # set origin of MAP reference frame
#    xM0 = min(X1)
#    xM1 = max(X1)
#    yM0 = min(Y1)
#    yM1 = max(Y1)
#    zM0 = min(Z1)
#    zM1 = max(Z1)
#    
#    
#    t = array(tVec).reshape(3)
#    
#    # calcular las puntas de los versores
#    if rVec.shape is (3,3):
#        [x,y,z] = s * rVec
#    else:
#        [x,y,z] = s * Rodrigues(rVec)[0]
#    # get versors of map in cam coords
#    [x,y,z] = s * rVec.T
#
##print(array([x,y,z]))
#    [x,y,z] = [x,y,z] + t
#    #print(t)
    
    
    fig = figure()
    ax = fig.gca(projection='3d')
    
    ax.plot(fiducialCam[0], fiducialCam[1], fiducialCam[2],
            'xr', markersize=10, label=label1)
    
    
#    # get plot range
#    if fiducial2 is not None:
#        fiducial2 = fiducial2.reshape(-1,3)
#        X2, Y2, Z2 = fiducial2.T
#        
#        ax.plot(X2,Y2,Z2,'+b', markersize=10, label=label2)
#        
#        xmin = min([xM0, min(X2), t[0], x[0], y[0], z[0]])
#        ymin = min([yM0, min(Y2), t[1], x[1], y[1], z[1]])
#        zmin = min([zM0, min(Z2), t[2], x[2], y[2], z[2]])
#        xmax = max([xM1, max(X2), t[0], x[0], y[0], z[0]])
#        ymax = max([yM1, max(Y2), t[1], x[1], y[1], z[1]])
#        zmax = max([zM1, max(Z2), t[2], x[2], y[2], z[2]])
#    else:
#        xmin = min([xM0, t[0], x[0], y[0], z[0]])
#        ymin = min([yM0, t[1], x[1], y[1], z[1]])
#        zmin = min([zM0, t[2], x[2], y[2], z[2]])
#        xmax = max([xM1, t[0], x[0], y[0], z[0]])
#        ymax = max([yM1, t[1], x[1], y[1], z[1]])
#        zmax = max([zM1, t[2], x[2], y[2], z[2]])
    
#    ax.set_xbound(xmin, xmax)
#    ax.set_ybound(ymin, ymax)
#    ax.set_zbound(zmin, zmax)
#    
#    ejeX = Arrow3D([xM0, xM1],
#                   [yM0, yM0],
#                   [zM0, zM0],
#                   mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
#    ejeY = Arrow3D([xM0, xM0],
#                   [yM0, yM1],
#                   [zM0, zM0],
#                   mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
#    ejeZ = Arrow3D([xM0, xM0],
#                   [yM0, yM0],
#                   [zM0, zM1],
#                   mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
#    
#    origen = Arrow3D([xM0, t[0]],
#                     [yM0, t[1]],
#                     [zM0, t[2]],
#                     mutation_scale=20, lw=1, arrowstyle="-", color="k",
#                     linestyle="dashed")
    
    ax.plot([0, s], [0, 0], [0, 0], "-r")
    ax.plot([0, 0], [0, s], [0, 0], "-b")
    ax.plot([0, 0], [0, 0], [0, s], "-k")
#    ejeYc = Arrow3D([t[0], y[0]],
#                   [t[1], y[1]],
#                   [t[2], y[2]],
#                   mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
#    ejeZc = Arrow3D([t[0], z[0]],
#                   [t[1], z[1]],
#                   [t[2], z[2]],
#                   mutation_scale=20, lw=3, arrowstyle="-|>", color="b")
    
#    ax.add_artist(ejeX)
#    ax.add_artist(ejeY)
#    ax.add_artist(ejeZ)
#    ax.add_artist(origen)
#    ax.add_artist(ejeXc)
#    ax.add_artist(ejeYc)
#    ax.add_artist(ejeZc)
#    
    ax.legend()
    
    show()


def joinPoints(pts1, pts2):
    figure()
    plot(pts1[:,0], pts1[:,1],'+k');
    plot(pts2[:,0], pts2[:,1],'xr');
    
    # unir con puntos
    for i in range(pts1.shape[0]):
        plot([pts1[i,0], pts2[i,0]],
                 [pts1[i,1], pts2[i,1]],'-k');
    return gcf()


def plotHomographyToMatch(objectPoints, corners, f, imgSize, images=None):
    
    src = objectPoints[0]+[0,0,1]
    
    
    for i in range(len(corners)):
        # rectify corners and image
        dst = [0, imgSize[0]] + corners[i,:,0,:2]*[1,-1]
        
        
        figure()
        if images!=None:
            img = imread(images[i])
            imshow(flipud(img),origin='lower');
        aux = src[:,:2]*f+imgSize/2
        plot(dst[:,0], dst[:,1], '+r')
        plot(aux[:,0], aux[:,1], 'xk')
        for j in range(src.shape[0]):
            plot([dst[:,0], aux[:,0]],
                 [dst[:,1], aux[:,1]],'-k');



def plotForwardHomography(objectPoints, corners, f, imgSize, Hs, images=None):
    src = objectPoints[0]+[0,0,1]
    
    for i in range(len(corners)):
        # cambio de signo y corrimiento de 'y'
        dst = [0, imgSize[0]] + corners[i,:,0,:2]*[1,-1]
        
        # calculate forward, destination "Projected" points
        dstP = array([dot(Hs[i], sr) for sr in src])
        dstP = array([dstP[:,0]/dstP[:,2], dstP[:,1]/dstP[:,2]]).T
        dstP = f*dstP+ imgSize/2
        
        # plot data
        fig = joinPoints(dst, dstP)
        if images!=None:
            img = imread(images[i])
            ax = fig.gca()
            ax.imshow(flipud(img),origin='lower');


def plotBackwardHomography(objectPoints, corners, f, imgSize, Hs):
    src = objectPoints[0]+[0,0,1]
    unos = ones((src.shape[0],1))
    
    for i in range(len(corners)):
        Hi = inv(Hs[i])
        # cambio de signo y corrimiento de 'y'
        dst = [0, imgSize[0]] + corners[i,:,0,:2]*[1,-1]
        dst = (dst - imgSize/2) /f  # pinhole
        dst = concatenate((dst,unos),axis=1)
        
        # calculate backward, source "Projected" points
        srcP = array([dot(Hi, ds) for ds in dst])
        srcP = array([srcP[:,0]/srcP[:,2],
                         srcP[:,1]/srcP[:,2],
                         unos[:,0]]).T
        
        
        fiducialProjected = (srcP-[0,0,1]).reshape(objectPoints.shape)
        
        rVec, tVec = homogr2pose(Hs[i]);
        fiducialComparison3D(rVec, tVec,
                             objectPoints, fiducialProjected,
                             label1="fiducial points",
                             label2="%d ajuste"%i)

def plotRationalDist(distCoeffs,imgSize, cameraMatrix):
    
    k = distCoeffs[[0,1,4,5,6,7],0]
    pNum = [0, 1, 0, k[0], 0, k[1], 0, k[2]]
    pDen = [1, 0, k[3], 0, k[4], 0, k[5]]
    
    # buscar un buen rango de radio
    rDistMax = sqrt((cameraMatrix[0,2]/cameraMatrix[0,0])**2 +
                    (cameraMatrix[1,2]/cameraMatrix[1,1])**2)
    
    # polynomial coeffs, grade 7
    # # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    poly = [k[2], # k3
             -rDistMax*k[5], # k6
             k[1], # k2
             -rDistMax*k[4], # k5
             k[0], # k1
             -rDistMax*k[3], # k4
             1,
             -rDistMax]
    
    rootsPoly = roots(poly)
    realRoots = rootsPoly[isreal(rootsPoly)].real
    rMax = max(realRoots)
    
    r = linspace(0,rMax)
    rDist = polyval(pNum,r) / polyval(pDen,r)
    
    figure()
    plot(r, rDist)
    xlabel("radio")
    ylabel("radio distorsionado")
    

# %%
from scipy.special import chdtriv
fi = linspace(0,2*pi,20)
Xcirc = array([cos(fi), sin(fi)]) * chdtriv(0.1, 2)

def plotEllipse(ax, C, mux, muy, col):
    '''
    se grafica una elipse asociada a la covarianza c, centrada en mux, muy
    '''
    Ci = inv(C)
    
    l, v = eig(Ci)
    
    # matrix such that A.dot(A.T)==Ci
    A = sqrt(l.real) * v
    # roto eescaleo para llevae al cicuulo a la elipse
    xeli, yeli = dot(Xcirc.T, inv(A)).T
    
    ax.plot(xeli+mux, yeli+muy, c=col, lw=0.5)

#



# %% INRINSIC CALIBRATION
switcherIntrinsicFunc = {
    'poly' : poly.calibrateIntrinsic,
    'rational' : rational.calibrateIntrinsic,
    'fisheye' : fisheye.calibrateIntrinsic
    }

switcherIntrCalibFlags = {
    # CALIB_ZERO_TANGENT_DIST
    'poly' : 8,  #  (1 << 3)
    # decided to not fix ppal point, CALIB_FIX_PRINCIPAL_POINT + (1 << 2)
    # CALIB_FIX_ASPECT_RATIO, (1 << 1) +
    # add CALIB_RATIONAL_MODEL
    'rational' : 16386,  # (1 << 1) + (1 << 14)
    # CALIB_FIX_SKEW, CALIB_RECOMPUTE_EXTRINSIC
    'fisheye' : 10  # (1 << 3) + (1 << 1)
    # decided to not fix ppal point, CALIB_FIX_PRINCIPAL_POINT + (1 << 9)
    }


switcherIntrCalibD = {
    'poly' : zeros((1, 5)),  # (1 << 1) + (1 << 3)
    'rational' : zeros((1, 8)),  # (1 << 1) + (1 << 3) + (1 << 14)
    'fisheye' : zeros((1, 4))   # (1 << 3)
    }



def calibrateIntrinsic(objpoints, imgpoints, imgSize, model, K=None, D=None,
                       flags=None, criteria=None):
    '''
    only available for rational and fisheye, we use opencv's functions
    exclusively here
    
    parameters defined by me for default use:
        K = [[600.0, 0.0,   imgSize[1]/2],
             0.0,    600.0, imgSize[0]/2],
             0.0,    0.0,        1]]
        
        flags =  no skew and fixed ppal point, seems more apropiate
                          to our camera model
        criteria = (3, int(1e5), 1e-15)
    
    return rms, K, D, rVecs, tVecs
    '''
    
    if K is None:
        K = eye(3)
        K[0, 2] = imgSize[1]/2
        K[1, 2] = imgSize[0]/2
        K[0, 0] = K[1, 1] = 600.0
    
    if D is None:
        D = switcherIntrCalibD[model]
        
#     calibrateCamera flags =========================
#     enum  	{
#      cv::CALIB_USE_INTRINSIC_GUESS = 0x00001, 2**0
#      cv::CALIB_FIX_ASPECT_RATIO = 0x00002,    2**1
#      cv::CALIB_FIX_PRINCIPAL_POINT = 0x00004, 2**2
#      cv::CALIB_ZERO_TANGENT_DIST = 0x00008,   2**3
#      cv::CALIB_FIX_FOCAL_LENGTH = 0x00010,    2**4
#      cv::CALIB_FIX_K1 = 0x00020,              2**5
#      cv::CALIB_FIX_K2 = 0x00040,              2**6
#      cv::CALIB_FIX_K3 = 0x00080,              2**7
#      cv::CALIB_FIX_K4 = 0x00800,              2**11
#      cv::CALIB_FIX_K5 = 0x01000,              2**12
#      cv::CALIB_FIX_K6 = 0x02000,              2**13
#      cv::CALIB_RATIONAL_MODEL = 0x04000,      2**14
#      cv::CALIB_THIN_PRISM_MODEL = 0x08000,    2**15
#      cv::CALIB_FIX_S1_S2_S3_S4 = 0x10000,     2**16
#      cv::CALIB_TILTED_MODEL = 0x40000,        2**18
#      cv::CALIB_FIX_TAUX_TAUY = 0x80000,       2**19
#      cv::CALIB_USE_QR = 0x100000,             2**20
#      cv::CALIB_FIX_INTRINSIC = 0x00100,       2**8
#      cv::CALIB_SAME_FOCAL_LENGTH = 0x00200,   2**9
#      cv::CALIB_ZERO_DISPARITY = 0x00400,      2**10
#      cv::CALIB_USE_LU = (1 << 17)             2**17
#    }
#     fisheye calibrate flags =========================
#    enum{
#      cv::fisheye::CALIB_USE_INTRINSIC_GUESS = 1 << 0,
#      cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC = 1 << 1,
#      cv::fisheye::CALIB_CHECK_COND = 1 << 2,
#      cv::fisheye::CALIB_FIX_SKEW = 1 << 3,
#      cv::fisheye::CALIB_FIX_K1 = 1 << 4,
#      cv::fisheye::CALIB_FIX_K2 = 1 << 5,
#      cv::fisheye::CALIB_FIX_K3 = 1 << 6,
#      cv::fisheye::CALIB_FIX_K4 = 1 << 7,
#      cv::fisheye::CALIB_FIX_INTRINSIC = 1 << 8,
#      cv::fisheye::CALIB_FIX_PRINCIPAL_POINT = 1 << 9
#    }
    
    flags = switcherIntrCalibFlags[model]
    
    if criteria is None:
        # terminaion criteria
        # cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, int(1e5), 1e-16)
        # estimated DBL_EPSILON as 4.440892098500626e-16 using the algortihm
        # in https://en.wikipedia.org/wiki/Machine_epsilon#Approximation
        criteria = (3, 50, 1e-15)
    
    return switcherIntrinsicFunc[model](objpoints, imgpoints, imgSize, K, D,
                                        flags=flags, criteria=criteria)

