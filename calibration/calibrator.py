# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:47:18 2016


@author: sebalander
"""
# %% IMPORTS
from matplotlib.pyplot import plot, imshow, legend, show, figure, gcf, imread
from matplotlib.pyplot import xlabel, ylabel
from cv2 import Rodrigues, findHomography
from numpy import min, max, ndarray, zeros, array, reshape, sqrt, roots
from numpy import sin, cos, cross, ones, concatenate, flipud, dot, isreal
from numpy import linspace, polyval, eye
from scipy.linalg import sqrtm, norm, inv
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# %% Z=0 PROJECTION

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

# %% HOMOGRAPHY
def pose2homogr(rVec,tVec):
    '''
    generates the homography from the pose descriptors
    '''
    
    # calcular las puntas de los versores
    if rVec.shape == (3,3):
        [x,y,z] = rVec
    else:
        [x,y,z] = Rodrigues(rVec)[0]
    
    
    H = array([x,y,tVec[:,0]]).T
    
    return H



def homogr2pose(H):
    '''
    returns pose from the homography matrix
    '''
    r1B = H[:,0]
    r2B = H[:,1]
    r3B = cross(r1B,r2B)
    
    # convert to versors of B described in A ref frame
    r1 = array([r1B[0],r2B[0],r3B[0]])
    r2 = array([r1B[1],r2B[1],r3B[1]])
    r3 = array([r1B[2],r2B[2],r3B[2]])
    
    rot = array([r1, r2, r3]).T
    # make sure is orthonormal
    rotNorm = rot.dot(inv(sqrtm(rot.T.dot(rot))))
    rVec = Rodrigues(rotNorm)[0]  # make into rot vector
    
    # rotate to get displ redcribed in A
    # tVec = np.dot(rot, -homography[2]).reshape((3,1))
    tVec = H[:,2].reshape((3,1))
    
    # rescale
    k = sqrt(norm(r1)*norm(r2))  # geometric mean
    tVec = tVec / k
    
    return [rVec, tVec]




def estimateInitialPose(fiducialPoints, corners, linearCoeffs):
    '''
    estimateInitialPose(fiducialPoints, corners, f, imgSize) -> [rVecs, tVecs, Hs]
    
    Recieves fiducial points and list of corners (one for each image), proposed
    focal distance 'f' and returns the estimated pose of the camera. asumes
    pinhole model.
<<<<<<< HEAD:poseCalibration.py
=======
    
    this function doesn't work very well, use at your own risk
>>>>>>> 6d680e7d79797264e8842d99b669d04af01ee6e0:calibration/poseCalibration.py
    '''
    
    src = fiducialPoints[0]+[0,0,1]
    unos = ones((src.shape[0],1))
    
    rVecs = list()
    tVecs = list()
    Hs = list()
    
    if len(corners.shape) < 4:
        corners = array([corners])
    
    for cor in corners:
        # flip 'y' coordinates, so it works
        # why flip image:
        # http://stackoverflow.com/questions/14589642/python-matplotlib-inverted-image
        # dst = [0, imgSize[0]] + cor[:,0,:2]*[1,-1]
        x,y = cor.transpose((2,0,1))
        # le saque el -1 y el corrimiento y parece que da mejor??
        
        # take to homogenous plane asuming intrinsic pinhole
        dst = concatenate(((x-linearCoeffs[0,2])/linearCoeffs[0,0],
                           (y-linearCoeffs[1,2])/linearCoeffs[1,1],
                           unos), axis=1)
        # fit homography
        H = findHomography(src[:,:2], dst[:,:2], method=0)[0]
        Hs.append(H)
        
        rVec, tVec = homogr2pose(H)
        rVecs.append(rVec)
        tVecs.append(tVec)
    
    return [array(rVecs), array(tVecs), array(Hs)]

# %%
from calibration import StereographicCalibration as stereographic
from calibration import UnifiedCalibration as unified
from calibration import RationalCalibration as rational
from calibration import FisheyeCalibration as fisheye
from calibration import PolyCalibration as poly


# %% PARAMETER HANDLING
def formatParameters(rVec, tVec, linearCoeffs, distCoeffs, model):
    
    switcher = {
    'stereographic' : stereographic.formatParameters,
    'unified' : unified.formatParameters,
    'rational' : rational.formatParameters,
    'poly' : poly.formatParameters,
    'fisheye' : fisheye.formatParameters
    }
    
    return switcher[model](rVec, tVec, linearCoeffs, distCoeffs)

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

def direct(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs, model):
    
    switcher = {
    'stereographic' : stereographic.direct,
    'unified' : unified.direct,
    'rational' : rational.direct,
    'poly' : poly.direct,
    'fisheye' : fisheye.direct
    }
    
    return switcher[model](fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs)

def residualDirect(params, fiducialPoints, imageCorners, model):
    
    switcher = {
    'stereographic' : stereographic.residualDirect,
    'unified' : unified.residualDirect,
    'rational' : rational.residualDirect,
    'poly' : poly.residualDirect,
    'fisheye' : fisheye.residualDirect
    }
    
    return switcher[model](params, fiducialPoints, imageCorners)


def calibrateDirect(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs, model):
    
    switcher = {
    'stereographic' : stereographic.calibrateDirect,
    'unified' : unified.calibrateDirect,
    'rational' : rational.calibrateDirect,
    'poly' : poly.calibrateDirect,
    'fisheye' : fisheye.calibrateDirect
    }
    
    return switcher[model](fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs)


# %% INVERSE PROJECTION

def inverse(imageCorners, rVec, tVec, linearCoeffs, distCoeffs, model):
    '''
    inverseFisheye(objPoints, rVec/rotMatrix, tVec, cameraMatrix,
                    distCoeffs)-> objPoints
    takes corners in image and returns coordinates in scene
    corners must be of size (n,1,2)
    objPoints has size (1,n,3)
    ignores tangential and tilt distortions
    '''
    switcher = {
    'stereographic' : stereographic.inverse,
    'unified' : unified.inverse,
    'rational' : rational.inverse,
    'poly' : poly.inverse,
    'fisheye' : fisheye.inverse
    }
    
    return switcher[model](imageCorners, rVec, tVec, linearCoeffs, distCoeffs)

def residualInverse(params, fiducialPoints, imageCorners, model):
    
    switcher = {
    'stereographic' : stereographic.residualInverse,
    'unified' : unified.residualInverse,
    'rational' : rational.residualInverse,
    'poly' : poly.residualInverse,
    'fisheye' : fisheye.residualInverse
    }
    
    return switcher[model](params, fiducialPoints, imageCorners)


def calibrateInverse(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs, model):
    
    switcher = {
    'stereographic' : stereographic.calibrateInverse,
    'unified' : unified.calibrateInverse,
    'rational' : rational.calibrateInverse,
    'poly' : poly.calibrateInverse,
    'fisheye' : fisheye.calibrateInverse
    }
    
    return switcher[model](fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs)

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
def fiducialComparison(fiducial1, fiducial2=None,
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
    t = tVec.reshape((3))
    
    # calcular las puntas de los versores
    if rVec.shape == (3,3):
        [x,y,z] = rVec.T
    else:
        [x,y,z] = Rodrigues(rVec)[0].T
    
    print(array([x,y,z]))
    [x,y,z] = [x,y,z] + t
    print(t)
    
    fiducial1 = fiducial1.reshape(-1,3)
    X1, Y1 ,Z1 = fiducial1.T  # [:,0]
    # Y1 = fiducial1[:,1]
    # Z1 = fiducial1[:,2]
    
    fig = figure()
    ax = fig.gca(projection='3d')
    
    ax.plot(X1,Y1,Z1,'xr', markersize=10, label=label1)
    
    if fiducial2 is not None:
        fiducial2 = fiducial2.reshape(-1,3)
        X2, Y2, Z2 = fiducial2.T
        # Y2 = fiducial2[:,1]
        # Z2 = fiducial2[:,2]
        ax.plot(X2,Y2,Z2,'+b', markersize=10, label=label2)
        xmin = min([0, min(X1), min(X2), min([t[0],x[0],y[0],z[0]])])
        ymin = min([0, min(Y1), min(Y2), min([t[1],x[1],y[1],z[1]])])
        zmin = min([0, min(Z1), min(Z2), min([t[2],x[2],y[2],z[2]])])
        xmax = max([0, max(X1), max(X2), max([t[0],x[0],y[0],z[0]])])
        ymax = max([0, max(Y1), max(Y2), max([t[1],x[1],y[1],z[1]])])
        zmax = max([0, max(Z1), max(Z2), max([t[2],x[2],y[2],z[2]])])
    else:
        xmin = min([0, min(X1), min([t[0],x[0],y[0],z[0]])])
        ymin = min([0, min(Y1), min([t[1],x[1],y[1],z[1]])])
        zmin = min([0, min(Z1), min([t[2],x[2],y[2],z[2]])])
        xmax = max([0, max(X1), max([t[0],x[0],y[0],z[0]])])
        ymax = max([0, max(Y1), max([t[1],x[1],y[1],z[1]])])
        zmax = max([0, max(Z1), max([t[2],x[2],y[2],z[2]])])
    
    ax.set_xbound(xmin, xmax)
    ax.set_ybound(ymin, ymax)
    ax.set_zbound(zmin, zmax)
    
    ejeX = Arrow3D([0, 1],
                   [0, 0],
                   [0, 0],
                   mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    ejeY = Arrow3D([0, 0],
                   [0, 1],
                   [0, 0],
                   mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    ejeZ = Arrow3D([0, 0],
                   [0, 0],
                   [0, 1],
                   mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    
    origen = Arrow3D([0, t[0]],
                     [0, t[1]],
                     [0, t[2]],
                     mutation_scale=20, lw=1, arrowstyle="-", color="k",
                     linestyle="dashed")
    
    ejeXc = Arrow3D([t[0], x[0]],
                   [t[1], x[1]],
                   [t[2], x[2]],
                   mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
    ejeYc = Arrow3D([t[0], y[0]],
                   [t[1], y[1]],
                   [t[2], y[2]],
                   mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
    ejeZc = Arrow3D([t[0], z[0]],
                   [t[1], z[1]],
                   [t[2], z[2]],
                   mutation_scale=20, lw=3, arrowstyle="-|>", color="b")
    
    ax.add_artist(ejeX)
    ax.add_artist(ejeY)
    ax.add_artist(ejeZ)
    ax.add_artist(origen)
    ax.add_artist(ejeXc)
    ax.add_artist(ejeYc)
    ax.add_artist(ejeZc)
    
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


def plotHomographyToMatch(fiducialPoints, corners, f, imgSize, images=None):
    
    src = fiducialPoints[0]+[0,0,1]
    
    
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



def plotForwardHomography(fiducialPoints, corners, f, imgSize, Hs, images=None):
    src = fiducialPoints[0]+[0,0,1]
    
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


def plotBackwardHomography(fiducialPoints, corners, f, imgSize, Hs):
    src = fiducialPoints[0]+[0,0,1]
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
        
        
        fiducialProjected = (srcP-[0,0,1]).reshape(fiducialPoints.shape)
        
        rVec, tVec = homogr2pose(Hs[i]);
        fiducialComparison3D(rVec, tVec,
                             fiducialPoints, fiducialProjected,
                             label1="fiducial points",
                             label2="%d ajuste"%i)

def plotRationalDist(distCoeffs,imgSize, linearCoeffs):
    
    k = distCoeffs[[0,1,4,5,6,7],0]
    pNum = [0, 1, 0, k[0], 0, k[1], 0, k[2]]
    pDen = [1, 0, k[3], 0, k[4], 0, k[5]]
    
    # buscar un buen rango de radio
    rDistMax = sqrt((linearCoeffs[0,2]/linearCoeffs[0,0])**2 +
                    (linearCoeffs[1,2]/linearCoeffs[1,1])**2)
    
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
    

# %% INRINSIC CALIBRATION

def calibrateIntrinsic(objpoints, imgpoints, imgSize, model, K=None, D=None,
                       flags=None, criteria=None):
    '''
    only available for rational and fisheye, we use opencv's functions
    exclusively here
    
    parameters defined by me for default use:
        K = [[600.0, 0.0,   imgSize[1]/2],
             0.0,    600.0, imgSize[0]/2],
             0.0,    0.0,        1]]
        
        flags = 1 + 512 # no skew and fixed ppal point, seems more apropiate
                          to our camera model
        criteria = (3, int(1e5), 1e-15)
    
    return rms, K, D, rVecs, tVecs
    '''
    
    if K is None:
        K = eye(3)
        K[0, 2] = imgSize[1]/2
        K[1, 2] = imgSize[0]/2
        K[0, 0] = K[1, 1] = 600.0
    
    if flags is None:
        #.CALIB_FIX_SKEW = 1
        # CALIB_FIX_PRINCIPAL_POINT = 512
        # CALIB_ZERO_TANGENT_DIST = 8
        flags = 1 + 8 + 512
    
    if criteria is None:
        # terminaion criteria
        # cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, int(1e5), 1e-16)
        # estimated DBL_EPSILON as 4.440892098500626e-16 using the algortihm
        # in https://en.wikipedia.org/wiki/Machine_epsilon#Approximation
        criteria = (3, 50, 1e-15)
    
    switcher = {
    'rational' : rational.calibrateIntrinsic,
    'poly' : poly.calibrateIntrinsic,
    'fisheye' : fisheye.calibrateIntrinsic
    }
    
    return switcher[model](objpoints, imgpoints, imgSize, K, D,
                           flags=flags, criteria=criteria)

