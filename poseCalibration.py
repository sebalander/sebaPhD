# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:47:18 2016


@author: sebalander
"""
# %% IMPORTS
from matplotlib.pyplot import plot, imshow, legend, show, figure, gcf, imread
from cv2 import Rodrigues, findHomography
from numpy import min, max, ndarray, zeros, array, reshape, sqrt
from numpy import sin, cos, cross, ones, concatenate, flipud, dot
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




def estimateInitialPose(fiducialPoints, corners, f, imgSize):
    '''
    estimateInitialPose(fiducialPoints, corners, f, imgSize) -> [rVecs, tVecs, Hs]
    
    Recieves fiducial points and list of corners (one for each image), proposed
    focal distance 'f' and returns the estimated pose of the camera. asumes
    pinhole model.
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
        dst = [0, 0] + cor[:,0,:2]*[1,1]
        # le saque el -1 y el corrimiento y parece que da mejor??
        
        # take to homogenous plane asuming intrinsic pinhole
        dst = concatenate( ((dst-imgSize/2)/f, unos), axis=1)
        # fit homography
        H = findHomography(src[:,:2], dst[:,:2], method=0)[0]
        rVec, tVec = homogr2pose(H)
        rVecs.append(rVec)
        tVecs.append(tVec)
        Hs.append(H)
    
    return [array(rVecs), array(tVecs), array(Hs)]

# %%
import poseStereographicCalibration as stereographic
import poseUnifiedCalibration as unified
import poseRationalCalibration as rational
import poseFisheyeCalibration as fisheye

# %% PARAMETER HANDLING
def formatParameters(rVec, tVec, linearCoeffs, distCoeffs, model):
    
    switcher = {
    'stereographic' : stereographic.formatParameters,
    'unified' : unified.formatParameters,
    'rational' : rational.formatParameters,
    'fisheye' : fisheye.formatParameters
    }
    
    return switcher[model](rVec, tVec, linearCoeffs, distCoeffs)

def retrieveParameters(params, model):
    
    switcher = {
    'stereographic' : stereographic.retrieveParameters,
    'unified' : unified.retrieveParameters,
    'rational' : rational.retrieveParameters,
    'fisheye' : fisheye.retrieveParameters
    }
    
    return switcher[model](params)

# %% DIRECT PROJECTION

def direct(fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs, model):
    
    switcher = {
    'stereographic' : stereographic.direct,
    'unified' : unified.direct,
    'rational' : rational.direct,
    'fisheye' : fisheye.direct
    }
    
    return switcher[model](fiducialPoints, rVec, tVec, linearCoeffs, distCoeffs)

def residualDirect(params, fiducialPoints, imageCorners, model):
    
    switcher = {
    'stereographic' : stereographic.residualDirect,
    'unified' : unified.residualDirect,
    'rational' : rational.residualDirect,
    'fisheye' : fisheye.residualDirect
    }
    
    return switcher[model](params, fiducialPoints, imageCorners)


def calibrateDirect(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs, model):
    
    switcher = {
    'stereographic' : stereographic.calibrateDirect,
    'unified' : unified.calibrateDirect,
    'rational' : rational.calibrateDirect,
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
    'fisheye' : fisheye.inverse
    }
    
    return switcher[model](imageCorners, rVec, tVec, linearCoeffs, distCoeffs)

def residualInverse(params, fiducialPoints, imageCorners, model):
    
    switcher = {
    'stereographic' : stereographic.residualInverse,
    'unified' : unified.residualInverse,
    'rational' : rational.residualInverse,
    'fisheye' : fisheye.residualInverse
    }
    
    return switcher[model](params, fiducialPoints, imageCorners)


def calibrateInverse(fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs, model):
    
    switcher = {
    'stereographic' : stereographic.calibrateInverse,
    'unified' : unified.calibrateInverse,
    'rational' : rational.calibrateInverse,
    'fisheye' : fisheye.calibrateInverse
    }
    
    return switcher[model](fiducialPoints, imageCorners, rVec, tVec, linearCoeffs, distCoeffs)

# %% PLOTTING

# plot corners and their projection
def cornerComparison(img, corners1, corners2, label1='Corners', label2='Proyectados'):
    X1 = corners1[:,0,0]
    Y1 = corners1[:,0,1]
    X2 = corners2[:,0,0]
    Y2 = corners2[:,0,1]
    
    figure()
    imshow(img)
    plot(X1, Y1, 'xr', markersize=10, label=label1)
    plot(X2, Y2, '+b', markersize=10, label=label2)
    legend()
    show()

# compare fiducial points
def fiducialComparison(fiducial1, fiducial2, label1='Calibration points', label2='Projected points'):
    X1 = fiducial1[0,:,0]
    Y1 = fiducial1[0,:,1]
    X2 = fiducial2[0,:,0]
    Y2 = fiducial2[0,:,1]
    
    figure()
    plot(X1, Y1, 'xr', markersize=10, label=label1)
    plot(X2, Y2, '+b', markersize=10, label=label2)
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

def fiducialComparison3D(rVec, tVec, fiducial1, fiducial2 = False, label1 = 'Fiducial points', label2 = 'Projected points'):
    
    t = tVec[:,0]
    
    # calcular las puntas de los versores
    if rVec.shape == (3,3):
        [x,y,z] = rVec.T
    else:
        [x,y,z] = Rodrigues(rVec)[0].T
    
    print(array([x,y,z]))
    [x,y,z] = [x,y,z] + t
    print(t)
    X1 = fiducial1[0,:,0]
    Y1 = fiducial1[0,:,1]
    Z1 = fiducial1[0,:,2]
    
    fig = figure()
    ax = fig.gca(projection='3d')
    
    ax.plot(X1,Y1,Z1,'xr', markersize=10, label=label1)
    
    if type(fiducial2) is ndarray:
        X2 = fiducial2[0,:,0]
        Y2 = fiducial2[0,:,1]
        Z2 = fiducial2[0,:,2]
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
                     mutation_scale=20, lw=1, arrowstyle="-", color="k", linestyle="dashed")
    
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

