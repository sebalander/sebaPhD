# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:47:18 2016


@author: sebalander
"""
# %% IMPORTS
from matplotlib.pyplot import plot, imshow, legend, show, figure
from cv2 import Rodrigues
from numpy import min, max, ndarray, zeros, array, reshape
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import poseStereographicCalibration as stereographic
import poseUnifiedCalibration as unified
import poseRationalCalibration as rational
import poseFisheyeCalibration as fisheye

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

def fiducialComparison3D(rVec, tVec, fiducail1, fiducial2 = False, label1 = 'Fiducial points', label2 = 'Projected points'):
    
    t = tVec[:,0]
    
    # calcular las puntas de los versores
    if rVec.shape == (3,3):
        [x,y,z] = rVec
    else:
        [x,y,z], _ = Rodrigues(rVec)
    
    [x,y,z] = [x,y,z] + t
    
    X1 = fiducail1[0,:,0]
    Y1 = fiducail1[0,:,1]
    Z1 = fiducail1[0,:,2]
    
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
    
    ax.add_artist(origen)
    ax.add_artist(ejeX)
    ax.add_artist(ejeY)
    ax.add_artist(ejeZ)
    ax.add_artist(ejeXc)
    ax.add_artist(ejeYc)
    ax.add_artist(ejeZc)
    
    ax.legend()
    
    show()
