#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:46:50 2017

@author: sebalander
"""
# %%
import numpy as np
import glob
from calibration import calibrator as cl
import matplotlib.pyplot as plt
from importlib import reload
import scipy.linalg as ln
import cv2

# %%
from scipy.special import chdtriv
fi = np.linspace(0,2*np.pi,20)
Xcirc = np.array([np.cos(fi), np.sin(fi)]) * chdtriv(0.1, 2)

def plotEllipse(ax, C, mux, muy, col):
    '''
    se grafica una elipse asociada a la covarianza c, centrada en mux, muy
    '''
    Ci = ln.inv(C)
    
    l, v = ln.eig(Ci)
    
    # matrix such that A.dot(A.T)==Ci
    A = np.sqrt(l.real) * v
    # roto eescaleo para llevae al cicuulo a la elipse
    xeli, yeli = np.dot(Xcirc.T, ln.inv(A)).T
    
    ax.plot(xeli+mux, yeli+muy, c=col, lw=0.5)

#


# %% LOAD DATA
# input
plotCorners = False
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye']
model = modelos[1]

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# load data
imagePoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')

nIm, _, nPts, _ = imagePoints.shape  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)

distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# load model specific data
distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)
rVecs = np.load(rVecsFile)
tVecs = np.load(tVecsFile)

# %% choose image
j = 10 # 18
print('\t imagen', j)
imagePoints = np.load(cornersFile)
imagePoints = imagePoints[j,0]

img = plt.imread(images[j])

# covariances
Cccd = 2 * np.array([np.eye(2)]*imagePoints.shape[0])

# %% plot initial uncertanties
fig = plt.figure()
ax = fig.gca()
ax.imshow(img)
for i in range(nPts):
    x, y = imagePoints[i]
    plotEllipse(ax, Cccd[i], x, y, 'b')

#

# %% propagate to homogemous

xpp, ypp, Cpp = cl.ccd2hom(imagePoints, cameraMatrix, cov=Cccd)

fig = plt.figure()
ax = fig.gca()
for i in range(nPts):
    plotEllipse(ax, Cpp[i], xpp[i], ypp[i], 'b')

# 

# %% go to undostorted homogenous
rpp = ln.norm([xpp, ypp], axis=0)

# calculate ratio of undistorition and it's derivative wrt radius
q, _, dqI = cl.undistort[model](rpp, distCoeffs, quot=True, der=True)

xp = q * xpp # undistort in homogenous coords
yp = q * ypp

rp = rpp * q
plt.plot(rp, rpp, '+')

xpp2 = xpp**2
ypp2 = ypp**2
xypp = xpp * ypp
dqIrpp = dqI / rpp

# calculo jacobiano
J = np.array([[xpp2, xypp],[xypp, ypp2]])
J *= dqIrpp.reshape(1,1,-1)
J[0,0,:] += q
J[1,1,:] += q

Cp = np.empty_like(Cpp)
for i in range(nPts):
    Cp[i] = J[:,:,i].dot(Cpp[i]).dot(J[:,:,i].T)


fig = plt.figure()
ax = fig.gca()
for i in range(nPts):
    plotEllipse(ax, Cp[i], xp[i], yp[i], 'b')

# cargo la rototraslacion
rV = rVecs[j]
tV = tVecs[j]

def rototrasCovariance(Cp, xp, yp, rV, tV):
    a, b, _, c = Cp.flatten()
    mx, my = (xp, yp)
    r11, r12, r21, r22, r31,r32 = cv2.Rodrigues(rV)[0][:,:2].flatten()
    tx, ty, tz = tV.flatten()
    
    C11 = (a*mx**2*r31**2 - 2*a*mx*r11*r31 + a*r11**2 + 2*b*mx*my*r31**2 - 2*b*mx*r21*r31 - 2*b*my*r11*r31 + 2*b*r11*r21 + c*my**2*r31**2 - 2*c*my*r21*r31 + c*r21**2)
    
    C12 = (a*mx**2*r31*r32 - a*mx*r11*r32 - a*mx*r12*r31 + a*r11*r12 + 2*b*mx*my*r31*r32 - b*mx*r21*r32 - b*mx*r22*r31 - b*my*r11*r32 - b*my*r12*r31 + b*r11*r22 + b*r12*r21 + c*my**2*r31*r32 - c*my*r21*r32 - c*my*r22*r31 + c*r21*r22)
    
    C22 = (a*mx**2*r32**2 - 2*a*mx*r12*r32 + a*r12**2 + 2*b*mx*my*r32**2 - 2*b*mx*r22*r32 - 2*b*my*r12*r32 + 2*b*r12*r22 + c*my**2*r32**2 - 2*c*my*r22*r32 + c*r22**2)
    
    C = np.array([[C11, C12], [C12, C22]])
    
    #s=1
    #alfa = -(a*mx**2*r31*tz - a*mx*r11*tz - a*mx*r31*tx + a*r11*tx + 2*b*mx*my*r31*tz - b*mx*r21*tz - b*mx*r31*ty - b*my*r11*tz - b*my*r31*tx + b*r11*ty + b*r21*tx + c*my**2*r31*tz - c*my*r21*tz - c*my*r31*ty + c*r21*ty - r31*s)
    #
    #beta = -(a*mx**2*r32*tz - a*mx*r12*tz - a*mx*r32*tx + a*r12*tx + 2*b*mx*my*r32*tz - b*mx*r22*tz - b*mx*r32*ty - b*my*r12*tz - b*my*r32*tx + b*r12*ty + b*r22*tx + c*my**2*r32*tz - c*my*r22*tz - c*my*r32*ty + c*r22*ty - r32*s)
    #
    #MUx, MUy = ln.inv(C).dot([alfa, beta])

    return C

C = np.empty_like(Cp)
for i in range(nPts):
    C[i] = rototrasCovariance(Cp[i], xp[i], yp[i], rV, tV)

xm, ym, _ = cl.inverse(imagePoints, rV, tV, cameraMatrix, distCoeffs, model).T

fig = plt.figure()
ax = fig.gca()
ax.plot(xm, ym, '+', markersize=2)
for i in range(nPts):
    plotEllipse(ax, C[i], xm[i], ym[i], 'b')

#

# %% step by step calculation
reload(cl)

xp, yp, Cp = cl.ccd2homUndistorted(imagePoints[j].reshape(-1,2), cameraMatrix,  distCoeffs, model, cov=Cccd)

Cp



# %% plot ellipses
fig = plt.figure()
ax = fig.gca()

for i in range(1,len(xp)):
    print(Cp[i], xp[i], yp[i])
    plotEllipse(ax, Cp[i], xp[i], yp[i], 'k')
    
#    c, mux, muy, col = (Cp[i], xp[i], yp[i], 'k')
#    
#    l, v  = ln.eig(c)
#    print(i, l)
#    
#    D = v.T*np.sqrt(l) # queda con los autovectores como filas
#    
#    xeli, yeli = np.dot(ln.inv(D), Xcirc)
#    
#    ax.plot(xeli+mux, yeli+muy, c=col, lw=0.5)

#

# %%
rvec = rVecs[j]
tvec = tVecs[j]

imagePointsProjected = cl.direct(chessboardModel, rvec, tvec,
                                 cameraMatrix, distCoeffs, model)
imagePointsProjected = imagePointsProjected.reshape((-1,2))

objectPointsProjected = cl.inverse(imagePoints[j,0], rvec, tvec,
                                    cameraMatrix, distCoeffs, model)
objectPointsProjected = objectPointsProjected.reshape((-1,3))

if plotCorners:
    imagePntsX = imagePoints[j, 0, :, 0]
    imagePntsY = imagePoints[j, 0, :, 1]

    xPos = imagePointsProjected[:, 0]
    yPos = imagePointsProjected[:, 1]

    plt.figure(j)
    im = plt.imread(images[j])
    plt.imshow(im)
    plt.plot(imagePntsX, imagePntsY, 'xr', markersize=10)
    plt.plot(xPos, yPos, '+b', markersize=10)
    #fig.savefig("distortedPoints3.png")

# calculate distorted radius
xpp, ypp = cl.ccd2hom(imagePoints[j,0], cameraMatrix)
RPP[model].append(ln.norm([xpp,ypp], axis=0))

# calculate undistorted homogenous radius from 3D rototraslation
xyp = cl.rotoTrasHomog(chessboardModel, rVecs[j], tVecs[j])
RP[model].append(ln.norm(xyp, axis=1))





