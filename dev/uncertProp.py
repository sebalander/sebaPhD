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


def cart2sphe(X):
    '''
    convert vector to spherical coordinates
    '''
    r = ln.norm(X)
    # se asume seno de b positivo porque b esta en [0, pi]
    a = np.arctan2(X[1], X[0])
    
    # elijo un par de intervalos donde debo usar el seno, para el resto uso cos
    if (1 < a < 2.5) or (-2.5 < a < -1):
        b = np.arctan2(X[1]/np.sin(a), X[2])
    else:
        b = np.arctan2(X[0]/np.cos(a), X[2])
    
    return r, a, b


def jacobianosHom2Map(r, a, b, tx, ty, tz, xp, yp):
    '''
    returns the jacobians needed to calculate the propagation of uncertainty
    
    '''
    x0 = ty - tz*yp
    x3, x6, x2 = np.sin([r, a, b])
    x9, x1, x8 = np.cos([r, a, b])
    x4 = x2*x3
    x5 = x1*x4
    x7 = x2*x6
    x10 = -x9
    x11 = x10 + 1
    x12 = x11*x8
    x13 = x12*x7
    x14 = -x13 - x5
    x15, x16, x22, x26 = np.square([x6, x2, x1, x8])
    x17 = x11*x16
    x18 = x15*x17
    x19 = x13 + x5
    x20 = x19*yp
    x21 = x18 - x20 + x9
    x23 = x17*x22
    x24 = x3*x7
    x25 = -x24
    x27 = x11*x26
    x28 = x25 + x27
    x29 = x28*xp
    x30 = x23 - x29 + x9
    x31 = x3*x8
    x32 = x1*x6
    x33 = x17*x32
    x34 = -x28*yp + x31 + x33
    x35 = -x19*xp - x31 + x33
    x36 = x21*x30 - x34*x35
    x37 = 1/x36
    x38 = x24 - x27
    x39 = x14*x34 - x21*x38
    x40 = tx - tz*xp
    x41 = x36**(-2)
    x42 = x41*(x0*x35 - x21*x40)
    x43 = -x14*x30 + x35*x38
    x44 = x41*(-x0*x30 + x34*x40)
    x45 = -x18
    x46 = -x3
    x47 = x16*x3
    x48 = x1*x2
    x49 = x31*x7 + x48*x9
    x50 = x15*x47 + x46 - x49*yp
    x51 = x8*x9
    x52 = x32*x47
    x53 = -x49*xp - x51 + x52
    x54 = x26*x3 - x7*x9
    x55 = x22*x47 + x46 - x54*xp
    x56 = x51 + x52 - x54*yp
    x57 = -x21*x55 - x30*x50 + x34*x53 + x35*x56
    x58 = 2*x33
    x59 = x12*x48 + x25
    x60 = x58 - x59*yp
    x61 = x23 + x45
    x62 = -x59*xp + x61
    x63 = x5*xp - x58
    x64 = x5*yp + x61
    x65 = -x21*x63 - x30*x60 + x34*x62 + x35*x64
    x66 = 2*x12*x2
    x67 = x1*x31 - x17*x6 + x27*x6
    x68 = x15*x66 - x67*yp
    x69 = 2*x1*x13
    x70 = x4 - x67*xp + x69
    x71 = -x31*x6 - x66
    x72 = x22*x66 - x71*xp
    x73 = -x4 + x69 - x71*yp
    x74 = -x21*x72 - x30*x68 + x34*x70 + x35*x73
    x75 = r*x2
    x76 = r*x8
    
    # jacobiano de la pos en el mapa con respecto a las posiciones homogeneas
    JX_Xp = np.array([[x37*(tz*x21 + x0*x14) + x39*x42,
                       x37*(-tz*x35 - x14*x40) + x42*x43],
                      [x37*(-tz*x34 - x0*x38) + x39*x44,
                       x37*(tz*x30 + x38*x40) + x43*x44]], dtype=float)
    
    # jacobiano respecto a la traslacion
    JX_tV = np.array([[x37*(x10 + x20 + x45), x35*x37, x37*(x21*xp - x35*yp)],
                      [x34*x37, x37*(x10 - x23 + x29), x37*(x30*yp - x34*xp)]],
                     dtype=float)
    
    # jacobiano de  posiciones en mapa wrt vector de rodrigues en esfericas
    JX_rVsph = np.array([[x37*(x0*x53 - x40*x50) + x42*x57,
                          x37*(x0*x62 - x40*x60) + x42*x65,
                          x37*(x0*x70 - x40*x68) + x42*x74],
                         [x37*(-x0*x55 + x40*x56) + x44*x57,
                          x37*(-x0*x63 + x40*x64) + x44*x65,
                          x37*(-x0*x72 + x40*x73) + x44*x74]], dtype=float)
    
    # jacobiano de vector de rodrigues cartesiano wrt esfericas
    JrV_rVsph = np.array([[x48, -x6*x75, x1*x76],
                          [x7,   x1*x75, x6*x76],
                          [x8,        0,   -x75]], dtype=float)

    JrVsph_rV = ln.inv(JrV_rVsph)

    return JX_Xp, JX_tV, JX_rVsph, JrVsph_rV

#

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
objpoints = np.array([chessboardModel]*nIm)

distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# load model specific data
distCoeffs = np.load(distCoeffsFile).reshape(-1)
cameraMatrix = np.load(linearCoeffsFile)
rVecs = np.load(rVecsFile)
tVecs = np.load(tVecsFile)

# covarianza de los parametros intrinsecos de la camara
Cf = np.eye(4)*(0.1)**2
Ck = np.diag((distCoeffs[[0,1,4,5,6,7]]*0.001)**2)  # 0.1% error distorsion

# %% choose image
j = 2
print('\t imagen', j)
imagePoints = np.load(cornersFile)
imagePoints = imagePoints[j,0]

img = plt.imread(images[j])

# covariances
Cccd = (0.5**2) * np.array([np.eye(2)]*imagePoints.shape[0])

# plot initial uncertanties
fig = plt.figure()
ax = fig.gca()
ax.imshow(img)
for i in range(nPts):
    x, y = imagePoints[i]
    plotEllipse(ax, Cccd[i], x, y, 'b')

#

# %% propagate to homogemous

xpp, ypp, Cpp = cl.ccd2hom(imagePoints, cameraMatrix, cov=Cccd, Cf=Cf)

fig = plt.figure()
ax = fig.gca()
for i in range(nPts):
    plotEllipse(ax, Cpp[i], xpp[i], ypp[i], 'b')

# 

# %% go to undistorted homogenous
rpp = ln.norm([xpp, ypp], axis=0)

# calculate ratio of undistorition and it's derivative wrt radius
q, _, dqI, dqDk = cl.undistort[model](rpp, distCoeffs, quot=True, der=True)

xp = q * xpp # undistort in homogenous coords
yp = q * ypp

rp = rpp * q
#plt.figure()
#plt.plot(rp, rpp, '+')

xpp2 = xpp**2
ypp2 = ypp**2
xypp = xpp * ypp
dqIrpp = dqI / rpp

# calculo jacobiano
J = np.array([[xpp2, xypp],[xypp, ypp2]])
J *= dqIrpp.reshape(1,1,-1)
J[0,0,:] += q
J[1,1,:] += q

Jk = np.array([xpp, ypp]).T.reshape(-1,2,1) * dqDk.T.reshape(-1,1,6)

Cp = np.empty_like(Cpp)

for i in range(nPts):
    Caux = Cpp[i] + Jk[i].dot(Ck).dot(Jk[i].T)
    Cp[i] = J[:,:,i].dot(Caux).dot(J[:,:,i].T)


fig = plt.figure()
ax = fig.gca()
for i in range(nPts):
    plotEllipse(ax, Cp[i], xp[i], yp[i], 'b')
#

# %% project to map
# cargo la rototraslacion
rV = rVecs[j].reshape(-1)
tV = tVecs[j].reshape(-1)

# invento unas covarianzas
Cr = np.diag((rV*0.001)**2)
Ct = np.diag((tV*0.001)**2)

r, a, b = cart2sphe(rV)
tx, ty, tz = tV.reshape(-1)

JX_Xp, JX_tV, JX_rVsph, JrVsph_rV = jacobianosHom2Map(r, a, b, tx, ty, tz, xp, yp)


C = np.empty_like(Cp)
#Caux = JrVsph_rV.dot(Cr).dot(JrVsph_rV.T)
#ln.svd(JrVsph_rV)[1]
for i in range(nPts):
    i, "      "
    JX_rV = JX_rVsph[:,:,i].dot(JrVsph_rV)
    ln.svd(JX_rV)[1]
    C[i] = JX_rV.dot(Cr).dot(JX_rV.T)
    
    ln.svd(JX_tV[:,:,i])[1]
    C[i] += JX_tV[:,:,i].dot(Ct).dot(JX_tV[:,:,i].T)
    
    ln.svd(JX_Xp[:,:,i])[1]
    C[i] += JX_Xp[:,:,i].dot(Cp[i]).dot(JX_Xp[:,:,i].T)


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





