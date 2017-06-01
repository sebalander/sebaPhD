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
from numpy import sqrt, cos, sin


def jacobianosHom2Map(rx, ry, rz, tx, ty, tz, xp, yp):
    '''
    returns the jacobians needed to calculate the propagation of uncertainty
    
    '''
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
    JXm_Xp = np.array([[x36*(tz*x21 + x0*x17) + x38*x41,
                        x36*(-tz*x33 - x17*x39) + x41*x42],
                       [x36*(-tz*x34 - x0*x37) + x38*x43,
                        x36*(tz*x28 + x37*x39) + x42*x43]])
    
    # jacobiano respecto al vector de rodriguez
    JXm_rV = np.array([[x36*(x0*x65 - x39*x58) + x41*x75,
                        x36*(x0*x85 - x39*x80) + x41*x91,
                        x100*x41 + x36*(x0*x95 - x39*x93)],
                       [x36*(-x0*x72 + x39*x74) + x43*x75,
                       x36*(-x0*x89 + x39*x90) + x43*x91,
                       x100*x43 + x36*(-x0*x98 + x39*x99)]])
    
    # jacobiano respecto a la traslacion
    JXm_tV = np.array([[x36*(x13 - x18 + x20),
                        x33*x36,
                        x36*(x21*xp - x33*yp)],
                       [x34*x36,
                        x36*(x13 - x22 + x27),
                        x36*(x28*yp - x34*xp)]])

    return JXm_Xp, JXm_rV, JXm_tV

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

xpp, ypp, Cpp = cl.ccd2hom(imagePoints, cameraMatrix, Cccd=Cccd, Cf=Cf)

fig = plt.figure()
ax = fig.gca()
for i in range(nPts):
    plotEllipse(ax, Cpp[i], xpp[i], ypp[i], 'b')

# 

# %% go to undistorted homogenous
#
#rpp = ln.norm([xpp, ypp], axis=0)
#
## calculate ratio of undistorition and it's derivative wrt radius
#q, _, dqI, dqDk = cl.undistort[model](rpp, distCoeffs, quot=True, der=True)
#
#xp = q * xpp # undistort in homogenous coords
#yp = q * ypp
#
#rp = rpp * q
##plt.figure()
##plt.plot(rp, rpp, '+')
#
#xpp2 = xpp**2
#ypp2 = ypp**2
#xypp = xpp * ypp
#dqIrpp = dqI / rpp
#
## calculo jacobiano
#J = np.array([[xpp2, xypp],[xypp, ypp2]])
#J *= dqIrpp.reshape(1,1,-1)
#J[0,0,:] += q
#J[1,1,:] += q
#
#Jk = np.array([xpp, ypp]).T.reshape(-1,2,1) * dqDk.T.reshape(-1,1,6)
#
#Cp = np.empty_like(Cpp)
#
#for i in range(nPts):
#    Caux = Cpp[i] + Jk[i].dot(Ck).dot(Jk[i].T)
#    Cp[i] = J[:,:,i].dot(Caux).dot(J[:,:,i].T)

xp, yp, Cp = cl.homDist2homUndist(xpp, ypp, distCoeffs, model, Cpp=Cpp, Ck=Ck)

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
Crt = [Cr, Ct]

#rx, ry, rz = rV.reshape(-1)
#tx, ty, tz = tV.reshape(-1)
#
#JXm_Xp, JXm_rV, JXm_tV = jacobianosHom2Map(rx, ry, rz, tx, ty, tz, xp, yp)
#
#
#C = np.empty_like(Cp)
#
#for i in range(nPts):
#    i, "      "
#    #ln.svd(JXm_Xp[:,:,i])[1]
#    a = JXm_Xp[:,:,i].dot(Cp[i]).dot(JXm_Xp[:,:,i].T)
#    #np.trace(a)
#    C[i] = a
#    
#    #ln.svd(JXm_rV[:,:,i])[1]
#    b = JXm_rV[:,:,i].dot(Cr).dot(JXm_rV[:,:,i].T)
#    #np.trace(b)
#    C[i] += b
#    
#    #ln.svd(JXm_tV[:,:,i])[1]
#    c = JXm_tV[:,:,i].dot(Ct).dot(JXm_tV[:,:,i].T)
#    #np.trace(c)
#    C[i] += c
#    
#    [np.trace(a), np.trace(b), np.trace(c)] / np.trace(C[i])
#
#xm, ym, _ = cl.inverse(imagePoints, rV, tV, cameraMatrix, distCoeffs, model).T


xm, ym, Cm = cl.xypToZplane(xp, yp, rV, tV, Cp=Cp, Crt=[Cr, Ct])

fig = plt.figure()
ax = fig.gca()
ax.plot(xm, ym, '+', markersize=2)
for i in range(nPts):
    plotEllipse(ax, Cm[i], xm[i], ym[i], 'b')



# %% in one line

cl.inverse(imagePoints, rV, tV, cameraMatrix, distCoeffs, model, Cccd, Cf, Ck, Crt)

fig = plt.figure()
ax = fig.gca()
ax.plot(xm, ym, '+', markersize=2)
for i in range(nPts):
    plotEllipse(ax, Cm[i], xm[i], ym[i], 'b')