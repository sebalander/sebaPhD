#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 17:26:31 2017

test intrinsic calibration paramters

@author: sebalander
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:30:53 2016

calibrates intrinsic with diff distortion model

@author: sebalander
"""

# %%
import glob
import numpy as np
import scipy.linalg as ln
from calibration import calibrator as cl
import matplotlib.pyplot as plt
from importlib import reload


# %% LOAD DATA
# input
plotCorners = False
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye']

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# load data
imagePoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')

n = len(imagePoints)  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)

# %% calculate change in radius for all points in homogenous plane
reload(cl)
RP = dict()  # undistorted and distorted homogenous points
RPP = dict()
K = dict()  # linear and non linear distortion
D = dict()
RV = dict()  # rvect tvecs
TV = dict()
IP = dict()  # PROJECTEED image points and object points
OP = dict()

for model in modelos:
    print('processing model', model)
    # model data files
    distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
    linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
    tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
    rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"
    # load model specific data
    distCoeffs = np.load(distCoeffsFile)
    cameraMatrix = np.load(linearCoeffsFile)
    rVecs = np.load(rVecsFile)
    tVecs = np.load(tVecsFile)
    
    # initiate dictionaries
    RP[model] = []
    RPP[model] = []
    K[model] = cameraMatrix
    D[model] = distCoeffs
    RV[model] = rVecs
    TV[model]= tVecs
    IP[model] = []
    OP[model] = []

    # MAP TO HOMOGENOUS PLANE TO GET RADIUS
    for j in range(n):
        print('\t imagen', j)
        
        rvec = rVecs[j]
        tvec = tVecs[j]
        
        imagePointsProjected = cl.direct(chessboardModel, rvec, tvec,
                                         cameraMatrix, distCoeffs, model)
        imagePointsProjected = imagePointsProjected.reshape((-1,2))
        IP[model].append(imagePointsProjected)
        
        objectPointsProjected = cl.inverse(imagePoints[j,0], rvec, tvec,
                                            cameraMatrix, distCoeffs, model)
        #objectPointsProjected = opbjectPointsProjected.reshape((-1,3))
        OP[model].append(objectPointsProjected)
        
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
        
    # to array
    RP[model] = np.array(RP[model]).flatten()
    RPP[model] = np.array(RPP[model]).flatten()
    OP[model] = np.array(OP[model])
    IP[model] = np.array(IP[model])

0

# %% plot comparison of models
reload(cl)
rpMax = np.max([np.max(RP[model]) for model in modelos])
rppMax = np.max([np.max(RPP[model]) for model in modelos])
rp0 = np.linspace(0, rpMax*1.2 ,100)
rpp0 = np.linspace(0, rppMax*1.2 ,100)

clr = {modelos[0]:'b', modelos[1]:'r', modelos[2]:'k'}

plt.figure(n)
plt.xlim([0, rpMax*1.2])
plt.ylim([0, rppMax*1.2])
plt.xlabel('Undistorted radius (r\')')
plt.ylabel('Distorted radius (r\")')




for model in modelos:
    plt.plot(RP[model], RPP[model], '.', markersize=3)
    
    rpp1 = cl.distort[model](rp0, D[model])
    plt.plot(rp0, rpp1, '-',c=clr[model], lw=1, label=model+' direct')
    
    rp1, _ = cl.undistort[model](rpp0, D[model])
    plt.plot(rp1, rpp0, '--',c=clr[model], lw=1, label=model+' inverse')


plt.legend()
plt.tight_layout()

# %% comparo tvecs
# se nota que el modelo compensa diferencia de distorsion con un cambio en la
# rototraslacion. pero en todos los casos tiene una performace cualitativamente
# aceptable

plt.figure(n+1)
plt.subplot(221)
plt.plot([-20, 30], [-20, 30], '--k', lw=0.5)
plt.plot(TV['rational'][:,:,0],TV['fisheye'][:,:,0], '+')
plt.xlabel('Tvec rational')
plt.ylabel('Tvec fisheye')
plt.legend()

plt.subplot(222)
plt.plot([-20, 30], [-20, 30], '--k', lw=0.5)
plt.plot(TV['rational'][:,:,0],TV['poly'][:,:,0], '+')
plt.xlabel('Tvec rational')
plt.ylabel('Tvec poly')
plt.legend()

plt.subplot(223)
plt.plot([-np.pi, np.pi], [-np.pi, np.pi], '--k', lw=0.5)
plt.plot(RV['rational'][:,:,0],RV['fisheye'][:,:,0], '+',)
plt.xlabel('Rvec rational')
plt.ylabel('Rvec fisheye')
plt.legend()

plt.subplot(224)
plt.plot([-np.pi, np.pi], [-np.pi, np.pi], '--k', lw=0.5)
plt.plot(RV['rational'][:,:,0],RV['poly'][:,:,0], '+')
plt.xlabel('Rvec rational')
plt.ylabel('Rvec poly')
plt.legend()

plt.tight_layout()


# %%
from scipy.special import chdtriv
fi = np.linspace(0,2*np.pi,20)
Xcirc = np.array([np.cos(fi), np.sin(fi)]) * chdtriv(0.1, 2)

def plotEllipse(ax, c, mux, muy, col):
    l, v = ln.eig(ln.inv(c))
    
    D = v.T*np.sqrt(l.real) # queda con los autovectores como filas
    
    xeli, yeli = np.dot(ln.inv(D), Xcirc)
    
    ax.plot(xeli+mux, yeli+muy, c=col, lw=0.5)

#

# %% comparo los errores proyectando sobre las imagenes
from matplotlib.patches import Ellipse
objectPoints = chessboardModel.reshape(-1,3)
imagePoints.shape


plt.figure(n+2)
# corro sobre las imagenes
plt.subplot(121)
x0, y0 = imagePoints[:,0,:].reshape(-1,2).T
plt.plot(x0, y0, '+k', markersize=1)
for model in modelos:
    x1, y1 = IP[model].reshape(-1,2).T
    plt.plot(x1, y1, '.', color=clr[model], markersize=1)


plt.subplot(122)
x0, y0 = objectPoints[:,:2].T

ax = plt.gca()
ax.plot(x0, y0, '+k', markersize=7)
for model in modelos:
    x1, y1, _ = OP[model].transpose((2,1,0))
    ax.plot(x1, y1, '.', color=clr[model], markersize=2)
    # centrides
    mux = np.mean(x1, axis=1)
    muy = np.mean(y1, axis=1)
    # errores
    ex = x1 - mux.reshape(-1,1)
    ey = y1 - muy.reshape(-1,1)
    E = np.array([ex, ey]).transpose(1,0,2)
    # covarianzas
    C = np.array([np.dot(EE, EE.T) for EE in E]) / x1.shape[0]
    
    # calculo de la elipse a partir de la covarianza
    for i in range(len(C)):
        plotEllipse(ax, C[i], mux[i], muy[i], clr[model])
        

plt.tight_layout()

# %% check projection in one particular image
imFiles = glob.glob(imagesFolder + "*.png")  # list of images
image2check = 'vlcsnap-2017-04-03-22h00m06s444.png'

n = np.argwhere([f==imagesFolder+image2check for f in imFiles])
n.shape = -1
n = n[0]

# %%
plt.figure()
plt.plot(x0, y0, '+k', markersize=7)

model='fisheye'
x1, y1, _ = OP[model].transpose((2,1,0))
plt.plot(x1[:,n], y1[:,n], 'x', color=clr[model], markersize=5)
# centrides
#
