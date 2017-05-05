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

0

# %% plot comparison of models
rp0 = np.linspace(0,np.max(RP[model])*1.2,500)

plt.figure(n)
plt.xlim([rp0[0], rp0[-1]])

for model in modelos:
    plt.plot(RP[model], RPP[model], '.', markersize=3)
    plt.ylim([0, np.max(RPP[model])*1.2])
    
    rpp0 = cl.distort[model](rp0, D[model])
    plt.plot(rp0, rpp0, '-', lw=1, label=model)

plt.legend()
plt.plot()

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

plt.subplot(222)
plt.plot([-20, 30], [-20, 30], '--k', lw=0.5)
plt.plot(TV['rational'][:,:,0],TV['poly'][:,:,0], '+')
plt.xlabel('Tvec rational')
plt.ylabel('Tvec poly')

plt.subplot(223)
plt.plot([-np.pi, np.pi], [-np.pi, np.pi], '--k', lw=0.5)
plt.plot(RV['rational'][:,:,0],RV['fisheye'][:,:,0], '+')
plt.xlabel('Rvec rational')
plt.ylabel('Rvec fisheye')

plt.subplot(224)
plt.plot([-np.pi, np.pi], [-np.pi, np.pi], '--k', lw=0.5)
plt.plot(RV['rational'][:,:,0],RV['poly'][:,:,0], '+')
plt.xlabel('Rvec rational')
plt.ylabel('Rvec poly')


plt.tight_layout()


# %% comparo los errores proyectando sobre las imagenes
objectPoints = chessboardModel.reshape(-1,3)
imagePoints.shape

clr = {modelos[0]:'b', modelos[1]:'r', modelos[2]:'k'}


plt.figure(n+2)
plt.subplot(121)
for model in modelos:
    # corro sobre las imagenes
    for i in range(n):
        x0, y0 = imagePoints[i,0,:].T
        x1, y1 = IP[model][i].T
        
        plt.plot([x0,x1], [y0,y1], color=clr[model], lw=1)


plt.subplot(122)
x0, y0 = objectPoints[:,:2].T

# %%
model = modelos[0]
for model in modelos:
    # corro sobre las imagenes
    for i in range(n):
        x1, y1, _ = OP[model][i].T
        
        # plt.plot([x0, x1], [y0, y1], color=clr[model], lw=1)
        plt.scatter(x1, y1,c=clr[model],s=1)
    plt.scatter(x0, y0,c='g',marker='+')
0

