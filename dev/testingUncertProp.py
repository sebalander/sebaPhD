#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:35:03 2017

test functions that propagate uncertanty

@author: sebalander
"""
# %%
#import time
#import timeit

import numpy as np
#import scipy.linalg as ln
from calibration import calibrator as cl
import matplotlib.pyplot as plt

from importlib import reload
reload(cl)

# %% LOAD DATA
# input
plotCorners = False
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye']
model = modelos[2]

imagesFolder = "/home/sebalander/Desktop/Code/sebaPhD/resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# %% load data
imagePoints = np.load(cornersFile)
n = len(imagePoints)  # cantidad de imagenes
#indexes = np.arange(n)
#
#np.random.shuffle(indexes)
#indexes = indexes
#
#imagePoints = imagePoints[indexes]
#n = len(imagePoints)  # cantidad de imagenes

chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
#images = glob.glob(imagesFolder+'*.png')

# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)
m = chessboardModel.shape[1]  # cantidad de puntos


# model data files
distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"
# load model specific data
distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)
rVecs = np.load(rVecsFile)#[indexes]
tVecs = np.load(tVecsFile)#[indexes]

#
#
## %% simplest test
#j = 0
#
#plt.figure()
#plt.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1],
#            marker='+', c='k', s=100)
#
#for j in range(0,n):
#    xm, ym, Cm = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j],                                        
#                            cameraMatrix, distCoeffs, model)
#    
#    plt.scatter(xm, ym, marker='x', c='b')
#
#
## %%
#ep = 0.01  # realtive standard deviation in parameters
#
## matriz de incerteza de deteccion en pixeles
#Cccd = np.repeat([np.eye(2,2)*0.1**2],imagePoints[j,0].shape[0], axis=0)
#Cf = np.diag(cameraMatrix[[0,1,0,1],[0,1,2,2]] * ep)**2
#Ck = np.diag((distCoeffs.reshape(-1) * ep )**2)
#Cr = np.diag((rVecs[j].reshape(-1) * ep )**2)
#Ct = np.diag((tVecs[j].reshape(-1) * ep )**2)
#
#
#Crt = [Cr, Ct]
#
#
## %%
#xpp, ypp, Cpp = cl.ccd2hom(imagePoints[j,0], cameraMatrix, Cccd, Cf)
#
#
## %% undistort
#xp, yp, Cp = cl.homDist2homUndist(xpp, ypp, distCoeffs, model, Cpp, Ck)
#
#
## %% project to plane z=0 from homogenous
#xm, ym, Cm = cl.xypToZplane(xp, yp, rVecs[j], tVecs[j], Cp, Crt)
#
#Caux = Cm
## %%
#xm, ym, Cm = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j],                                        
#                        cameraMatrix, distCoeffs, model,
#                        Cccd, Cf, Ck, Crt)
#
#fig = plt.figure()
#ax = fig.gca()
#ax.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1])
#cl.plotPointsUncert(ax, Cm, xm, ym, 'k')
#
#
#
#er = [xm, ym] - chessboardModel[0,:,:2].T
##
## %%
#statement1 = '''
#p1 = np.tensordot(er, Cm, axes=(0,1))[range(54),range(54)]
#p2 = p1.dot(er)[range(54),range(54)]
#'''
#
## %%
#statement2 = '''
#Er = np.empty_like(xp)
#for i in range(len(xp)):
#    Er[i] = er[:,i].dot(Cm[i]).dot(er[:,i])
#'''
#
## %%
#statement3 = '''
#q1 = [np.sum(Cm[:,:,0]*er.T,1), np.sum(Cm[:,:,1]*er.T,1)];
#q2 = np.sum(q1*er,0)
#'''
## %%
#
#t1 = timeit.timeit(statement1, globals=globals(), number=10000) / 1e4
#
#t2 = timeit.timeit(statement2, globals=globals(), number=10000) / 1e4
#
#t3 = timeit.timeit(statement3, globals=globals(), number=10000) / 1e4
#
#print(t1/t3, t2/t3)


## %%
#ep = 0.0001  # relative standard deviation in parameters
#
## matriz de incerteza de deteccion en pixeles
#Cccd = np.repeat([np.eye(2,2)*0.1**2],imagePoints[j,0].shape[0], axis=0)
#Cf = np.diag(cameraMatrix[[0,1,0,1],[0,1,2,2]] * ep)**2
#Ck = np.diag((distCoeffs.reshape(-1) * ep )**2)
#
#
#fig = plt.figure()
#ax = fig.gca()
#
#ax.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1],
#            marker='+', c='k', s=100)
#
#for j in range(0,n,3):
#    Cr = np.diag((rVecs[j].reshape(-1) * ep )**2)
#    Ct = np.diag((tVecs[j].reshape(-1) * ep )**2)
#    
#    Crt = [Cr, Ct]
#    
#    xm, ym, Cm = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j],                                        
#                                         cameraMatrix, distCoeffs, model,
#                                         Cccd, Cf, Ck, Crt)
#    
#    cl.plotPointsUncert(ax, Cm, xm, ym, 'k')



# %% poniendo ruido
def sacoParmams(pars):
    d = pars[4:8]
    r = pars[8:11]
    t = pars[11:]
    
    k = np.zeros((3, 3), dtype=float)
    k[[0,1,0,1],[0,1,2,2]] = pars[:4]
    k[2,2] = 1
    
    return r, t, k, d

def sdt2covs(covAll):
    Cf = np.diag(covAll[:4])
    Ck = np.diag(covAll[4:8])
    
    Crt = np.diag(covAll[8:])
    return Cf, Ck, Crt


j = 10 # elijo una imagen para trabajar

import glob
imagenFile = glob.glob(imagesFolder+'*.png')[j]
plt.imshow(plt.imread(imagenFile), origin='lower')
plt.scatter(imagePoints[j,0,:,0], imagePoints[j,0,:,1])

nSampl = int(1e6)  # cantidad de samples

ep = 1e-6  # relative standard deviation in parameters
stdIm = 1e-6

# apilo todos los parametros juntos
parsAll = np.hstack([cameraMatrix[[0,1,0,1],[0,1,2,2]],
                    distCoeffs.reshape(-1),
                    rVecs[j].reshape(-1),
                    tVecs[j].reshape(-1)])

# standard deviations
sAll = ep * parsAll

Cccd = np.repeat([np.eye(2) * stdIm**2], imagePoints[j,0].shape[0], axis=0)
Cf, Ck, Crt = sdt2covs(sAll**2)

# genero todo el ruido
noiseParam = np.random.randn(nSampl, sAll.shape[0])
noisePos = np.random.randn(nSampl, imagePoints[j,0].shape[0], imagePoints[j,0].shape[1])

# dejo los valores preparados
parsGen = parsAll.reshape((1, -1)) + noiseParam * sAll.reshape((1, -1))
posIgen = imagePoints[j,0].reshape((1,-1,2)) + noisePos * stdIm
posMap = np.zeros_like(posIgen)


# %% hago todos los mapeos de Monte Carlo
for posM, posI, pars in zip(posMap, posIgen, parsGen):
    r, t, k, d = sacoParmams(pars)
    posM[:,0], posM[:,1], _ = cl.inverse(posI, r, t, k, d, model)


# %% saco media y varianza de cada nube de puntos
posMapMean = np.mean(posMap, axis=0)
dif = (posMap - posMapMean).T
posMapVar = np.mean([dif[0] * dif, dif[1] * dif], axis=-1).T



# %% mapeo propagando incerteza los valores con los que comparar
xmJ, ymJ, CmJ = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j],                                        
                                     cameraMatrix, distCoeffs, model,
                                     Cccd, Cf, Ck, Crt)

XJ = np.array([xmJ, ymJ]).T

# %% grafico
xm, ym = posMap.reshape(-1,2).T

fig = plt.figure()
ax = fig.gca()

ax.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1],
            marker='+', c='k', s=100)

ax.scatter(xm, ym, marker='.', c='b', s=1)
cl.plotPointsUncert(ax, CmJ, xmJ, ymJ, 'k')

cl.plotPointsUncert(ax, posMapVar, posMapMean[:,0], posMapMean[:,1], 'b')

#plt.figure()
## ploteo el cociente entre varianzas
#quot = posMapVar[:, [0,1],[0,1]] / CmJ[:, [0,1],[0,1]]
#plt.plot(quot.T, 'x')



# %% TEST EACH STEP. STEP 1: CCD TO HOMOGENOUS

# mapeo propagando incerteza para tener con quien comparar
xpp, ypp, Cpp = cl.ccd2hom(imagePoints[j,0], cameraMatrix, Cccd, Cf)


# hago todos los mapeos de Monte Carlo
for posM, posI, pars in zip(posMap, posIgen, parsGen):
    r, t, k, d = sacoParmams(pars)
    posM[:,0], posM[:,1], _ = cl.ccd2hom(posI, k)


# saco media y varianza de cada nube de puntos
posMapMean = np.mean(posMap, axis=0)
dif = (posMap - posMapMean).T
posMapVar = np.mean([dif[0] * dif, dif[1] * dif], axis=-1).T

xm, ym = posMap.reshape(-1,2).T

fig = plt.figure()
ax = fig.gca()
ax.scatter(xm, ym, marker='.', c='b', s=1)
cl.plotPointsUncert(ax, Cpp, xpp, ypp, 'k')
cl.plotPointsUncert(ax, posMapVar, posMapMean[:,0], posMapMean[:,1], 'b')

posMapVar / Cpp
posMapMean[:,0] / xpp
posMapMean[:,1] / ypp

# %% TEST EACH STEP. STEP 2: HOMOGENOUS UNDISTORTION
# mapeo propagando incerteza para tener con quien comparar
xp, yp, Cp = cl.homDist2homUndist(xpp, ypp, distCoeffs, model, Cpp=Cpp, Ck=Ck)


# dejo los valores preparados
xPPgen = (xpp.reshape((-1,1)) + noisePos.T[0] * np.sqrt(Cpp[:,0,0]).reshape(-1,1)).T
yPPgen = (ypp.reshape((-1,1)) + noisePos.T[1] * np.sqrt(Cpp[:,1,1]).reshape(-1,1)).T
xP = np.zeros_like(xPPgen)
yP = np.zeros_like(yPPgen)

fig = plt.figure()
ax = fig.gca()
ax.plot(xPPgen, yPPgen, '.b')
cl.plotPointsUncert(ax, Cpp, xpp, ypp, 'k')


# hago todos los mapeos de Monte Carlo, una iteracion por sample
for i in range(nSampl):
    r, t, k, d = sacoParmams(parsGen[i])
    xP[i], yP[i], _ = cl.homDist2homUndist(xPPgen[i], yPPgen[i], d, model)


# saco media y varianza de cada nube de puntos
xPMean = np.mean(xP, axis=0)
yPMean = np.mean(yP, axis=0)
difX = xP - xPMean
difY = yP - yPMean

difCross = difX * difY
posPVar = np.mean([[difX**2, difCross], [difCross, difY**2]], axis=2).T


fig = plt.figure()
ax = fig.gca()
ax.scatter(xP, yP, marker='.', c='b', s=1)
cl.plotPointsUncert(ax, Cp, xp, yp, 'k')
cl.plotPointsUncert(ax, posPVar, xPMean, yPMean, 'b')

posPVar / Cp
xPMean / xp
yPMean / yp

# %% TEST EACH STEP. STEP 3: PROJECT TO MAP, UNDO ROTOTRASLATION
# mapeo propagando incerteza para tener con quien comparar
xm, ym, Cm= cl.xypToZplane(xp, yp, rVecs[j], tVecs[j], Cp=Cp, Crt=Crt)

# dejo los valores preparados
xPgen = (xp.reshape((-1,1)) + noisePos.T[0] * np.sqrt(Cp[:,0,0]).reshape(-1,1)).T
yPgen = (yp.reshape((-1,1)) + noisePos.T[1] * np.sqrt(Cp[:,1,1]).reshape(-1,1)).T
xM = np.zeros_like(xPgen)
yM = np.zeros_like(yPgen)

fig = plt.figure()
ax = fig.gca()
ax.plot(xPgen, yPgen, '.b', markersize=0.3)
cl.plotPointsUncert(ax, Cp, xp, yp, 'k')


# hago todos los mapeos de Monte Carlo, una iteracion por sample
for i in range(nSampl):
    r, t, k, d = sacoParmams(parsGen[i])
    xM[i], yM[i], _ = cl.xypToZplane(xPgen[i], yPgen[i], r, t)

# saco media y varianza de cada nube de puntos
xMMean = np.mean(xM, axis=0)
yMMean = np.mean(yM, axis=0)
difX = xM - xMMean
difY = yM - yMMean

difCross = difX * difY
posMVar = np.mean([[difX**2, difCross], [difCross, difY**2]], axis=2).T


fig = plt.figure()
ax = fig.gca()
ax.scatter(xM, yM, marker='.', c='b', s=1)
cl.plotPointsUncert(ax, Cm, xm, ym, 'k')
cl.plotPointsUncert(ax, posMVar, xMMean, yMMean, 'b')


