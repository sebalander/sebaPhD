#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:35:03 2017

test functions that propagate uncertanty

@author: sebalander
"""
# %%
import time
import timeit

import numpy as np
import scipy.linalg as ln
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



# %% simplest test
j = 0

plt.figure()
plt.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1],
            marker='+', c='k', s=100)

for j in range(0,n,3):
    xm, ym, Cm = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j],                                        
                            cameraMatrix, distCoeffs, model)
    
    plt.scatter(xm, ym, marker='x', c='b')


# %%
ep = 0.01  # realtive standard deviation in parameters

# matriz de incerteza de deteccion en pixeles
Cccd = np.repeat([np.eye(2,2)*0.1**2],imagePoints[j,0].shape[0], axis=0)
Cf = np.diag(cameraMatrix[[0,1,0,1],[0,1,2,2]] * ep)**2
Ck = np.diag((distCoeffs.reshape(-1) * ep )**2)
Cr = np.diag((rVecs[j].reshape(-1) * ep )**2)
Ct = np.diag((tVecs[j].reshape(-1) * ep )**2)


Crt = [Cr, Ct]


# %%
xpp, ypp, Cpp = cl.ccd2hom(imagePoints[j,0], cameraMatrix, Cccd, Cf)


# %% undistort
xp, yp, Cp = cl.homDist2homUndist(xpp, ypp, distCoeffs, model, Cpp, Ck)


# %% project to plane z=0 from homogenous
xm, ym, Cm = cl.xypToZplane(xp, yp, rVecs[j], tVecs[j], Cp, Crt)

Caux = Cm
# %%
xm, ym, Cm = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j],                                        
                        cameraMatrix, distCoeffs, model,
                        Cccd, Cf, Ck, Crt)

fig = plt.figure()
ax = fig.gca()
ax.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1])
cl.plotPointsUncert(ax, Cm, xm, ym, 'k')



er = [xm, ym] - chessboardModel[0,:,:2].T

# %%
statement1 = '''
p1 = np.tensordot(er, Cm, axes=(0,1))[range(54),range(54)]
p2 = p1.dot(er)[range(54),range(54)]
'''

# %%
statement2 = '''
Er = np.empty_like(xp)
for i in range(len(xp)):
    Er[i] = er[:,i].dot(Cm[i]).dot(er[:,i])
'''

# %%
statement3 = '''
q1 = [np.sum(Cm[:,:,0]*er.T,1), np.sum(Cm[:,:,1]*er.T,1)];
q2 = np.sum(q1*er,0)
'''
# %%

t1 = timeit.timeit(statement1, globals=globals(), number=10000) / 1e4

t2 = timeit.timeit(statement2, globals=globals(), number=10000) / 1e4

t3 = timeit.timeit(statement3, globals=globals(), number=10000) / 1e4

print(t1/t3, t2/t3)


# %%
ep = 0.001  # relative standard deviation in parameters

# matriz de incerteza de deteccion en pixeles
Cccd = np.repeat([np.eye(2,2)*0.1**2],imagePoints[j,0].shape[0], axis=0)
Cf = np.diag(cameraMatrix[[0,1,0,1],[0,1,2,2]] * ep)**2
Ck = np.diag((distCoeffs.reshape(-1) * ep )**2)


fig = plt.figure()
ax = fig.gca()

ax.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1],
            marker='+', c='k', s=100)

for j in range(0,n,3):
    Cr = np.diag((rVecs[j].reshape(-1) * ep )**2)
    Ct = np.diag((tVecs[j].reshape(-1) * ep )**2)
    
    Crt = [Cr, Ct]
    
    xm, ym, Cm = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j],                                        
                                         cameraMatrix, distCoeffs, model,
                                         Cccd, Cf, Ck, Crt)
    
    cl.plotPointsUncert(ax, Cm, xm, ym, 'k')



# %% testeo poniendo ruido en la imagen
j = 0 # elijo una imagen para trabajar
nSampl = 100  # cantidad de samples

ep = 0.001  # relative standard deviation in parameters

# desviaciones estándar de cada cosa, es facil porque son independientes
sI = np.ones(2) * 0.1
sF = cameraMatrix[[0,1,0,1],[0,1,2,2]] * ep
sK = np.abs(distCoeffs.reshape(-1) * ep)
sR = np.abs(rVecs[j].reshape(-1) * ep)
sT = np.abs(tVecs[j].reshape(-1) * ep)


# genero todo el ruido
noise = np.random.randn(15+distCoeffs.shape[0]+imagePoints[j,0].shape[0],
                        nSampl)

# dejo los samples preparados

rSam = rVecs[j] + sR * noise[0:3]


plt.figure()
plt.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1],
            marker='+', c='k', s=100)


    xm, ym, Cm = cl.inverse(imagePoints[j,0], , tVecs[j],                                        
                            cameraMatrix, distCoeffs, model)
    
    plt.scatter(xm, ym, marker='x', c='b')










