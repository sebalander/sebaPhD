#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:16:43 2017

hacer la calibracion de los datos tomados en nov 2016

@author: sebalander
"""


# %%
import cv2
from copy import deepcopy as dc
from calibration import calibrator as cl
from calibration import RationalCalibration as rational

import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from numpy import sqrt, array, isreal, roots
xypToZplane = cl.xypToZplane
formatParameters = rational.formatParameters
retrieveParameters = rational.retrieveParameters

from lmfit import minimize, Parameters


# %% LOAD DATA
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
model = 'rational'

# model files
modelFile = "./resources/intrinsicCalib/" + camera + "/"
distCoeffsFile =   modelFile + camera + model + "DistCoeffs.npy"
cameraMatrixFile = modelFile + camera + model + "LinearCoeffs.npy"
imgShapeFile =     modelFile + camera + "Shape.npy"

# load data
cameraMatrix = np.load(cameraMatrixFile) # coef intrinsecos
distCoeffs = np.load(distCoeffsFile)
imgShape = np.load(imgShapeFile)

# data files
dataFile = './resources/nov16/'
imgFile = dataFile + 'vcaSnapShot.png'
dawCalibTxt = dataFile + 'puntosCalibracion.txt'

# initil pose
tVecIniFile = dataFile + 'tVecIni.npy'
rVecIniFile = dataFile + 'rVecIni.npy'

# %% load data
ptsCalib = np.loadtxt(dawCalibTxt)
img = cv2.imread(imgFile)

# corners in image must have shape (N,1,2)
imagePoints = ptsCalib[:, :2].reshape((-1,2))
# pongo longitud como X y latitud como Y
# points in 3D wolrd must have shape
objectPoints = np.concatenate((ptsCalib[:, 3:1:-1],
                               np.zeros((len(ptsCalib),1)) ),
                               axis=1).reshape((-1,3))

tVecIni = np.load(tVecIniFile)
rVecIni = np.load(rVecIniFile)
rVecIni = cv2.Rodrigues(rVecIni)[0]

imagePointsProjected = cl.direct(objectPoints, rVecIni, tVecIni,
                                 cameraMatrix, distCoeffs, model)

# chequear que caigan donde deben
cl.cornerComparison(img, imagePoints, imagePointsProjected)

objectPointsProj = cl.inverse(imagePoints, rVecIni, tVecIni, cameraMatrix, distCoeffs, model)

cl.fiducialComparison(rVecIni, tVecIni, objectPoints, objectPointsProj)



# %% calculo el error cuadratico

def Esq(imagePoints, objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model):
    objectPointsProj = cl.inverse(imagePoints, rVec, tVec, cameraMatrix,
                                  distCoeffs, model)
    
    e = objectPoints - objectPointsProj
    
    return np.sum(e**2)


# %% plotear la superficie a optimizar
rVec = dc(rVecIni).reshape(-1)
tVec = dc(tVecIni).reshape(-1)

# ancho de intervalos donde me interesa generar datos
a = np.pi / 20
h = tVecIni[2] / 50

N = 10**4  # cantidad de puntos
rNoise = (np.random.rand(N,3) - 0.5) * a  # ruidos
tNoise = (np.random.rand(N,3) - 0.5) * h
RVecs = rVec + rNoise  # coordenadas
TVecs = tVec + tNoise
EE = np.empty(N)


# calculo errores
for i in range(N):
    EE[i] = Esq(imagePoints, objectPoints, RVecs[i], TVecs[i], cameraMatrix, distCoeffs, model)

# all dat toghether
samples = np.concatenate((RVecs, TVecs, EE.reshape((-1,1))),axis=1)

import corner

# figure = corner.corner(samples, labels=['r1', 'r2', 'r3', 't1', 't2', 't3', 'E'])

we = np.exp(-samples[:,-1])
#we /= np.sum(we)

figure = corner.corner(samples[:,:-1], weights=we,
                       labels=['r1', 'r2', 'r3', 't1', 't2', 't3'])



# %% 
def gradE2(imagePoints, objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model):
    
    E0 = Esq(imagePoints, objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model)
    
    u = 1e-6
    DrVec = rVec*u # incrementos
    DtVec = tVec*u
    
    rV = [rVec]*3 + np.diag(DrVec) # aplico incrementos
    tV = [tVec]*3 + np.diag(DtVec)
    
    E1r = np.empty(3)
    E1t = np.empty(3)
    
    # calculo el error con cada uno de los incrementos
    for i in range(3):
        E1r[i] = Esq(imagePoints, objectPoints, rV[i], tVec,
                 cameraMatrix, distCoeffs, model)
        E1t[i] = Esq(imagePoints, objectPoints, rVec, tV[i],
                 cameraMatrix, distCoeffs, model)
    
    # retorno el gradiente numerico
    return (E1r - E0) / DrVec, (E1t - E0) / DtVec, E0


gradE2(imagePoints, objectPoints, rVecIni, tVecIni, cameraMatrix, distCoeffs, model)

# %% implementacion de gradiente descedente amano
rVec = dc(rVecIni)
tVec = dc(tVecIni)
N = 3 # cantidad de iteraciones
EE = []


# pongo limites para cuánto corregir cada vez
a = np.pi / 200
h = tVecIni[2] / 30
eta = 1e-5

for i in range(N):
    
    
    cl.fiducialComparison3D(rVec, tVec, objectPoints)
    
    imagePointsProjected = cl.direct(objectPoints, rVec, tVec,
                                     cameraMatrix, distCoeffs, model)
    # chequear que caigan donde deben
    cl.cornerComparison(img, imagePoints, imagePointsProjected)
    
    
    # calculo los gradientes de rotacion y traslacion
    gR, gT, E = gradE2(imagePoints, objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model)
    EE.append(E)
    
    print(gR, gT, E)
    
    # los pongo dentro del limite de lo razonable
    if np.any(a < gR):
        gR *= a / np.max(gR)
    if np.any(h < gT):
        gT *= h / np.max(gT)
    
    
    
    
    
    print(gR, gT, E)
    # aplico la corrección
    rVec -= eta * gR
    tVec -= eta * gT


plt.plot(EE,'-+')

#cl.fiducialComparison3D(rVec, tVec, objectPoints)

imagePointsProjected = cl.direct(objectPoints, rVec, tVec,
                                 cameraMatrix, distCoeffs, model)
# chequear que caigan donde deben
cl.cornerComparison(img, imagePoints, imagePointsProjected)


# %%
rVecOpt, tVecOpt, params = cl.calibrateInverse(objectPoints, imagePoints, rVecIni, tVecIni, cameraMatrix, distCoeffs, model)

rVecOpt = cv2.Rodrigues(rVecOpt)[0]


# %%
objectPointsProj = cl.inverse(imagePoints, rVecOpt, tVecOpt, cameraMatrix, distCoeffs, model)

cl.fiducialComparison(rVecOpt, tVecOpt, objectPoints, objectPointsProj)

