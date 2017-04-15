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

import corner


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
rVecIni = cv2.Rodrigues(rVecIni)[0].reshape(-1)

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


# %%
# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    '''
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    '''

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])



    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])

    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R



def rot2euler(R):
    '''
    takes rodrigues vector or rotation matrix and returns euler angles
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    '''
    if np.prod(R.shape)==3:
        R = cv2.Rodrigues(R)[0]

    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def anglePeriodPi(angs, axis):
    '''
    taken an axis of the lists of vectors of anglñes and takes it to a different
    interval
    '''

def rodrigues2euler(rV):
    '''
    converts a list of rodrigues vectors to euler angles
    '''

    euV = []

    for r in rV:
        mat = cv2.Rodrigues(r)[0]
        euV.append(rot2euler(mat))

    euV = np.array(euV)

    np.max(euV, axis=0) - np.min(euV, axis=0)

    return



# %% pl4otear la superficie a optimizar
rVec = dc(rVecIni).reshape(-1)
tVec = dc(tVecIni).reshape(-1)

# conviertto a coordenadas del mapa donde es más facil definir intervalos
rM = rot2euler(-rVec)
tM = - np.dot(cv2.Rodrigues(rVec)[0].T, tVec)


# ancho de intervalos donde me interesa generar datos
a = np.pi / 10
h = tM[2] / 5

N = 10**3  # cantidad de puntos
rNoise = (np.random.rand(N, 3) - 0.5) * a  # ruidos
tNoise = (np.random.rand(N, 3) - 0.5) * h
RVecsM = rM + rNoise  # coordenadas respecto al mapa, angulos de euler
TVecsM = tM + tNoise
EE = np.empty(N)

# coordenadas de la camara
RVecsC = np.empty_like(RVecsM)
TVecsC = np.empty_like(TVecsM)

# calculo errores
for i in range(N):
    # convierto a coordenadas de la camara
    r = eulerAnglesToRotationMatrix(RVecsM[i]).T
    t = - np.dot(r, TVecsM[i])
    EE[i] = Esq(imagePoints, objectPoints, r, t, cameraMatrix, distCoeffs, model)



# all data toghether
samples = np.concatenate((RVecsM, TVecsM),axis=1)
labls = ['r1', 'r2', 'r3', 't1', 't2', 't3']

#figure = corner.corner(samples, labels=labls)
we = 1/EE # np.exp(-EE / np.mean(EE))

figure = corner.corner(samples, weights=we, labels=labls)
#
#samples2 = np.concatenate((samples,EE.reshape((-1,1))),axis=1)
#labls2 = labls.append('E')
#figure = corner.corner(samples2, labels=labls2)
#
#for i in range(6):
#    plt.figure()
#    plt.scatter(samples[:,i],EE)

# %%
def gradE2(imagePoints, objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model):

    E0 = Esq(imagePoints, objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model)

    u = 1e-4
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

    # calculo los gradientes
    gR = (E1r - E0) / DrVec
    gT = (E1t - E0) / DtVec

    # retorno el gradiente numerico
    return gR, gT, E0


gR, gT, E = gradE2(imagePoints, objectPoints, rVecIni, tVecIni, cameraMatrix, distCoeffs, model)
gR, gT

# %% pongo el cero de coords mapa cenrca de los puntos de calibracion
rVecIniMap = -rVecIni
tVecIniMap = - np.dot(cv2.Rodrigues(rVecIniMap)[0], tVecIni)

# tomo el primero de los puntos como referencia, arbitrariamente
tMapOrig = objectPoints[0]
# me centro ahí
tVecIniMap -= tMapOrig
objectPointsOrig = objectPoints - tMapOrig

# vond iniciales con el cero corrido (la rotacion es la misma)
tVecIniOrig = - np.dot(cv2.Rodrigues(rVecIni)[0], tVecIniMap)

# %% implementacion de gradiente descedente amano
rVec = dc(rVecIni)
tVec = dc(tVecIniOrig)

alfa = 0.01
beta = 0.9
gamma = 1 - beta
N = 500 # cantidad de iteraciones

## pongo limites para cuánto corregir cada vez
#a = np.pi / 200
#h = tVecIni[2] / 30

gradRot, gradTra, Err2 = gradE2(imagePoints, objectPointsOrig, rVec, tVec,
                                cameraMatrix, distCoeffs, model)
# para ir guardando los valores intermedios
rV = [rVec]
tV = [tVec]

gR = [gR]
gT = [gT]
E = [Err2]
zR = [np.zeros_like(rVec)]
zT = [np.zeros_like(tVec)]


for i in range(N):
    #cl.fiducialComparison3D(rV[i], tV[i], objectPoints)
    #imagePointsProjected = cl.direct(objectPoints, rV[i], tV[i],
    #                                 cameraMatrix, distCoeffs, model)
    # chequear que caigan donde deben
    #cl.cornerComparison(img, imagePoints, imagePointsProjected)


    # calculo los gradientes de rotacion y traslacion
    gradRot, gradTra, Err2 = gradE2(imagePoints, objectPointsOrig, rV[i], tV[i],
                                    cameraMatrix, distCoeffs, model)
    gR.append(gradRot)
    gT.append(gradTra)
    E.append(Err2)

    # inercia
    zR.append(beta * zR[i] + gamma * gR[i+1] * 10**6)
    zT.append(beta * zT[i] + gamma * gT[i+1])
    # aplico la corrección
    rV.append(rV[i] - alfa * zR[i+1])
    tV.append(tV[i] - alfa * zT[i+1])

    # los pongo dentro del limite de lo razonable
    #if np.any(a < gR):
    #    gR *= a / np.max(gR)
    #if np.any(h < gT):
    #    gT *= h / np.max(gT)


rV = np.array(rV)
tV = np.array(tV)

plt.figure()
plt.plot(E,'-+')

#cl.fiducialComparison3D(rV[-1], tV[-1], objectPointsOrig)
imagePointsProjected = cl.direct(objectPointsOrig, rV[-1], tV[-1],
                                 cameraMatrix, distCoeffs, model)
# chequear que caigan donde deben
cl.cornerComparison(img, imagePoints, imagePointsProjected)



tVMap = - np.dot(cv2.Rodrigues(rVecIniMap)[0], tV.T)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(tV[:,0], tV[:,1], tV[:,2])
ax.scatter(objectPointsOrig[:,0], objectPointsOrig[:,1], objectPointsOrig[:,2])


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(tV[:,0], tV[:,1], tV[:,2])
ax.scatter(tV[0,0], tV[0,1], tV[0,2], 'ok')
ax.scatter(tV[-1,0], tV[-1,1], tV[-1,2], 'or')


#euV = rodrigues2euler(rV)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(rV[:,0], rV[:,1], rV[:,2])
ax.scatter(rV[0,0], rV[0,1], rV[0,2], 'ok')
ax.scatter(rV[-1,0], rV[-1,1], rV[-1,2], 'or')


# %%
rVecOpt, tVecOpt, params = cl.calibrateInverse(objectPoints, imagePoints, rVecIni, tVecIni, cameraMatrix, distCoeffs, model)

rVecOpt = cv2.Rodrigues(rVecOpt)[0]


# %%
objectPointsProj = cl.inverse(imagePoints, rVecOpt, tVecOpt, cameraMatrix, distCoeffs, model)

cl.fiducialComparison(rVecOpt, tVecOpt, objectPoints, objectPointsProj)

