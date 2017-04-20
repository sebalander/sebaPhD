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

# import corner


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


## saco los pun to mas lejanos
#dejar = objectPoints[:,0] <  -58.37
#
#objectPointsProj = cl.inverse(imagePoints[dejar], rVecIni, tVecIni, cameraMatrix, distCoeffs, model)
#cl.fiducialComparison(rVecIni, tVecIni, objectPoints[dejar], objectPointsProj)
#
#objectPoints = objectPoints[dejar]
#imagePoints = imagePoints[dejar]


# %% calculo el error cuadratico

def Esq(imagePoints, objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model):
    
    objectPointsProj = cl.inverse(imagePoints, rVec, tVec, cameraMatrix,
                                  distCoeffs, model)

    e = objectPoints - objectPointsProj

    return np.sum(e**2)

Esq(imagePoints, objectPoints, rVecIni, tVecIni, cameraMatrix, distCoeffs, model)

# %%
def gradE2(imagePoints, objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model):

    E0 = Esq(imagePoints, objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model)

    u = 1e-5
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

# %% 
def graDescMome(alfaR, alfaT, beta, N, Data, rVec, tVec, data):
    '''
    hace gradiente descendente con los parametros provistos y cond iniciales
    data = [imagePoints, objectPoints, cameraMatrix, model]
    '''
    imagePoints, objectPoints, cameraMatrix, model = data
    
    gamma = 1 - beta
    
    # para ir guardando los valores intermedios
    rVlis = [rVec]
    tVlis = [tVec]
    
    gradRot, gradTra, Err2 = gradE2(imagePoints, objectPoints, rVec, tVec,
                                    cameraMatrix, distCoeffs, model)
    gRlis = [gradRot]
    gTlis = [gradTra]
    Elis = [Err2]
    
    zRlis = [np.zeros_like(rVec)]
    zTlis = [np.zeros_like(tVec)]
    
    for i in range(N):
        #cl.fiducialComparison3D(rV[i], tV[i], objectPoints)
        #imagePointsProjected = cl.direct(objectPoints, rV[i], tV[i],
        #                                 cameraMatrix, distCoeffs, model)
        # chequear que caigan donde deben
        #cl.cornerComparison(img, imagePoints, imagePointsProjected)
    
    
        # calculo los gradientes de rotacion y traslacion
        gradRot, gradTra, Err2 = gradE2(imagePoints, objectPoints, rVlis[-1], tVlis[-1],
                                        cameraMatrix, distCoeffs, model)
        gRlis.append(gradRot)
        gTlis.append(gradTra)
        Elis.append(Err2)
    
        # inercia en el gradiente
        zRlis.append(beta * zRlis[-1] + gamma * gRlis[-1])
        zTlis.append(beta * zTlis[-1] + gamma * gTlis[-1])
        # aplico la corrección
        rVlis.append(rVlis[-1] - alfaR * zRlis[-1])
        tVlis.append(tVlis[-1] - alfaT * zTlis[-1])

    return rVlis, tVlis, Elis, gRlis, gTlis

0

# %% pongo el cero de coords mapa cenrca de los puntos de calibracion
# params de rototraslacion en marco ref del mapa:
rVecIniMap = - rVecIni
tVecIniMap = - np.dot(cv2.Rodrigues(rVecIniMap)[0], tVecIni)

# tomo el promedio de los puntos como referencia, arbitrariamente
tMapOrig = np.mean(objectPoints,axis=0)
# me centro ahí
tVecIniMap -= tMapOrig
objectPointsOrig = objectPoints - tMapOrig

## reescaleo para que la altura de la camara mida pi
#k = np.pi / tVecIniMap[2]
## reescaleo para que la desv estandar de los puntos sea pi
#k = np.pi / np.std(objectPointsOrig)
# escala de posicion en metros
k = 62.07 / np.linalg.norm([-58.370731 + 58.370678, -34.629440 + 34.628883])

tVecIniMap *= k
objectPointsOrig *= k

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.axis('equal')
#fig.gca().set_aspect('equal', adjustable='box')
ax.scatter(objectPointsOrig[:,0],
           objectPointsOrig[:,1],
           objectPointsOrig[:,2])
ax.scatter(tVecIniMap[0], tVecIniMap[1], tVecIniMap[2])


# cond iniciales con el cero corrido (la rotacion es la misma) en marco ref de
# la camara
rVecIniOrig = rVecIni
tVecIniOrig = - np.dot(cv2.Rodrigues(rVecIni)[0], tVecIniMap)


plt.figure()
plt.scatter(objectPointsOrig[:,0], objectPointsOrig[:,1])
plt.scatter(tVecIniMap[0], tVecIniMap[1])


# grafico los versores de la camara para ver cooq euda la rotacion
# convierto a marco de ref de la camara para ver que aeste todo bien
tx, ty, tz = tVecIniOrig
x, y, z = cv2.Rodrigues(tVecIniOrig)[0].T + tVecIniOrig

rotMat = cv2.Rodrigues(rVecIniOrig)[0]
objectPointsProjOrigCam = np.dot(rotMat, objectPointsOrig.T).T
objectPointsProjOrigCam += tVecIniOrig

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.axis('equal')
ax.plot([tx, x[0]], [ty, x[1]], [tz, x[2]], "-r")
ax.plot([tx, y[0]], [ty, y[1]], [tz, y[2]], "-b")
ax.plot([tx, z[0]], [ty, z[1]], [tz, z[2]], "-k")
ax.scatter(objectPointsProjOrigCam[:,0],
           objectPointsProjOrigCam[:,1],
           objectPointsProjOrigCam[:,2])
ax.plot([0, 1], [0, 0], [0, 0], "-r")
ax.plot([0, 0], [0, 1], [0, 0], "-b")
ax.plot([0, 0], [0, 0], [0, 1], "-k")


# %% implementacion de gradiente descedente con momentum
rVec = dc(rVecIniOrig)
tVec = dc(tVecIniOrig)

#objectPointsProj = cl.inverse(imagePoints, rVec, tVec, cameraMatrix, distCoeffs, model)
#cl.fiducialComparison(rVec, tVec, objectPointsOrig, objectPointsProj)


data = [imagePoints, objectPointsOrig, cameraMatrix, model]

alfaR = 1e-6
alfaT = 1e-3
beta = 0.95
N = 100 # cantidad de iteraciones

rVlis, tVlis, Elis, gRlis, gTlis = graDescMome(alfaR, alfaT, beta, N, data, rVec, tVec, data)


rV = np.array(rVlis)
tV = np.array(tVlis)
gR = np.array(gRlis)
gT = np.array(gTlis)

plt.figure()
#plt.set_title('Error cuadrático')
plt.plot(Elis,'-+')
#plt.show()

#plt.figure()
#plt.plot(np.linalg.norm(gR, axis=1))
#plt.plot(np.linalg.norm(gT, axis=1))


objectPointsProj = cl.inverse(imagePoints, rV[0], tV[0], cameraMatrix, distCoeffs, model)
cl.fiducialComparison(rV[0], tV[0], objectPointsOrig, objectPointsProj)


objectPointsProj = cl.inverse(imagePoints, rV[-1], tV[-1], cameraMatrix, distCoeffs, model)
cl.fiducialComparison(rV[-1], tV[-1], objectPointsOrig, objectPointsProj)


# levanto la camara 3 metros
tVMap = - np.dot(cv2.Rodrigues(-rV[-1])[0], tV[-1])
tVMap[2] = 19
t2 = - np.dot(cv2.Rodrigues(rV[-1])[0], tVMap)

Esq(imagePoints, objectPoints, rV[-1], tV[-1], cameraMatrix, distCoeffs, model)
Esq(imagePoints, objectPoints, rV[-1], t2, cameraMatrix, distCoeffs, model)

objectPointsProj = cl.inverse(imagePoints, rV[-1], t2, cameraMatrix, distCoeffs, model)
cl.fiducialComparison(rV[-1], t2, objectPointsOrig, objectPointsProj)


fig = plt.figure()
#ax.set_title('posicion de la camara')
ax = fig.gca(projection='3d')
ax.plot(tV[:,0], tV[:,1], tV[:,2])
ax.scatter(objectPointsOrig[:,0], objectPointsOrig[:,1], objectPointsOrig[:,2])
#fig.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_title('posicion de la camara')
ax.plot(tV[:,0], tV[:,1], tV[:,2], 'b-+')
ax.scatter(tV[0,0], tV[0,1], tV[0,2], 'ok')
ax.scatter(tV[-1,0], tV[-1,1], tV[-1,2], 'or')
#fig.show()


#euV = rodrigues2euler(rV)
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_title('vector de rodrigues')
ax.plot(rV[:,0], rV[:,1], rV[:,2], 'b-+')
ax.scatter(rV[0,0], rV[0,1], rV[0,2], 'ok')
ax.scatter(rV[-1,0], rV[-1,1], rV[-1,2], 'or')
#fig.show()

## %%
#rVecOpt, tVecOpt, params = cl.calibrateInverse(objectPoints, imagePoints, rVecIni, tVecIni, cameraMatrix, distCoeffs, model)
#
#rVecOpt = cv2.Rodrigues(rVecOpt)[0]
#
#
## %%
#objectPointsProj = cl.inverse(imagePoints, rVecOpt, tVecOpt, cameraMatrix, distCoeffs, model)
#
#cl.fiducialComparison(rVecOpt, tVecOpt, objectPoints, objectPointsProj)

# %% testeo con muchas condiciones iniciales
rVec = dc(rVecIniOrig)
tVec = dc(tVecIniOrig)

n = 10  # cantidad de subdivisiones por dimension
deltaR = np.pi / 20  # medio ancho del intervalo de angulos
deltaT = 5  # medio ancho del intervalo de posiciones (en metros en este caso)

dR = deltaR / n
dT = deltaT / n

undostres = np.arange(n).reshape(-1,1)

angles = ((rVec - deltaR) + dR * undostres).reshape(-1,1,3)
posici = ((tVec - deltaT) + dT * undostres).reshape(-1,1,3)

condIni = np.concatenate((angles, posici), axis=1)

for cond in condIni:
    rVec, tVec = cond
    
    print(rVec, tVec)

0







