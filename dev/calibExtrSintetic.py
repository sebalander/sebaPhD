#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:49:00 2017

probar calibrar con punto sinteticos

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


# %% calculo el error cuadratico

def Esq(imagePoints, objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model):
    
    objectPointsProj = cl.inverse(imagePoints, rVec, tVec, cameraMatrix,
                                  distCoeffs, model)

    e = objectPoints - objectPointsProj

    return np.sum(e**2)

0 # Esq(imagePoints, objectPoints, rVecIni, tVecIni, cameraMatrix, distCoeffs, model)

# %%
def gradE2(imagePoints, objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model):

    E0 = Esq(imagePoints, objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model)

    u = 1e-5
    uR = np.pi * u
    uT = -cv2.Rodrigues(rVec)[0][:,2].dot(tVec) * u # tomo la altura como referencia

    rV = [rVec] * 3 + np.eye(3,dtype=float) * uR  # aplico incrementos
    tV = [tVec] * 3 + np.eye(3,dtype=float) * uT

    E1r = np.empty(3)
    E1t = np.empty(3)

    # calculo el error con cada uno de los incrementos
    for i in range(3):
        E1r[i] = Esq(imagePoints, objectPoints, rV[i], tVec,
                 cameraMatrix, distCoeffs, model)
        E1t[i] = Esq(imagePoints, objectPoints, rVec, tV[i],
                 cameraMatrix, distCoeffs, model)

    # calculo los gradientes
    gR = (E1r - E0) / uR
    gT = (E1t - E0) / uT

    # retorno el gradiente numerico
    return gR, gT, E0

0 # gradE2(imagePoints, objectPoints, rVecIni, tVecIni, cameraMatrix, distCoeffs, model)

#rVec, tVec = rVecIni, tVecIni


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

    rV = np.array(rVlis)
    tV = np.array(tVlis)
    gR = np.array(gRlis)
    gT = np.array(gTlis)

    return rV, tV, Elis, gR, gT

0


# %% ploteo cositas

def ploteoCositasOptimizacion(rV, tV, Elis, data):
    
    imagePoints, objectPoints, cameraMatrix, model = data
    
    plt.figure()
    plt.plot(Elis,'-+')
    
    # ploteo condicion inicial
    objectPointsProj = cl.inverse(imagePoints, rV[0], tV[0], cameraMatrix, distCoeffs, model)
    cl.fiducialComparison(rV[0], tV[0], objectPoints, objectPointsProj)
    
    # ploteo condicion final
    objectPointsProj = cl.inverse(imagePoints, rV[-1], tV[-1], cameraMatrix, distCoeffs, model)
    cl.fiducialComparison(rV[-1], tV[-1], objectPoints, objectPointsProj)
    
    # convierto tVec a marco de ref del mapa
    tVm = np.empty_like(tV)
    for i in range(len(tV)):
        tVm[i] = - cv2.Rodrigues(-rV[i])[0].dot(tV[i])
    
    
    #posicion de la camara en marco ref del mapa
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(tVm[:,0], tVm[:,1], tVm[:,2])
    ax.scatter(tVm[0,0], tVm[0,1], tVm[0,2], 'ok')
    ax.scatter(tVm[-1,0], tVm[-1,1], tVm[-1,2], 'or')
    ax.scatter(objectPoints[:,0], objectPoints[:,1], objectPoints[:,2])
    
    # posicion de la camara
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(tV[:,0], tV[:,1], tV[:,2], 'b-+')
    ax.scatter(tV[0,0], tV[0,1], tV[0,2], 'ok')
    ax.scatter(tV[-1,0], tV[-1,1], tV[-1,2], 'or')
    
    # vector de rodrigues
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(rV[:,0], rV[:,1], rV[:,2], 'b-+')
    ax.scatter(rV[0,0], rV[0,1], rV[0,2], 'ok')
    ax.scatter(rV[-1,0], rV[-1,1], rV[-1,2], 'or')


0


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

# initil pose
#tVecIniFile = dataFile + 'tVecIni.npy'
#rVecIniFile = dataFile + 'rVecIni.npy'
#
## load data
#tVecIni = np.array([0, 0, 15.0])
#Rini = np.load(rVecIniFile)
#rVecIni = cv2.Rodrigues(Rini)[0].reshape(-1)
tVecIni = np.array([0, 0, 15.0])
Rini = np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)
rVecIni = cv2.Rodrigues(Rini)[0]


# %% genero puntos en el plano y los proyecto a la imagen
gridAx = np.linspace(-100,100,40)
objectPoints = np.meshgrid(gridAx, gridAx)
objectPoints = np.array(objectPoints).reshape(2,-1).T
# agrego ceros coord z
objectPoints = np.concatenate((objectPoints, np.zeros([len(objectPoints),1])), axis=1)

# proyecto a la imagen
imagePoints = cl.direct(objectPoints, rVecIni, tVecIni, cameraMatrix, distCoeffs, model)
imagePoints.shape = (-1,2)

#objectPointsProj = cl.inverse(imagePoints, rVecIni, tVecIni, cameraMatrix, distCoeffs, model)

# grafico desde el marco de referencia de la camara:
objectPointsC = np.dot(Rini, objectPoints.T) + tVecIni.reshape(3,1)
l = np.linalg.norm(objectPointsC, axis=0)
s = np.mean(l)/4

# saco lo puntos que estan demasiado cerca del horizonte
# tomo ~15 grados de margen
fi = np.arccos(objectPointsC[2] / l)
dejar = fi < np.pi * (0.49)

objectPoints = objectPoints[dejar]
imagePoints = imagePoints[dejar]
objectPointsC = objectPointsC.T[dejar]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.axis('equal')
ax.scatter(objectPointsC[:,0], objectPointsC[:,1], objectPointsC[:,2])
ax.plot([0, s], [0, 0], [0, 0], "-r")
ax.plot([0, 0], [0, s], [0, 0], "-b")
ax.plot([0, 0], [0, 0], [0, s], "-k")

plt.figure()
plt.axes(aspect='equal')
plt.scatter(imagePoints[:,0], imagePoints[:,1])
plt.xlim([0, cameraMatrix[0,2]*2])
plt.ylim([0, cameraMatrix[1,2]*2])

# %% implementacion de gradiente descedente con momentum
rVec = np.array([3, 0, 0], dtype=float)  # dc(rVecIni)
tVec = np.array([0, 3, 15], dtype=float)  # dc(tVecIni)

data = [imagePoints, objectPoints, cameraMatrix, model]

## proyecto a la imagen
#imagePoints = cl.direct(objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model)
#imagePoints.shape = (-1,2)
#
#plt.figure()
#plt.scatter(imagePoints[:,0], imagePoints[:,1])
#
## proyecto a la imagen
#imagePoints = cl.direct(objectPoints, rVecIni, tVecIni, cameraMatrix, distCoeffs, model)
#imagePoints.shape = (-1,2)
#plt.scatter(imagePoints[:,0], imagePoints[:,1])


#objectPointsProj = cl.inverse(imagePoints, rVec, tVec, cameraMatrix, distCoeffs, model)9
#cl.fiducialComparison(rVec, tVec, objectPointsOrig, objectPointsProj)

# alfas, inercia t cant de iteracioens
alfaR, alfaT, beta, N = 1e-7, 1e-4, 0.5, 100
rV, tV, Elis, _, _ = graDescMome(alfaR, alfaT, beta, N, data, rVec, tVec, data)

## alfas, inercia t cant de iteracioens
#alfaR, alfaT, beta, N = 0.0, 1e-4, 0.8, 10
#rV2, tV2, Elis2, _, _ = graDescMome(alfaR, alfaT, beta, N, data, rV[-1], tV[-1], data)
#
#rV = np.vstack((rV, rV2))
#tV = np.vstack((tV, tV2))
#Elis.extend(Elis2)
#
## alfas, inercia t cant de iteracioens
#alfaR, alfaT, beta, N = 1e-7, 1e-4, 0.99, 20
#rV2, tV2, Elis2, _, _ = graDescMome(alfaR, alfaT, beta, N, data, rV[-1], tV[-1], data)
#
#rV = np.vstack((rV, rV2))
#tV = np.vstack((tV, tV2))
#Elis.extend(Elis2)

ploteoCositasOptimizacion(rV, tV, Elis, data)

# %%
E = np.array(Elis)
dE = E[1:] - E[:-1]

plt.plot(dE/E[1:])

plt.subplot(311)
plt.plot(E)
plt.semilogy()

plt.subplot(312)
for t in tV.T:
    plt.plot(t - t[-1])

plt.subplot(313)
for r in rV.T:
    plt.plot(r - r[-1])

plt.tight_layout()

## %% testeo con muchas condiciones iniciales
#rVec = dc(rVecIniOrig)
#tVec = dc(tVecIniOrig)
#
#n = 10  # cantidad de subdivisiones por dimension
#deltaR = np.pi / 20  # medio ancho del intervalo de angulos
#deltaT = 5  # medio ancho del intervalo de posiciones (en metros en este caso)
#
#dR = deltaR / n
#dT = deltaT / n
#
#undostres = np.arange(n).reshape(-1,1)
#
#angles = ((rVec - deltaR) + dR * undostres).reshape(-1,1,3)
#posici = ((tVec - deltaT) + dT * undostres).reshape(-1,1,3)
#
#condIni = np.concatenate((angles, posici), axis=1)
#
#for cond in condIni:
#    rVec, tVec = cond
#    
#    print(rVec, tVec)
#
#0


# %% hago la prueba con el metodo lineal
# aca no existe directa o inversa
x1, y1 = objectPoints.T[:2]
imagePoints, objectPoints





