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
import scipy.linalg as ln
from threading import Thread

# %% LOAD DATA
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
model = 'rational'
sacarLejanos = True  # excluir puntos lejanos del analisis

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
dawCalibTxt = dataFile + 'puntosCalibracion.txt'  # file with corners and map gps
#dataFile = '/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/calibrExtr/'
#imgFile = dataFile + 'testPoseInChess.png'
#dawCalibTxt = dataFile + 'puntosChessTest.txt'  # file with corners and map gps

## initil pose # ya no se necesita
#tVecIniFile = dataFile + 'tVecIni.npy'
#rVecIniFile = dataFile + 'rVecIni.npy'

# %% load data
ptsCalib = np.loadtxt(dawCalibTxt)
img = plt.imread(imgFile)

# corners in image must have shape (N,1,2)
imagePoints = ptsCalib[:, :2].reshape((-1,2))
# pongo longitud como X y latitud como Y
# points in 3D wolrd must have shape
objectPoints = np.concatenate((ptsCalib[:, 3:1:-1],
                               np.zeros((len(ptsCalib),1)) ),
                               axis=1).reshape((-1,3))

# escala de posicion en metros
cen = np.mean(objectPoints, axis=0)
k = 62.07 / np.linalg.norm([-58.370731 + 58.370678, -34.629440 + 34.628883])

objectPoints = objectPoints - cen
objectPoints *= k


# filtrar los que estan lejos
if sacarLejanos:
    dejar = objectPoints[:,0] < 40
    
    objectPoints = objectPoints[dejar]
    imagePoints = imagePoints[dejar]


## %% cargo condiciones iniciales de pose (no es necesario ya)
#tVecIni = np.load(tVecIniFile)
#rVecIni = np.load(rVecIniFile)
#rVecIni = cv2.Rodrigues(rVecIni)[0].reshape(-1)
#
## cambio la T de la camara
#t = - cl.rotateRodrigues(tVecIni, -rVecIni) # voy a marco de ref mapa
#t = t -cen # cambio en origen de coords
#tVecIni = - cl.rotateRodrigues(t, rVecIni) # vuelvo a marco ref camara
#tVecIni *= k
#
#imagePointsProjected = cl.direct(objectPoints, rVecIni, tVecIni,
#                                 cameraMatrix, distCoeffs, model)
#
## chequear que caigan donde deben
#cl.cornerComparison(img, imagePoints, imagePointsProjected)
#
#objectPointsProj = cl.inverse(imagePoints, rVecIni, tVecIni,
#                              cameraMatrix, distCoeffs, model)
#cl.fiducialComparison(rVecIni, tVecIni, objectPoints, objectPointsProj)
#
##plt.scatter(objectPoints[:,0], objectPoints[:,1])
##plt.scatter(imagePointsProjected[:,0], imagePointsProjected[:,1])


# %% funciones para resolver linealmente
def dataMatrixPoseCalib(xm, ym, xp, yp):
    '''
    return data matrix for linear calibration
    input: object points un z=0 plane and homogenous undistorted coords
    '''
    ons = np.ones_like(xm)
    zer = np.zeros_like(xm)
    
    A1 = np.array([xm, zer, -xp*xm, ym, zer, -xp*ym, ons,  zer, -xp])
    A2 = np.array([zer, xm, -yp*xm, zer, ym, -yp*ym,  zer, ons, -yp])
    
    # tal que A*m = 0
    A = np.concatenate((A1,A2), axis=1).T
    
    return A


def poseLinearCalibration(objectPoints, imagePoints, cameraMatrix, distCoeffs, model, retMatrix=False):
    '''
    takes calibration points and estimate linearly camera pose. re
    '''
    # map coordinates with z=0
    xm, ym = objectPoints.T[:2]
    # undistort ccd points, x,y homogenous undistorted
    xp, yp = cl.ccd2homUndistorted(imagePoints, cameraMatrix, distCoeffs, model)
    
    A = dataMatrixPoseCalib(xm, ym, xp, yp)
    
    _, s, v = ln.svd(A)
    m = v[-1] # select right singular vector of smaller singular value
    
    # normalize and ensure that points are in front of the camera
    m /= np.sqrt(ln.norm(m[:3])*ln.norm(m[3:6])) * np.sign(m[-1])
    
    # rearrange as rVec, tVec
    R = np.array([m[:3], m[3:6], np.cross(m[:3], m[3:6])]).T
    rVec = cv2.Rodrigues(R)[0]
    tVec = m[6:]
    
    if retMatrix:
        return rVec, tVec, A
    
    return rVec, tVec

#

# %% apply linear calibration scheme
imagePoints.shape, objectPoints.shape

rV, tV, A = poseLinearCalibration(objectPoints, imagePoints, cameraMatrix, distCoeffs, model, True)


# %%
def projectionPlots(rV, tV, data):
    '''
    compara los puntos de calibracio y sus proyecciones en las tres etapas
    '''
    imagePoints, objectPoints, cameraMatrix, model = data
    
    imagePointsProj = cl.direct(objectPoints, rV, tV, cameraMatrix, distCoeffs, model)
    objectPointsProj = cl.inverse(imagePoints, rV, tV, cameraMatrix, distCoeffs, model)
    xc, yc, zc = cl.rotoTrasRodri(objectPoints,rV,tV).T
    
    xp, yp = cl.ccd2homUndistorted(imagePoints, cameraMatrix,  distCoeffs, model)
    xp2, yp2 = cl.rotoTrasHomog(objectPoints, rV, tV).T
    
    ll = np.linalg.norm(objectPoints, axis=0)
    r = np.mean(ll) / 7
    
    plt.figure()
    plt.title('image Points')
    plt.imshow(img)
    plt.scatter(imagePoints[:,0], imagePoints[:,1], marker='+', label='calibration')
    plt.scatter(imagePointsProj[:,0], imagePointsProj[:,1], marker='x', label='projected')
    plt.legend()
    
    #plt.figure()
    #plt.title('homogenous Points')
    #plt.scatter(xp, yp, marker='+', label='undistorted from image')
    #plt.scatter(xp2, yp2, marker='x', label='rptotraslated from map')
    #plt.legend()
    
    plt.figure()
    plt.title('object Points')
    plt.scatter(objectPoints[:,0], objectPoints[:,1], marker='+', label='calibration')
    plt.scatter(objectPointsProj[:,0], objectPointsProj[:,1], marker='x', label='projected')
    plt.legend()
    
    
    #fig = plt.figure()
    #plt.title('3D object Points, camera ref frame')
    #ax = fig.gca(projection='3d')
    #ax.axis('equal')
    #ax.scatter(xc, yc, zc)
    #ax.plot([0, r], [0, 0], [0, 0], "-r")
    #ax.plot([0, 0], [0, r], [0, 0], "-b")
    #ax.plot([0, 0], [0, 0], [0, r], "-k")
    
    return imagePointsProj, objectPointsProj

data = [imagePoints, objectPoints, cameraMatrix, model]
imagePointsProj, objectPointsProj = projectionPlots(rV, tV, data)

# %% calculo el error cuadratico

def Esq(imagePoints, objectPoints, rVec, tVec, cameraMatrix, distCoeffs, model):
    
    objectPointsProj = cl.inverse(imagePoints, rVec, tVec, cameraMatrix,
                                  distCoeffs, model)

    e = objectPoints - objectPointsProj

    return np.sum(e**2)

Esq(imagePoints, objectPoints, rV, tV, cameraMatrix, distCoeffs, model)

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


gR, gT, E = gradE2(imagePoints, objectPoints, rV, tV, cameraMatrix, distCoeffs, model)
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

#

# %%

def unHilo(realizData, M, h, E, rVoptim, tVoptim):
    
    print('thread', h, 'inicializing')
    alfaR, alfaT, beta, N, data, rV, tV, data = realizData
    
    for m in range(M):
        print('thread', h, 'iteracion', m, 'de', M)
        r, t, Elis, _, _ = graDescMome(alfaR, alfaT, beta, N, data, rV[m], tV[m], data)
        
        E.append(Elis)
        rVoptim.append(r[-1])
        tVoptim.append(t[-1])
    
        del Elis, r, t
    
    print('thread', h, 'terminated')
#

# %%
data = [imagePoints, objectPoints, cameraMatrix, model]

alfaR = 1e-6
alfaT = 1e-6
beta = 0.8

N = 50 # cantidad de iteraciones
M = 50  # cantidad de realizaciones por hilo
H = 4  # cantidad de hilos
# magnitud de la variacion
Dr = np.ones_like(rV) * np.pi * 30.0/180 # 30 grados de amplitud de variacion angular
Dt = np.ones_like(tV) * 10 # 10 metros de variacion de posicion

# comparo con la optimizacion a partir de la cond lineal
_, _, Elis, _, _ = graDescMome(alfaR, alfaT, beta, N, data, rV, tV, data)


# %%
rVoptim = list()
tVoptim = list()
E = list()
threads = list()

# genero posiciones iniciales
rVins = rV + (np.random.rand(H, M, 3) - 0.5) * Dr
tVins = tV + (np.random.rand(H, M, 3) - 0.5) * Dt

for h in range(H):
    realizData = [alfaR, alfaT, beta, N, data, rVins[h], tVins[h], data]
    
    t = Thread(target=unHilo, args=(realizData, M, h, E, rVoptim, tVoptim))
    
    threads.append(t)
    threads[h].start()

[t.join() for t in threads] # wait for all

E = np.array(E)


plt.figure()
plt.plot(E.T)
plt.plot(Elis,'-k', lw=2)
#imagePointsProj, objectPointsProj = projectionPlots(rVlis[-1], tVlis[-1], data)

## %% busqueda de outliers
## resulevo con autovectores
#A2 = A ## - np.mean(A, axis=0)  # saco la media
#u, s, v = ln.svd(A)
#SS = A.T.dot(A)
#s2, v2 = ln.eig(SS, right=True)  # v[:,i] es autovector i esimo
#s2 = np.real(s2)
## ordeno
#arsor = np.argsort(s2)
#s2 = s2[arsor[::-1]]
#v2 = v2[:, arsor[::-1]]
## proyecto los vectores de un metodo sobre el otro para cuantificar la similaridad
#vSim = np.abs(np.diag(v.dot(v2)))
#
## %% comparo
#plt.subplot(211)
#plt.plot(np.sqrt(s2), '-xr', label=' sqrt(autovalores)')
#plt.plot(s, '-+b', label='vals singuls')
#plt.semilogy()
#plt.legend()
#
#plt.subplot(212)
#plt.ylabel('proyección de autovects y vec sing')
#plt.plot(vSim)
#plt.ylim(0,1.1)
#plt.xlabel('indice de valor')
#plt.tight_layout()
#plt.subplots_adjust(hspace=0)
#
#
## %% ploteo la nube, proyectada en sus autovectores para detectar outliers
#B = A2.dot(v2).T
#
#fig = plt.figure()
#plt.plot(A.T,'+k')
##plt.plot(np.sqrt(s2/A.shape[0]))
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(B[0], B[1], B[2])
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(B[3], B[4], B[5])
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(B[6], B[7], B[8])
#
## %% veo que posiblemente haya outliers que tienen en su tercera componente
## (de indice [2]) valores menores a -1e8 y en su tercera componente, [5], 
## valores menores a -0.8e8.
##filtro los que serian primeros outlier
## parece que tambien corresponden a los segundos
#dejar = A[:,2] > -2e8
#
#fig = plt.figure()
#plt.plot(A[dejar].T,'+k')
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(A[dejar,0], A[dejar,1], A[dejar,2])
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(A[dejar,3], A[dejar,4], A[dejar,5])
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(A[dejar,6], A[dejar,7], A[dejar,8])
#
#_, s4, v4 = ln.svd(A[dejar])
#
#
#
#
## %%
#for i in range(9):
#    vv = v4[i]
#    
#    print('singular value',s4[i])
#    
#    r1 = ln.norm(vv[:3])
#    r2 = ln.norm(vv[3:6])
#    print('modulos',r1, r2)
#    
#    print('angulo', np.rad2deg(np.arccos(np.dot(vv[:3],vv[3:6])/ r1 /r2)))
#    print(vv/np.sqrt(r1*r2))
#    print('\n')
#
#0
#
## %% ajusto una transformacion lineal cualquiera sin importar si hay una roto
## traslacion asociada (tiene que haber)
## xh = Ad xm + Bd # la directa
## xm = Ai xh + Bi # la inversa
## agregando unos:
## Xm(2xn) = Ad(2x3) * XmUnos(3xn)
## Xh(2xn) = Ai(2x3) * XhUnos(3xn)
#
## armo los vectores de datos
#Xm = objectPoints.T[:2] # queda de 2xn
#Xh = cl.ccd2homUndistorted(imagePoints, cameraMatrix,  distCoeffs, model)
#Xh = np.array(Xh) # 2xn
#
#ones = np.ones_like(Xm[0]).reshape(1,-1)
#XmUnos = np.concatenate((Xm,ones),axis=0) # 3xn
#XhUnos = np.concatenate((Xh,ones),axis=0) # 3xn
#
## ahora resuelvo las matrices de transformacion
#Ad = Xh.dot(ln.pinv(XmUnos))
#Ai = Xm.dot(ln.pinv(XhUnos))
#
## proyecto para ver que da y comparar
#XhProj = Ad.dot(XmUnos)
#XmProj = Ai.dot(XhUnos)
#
## grafico par comparar
#plt.subplot(121)
#plt.scatter(Xh[0], Xh[1], marker='+')
#plt.scatter(XhProj[0], XhProj[1], marker='x')
#
#plt.subplot(122)
#plt.scatter(Xm[0], Xm[1], marker='+')
#plt.scatter(XmProj[0], XmProj[1], marker='x')
#
#
#
#
#
#
#
## %%
#r = np.array([ 2.63459211,  1.20459456, -0.27162026])*1.000003
#t = np.array([ 63.98385874,  19.93397018, -10.72342661])
#
#imagePointsProjected = cl.direct(objectPoints, r, t, cameraMatrix, distCoeffs, model)
#
#plt.clf()
#plt.imshow(img)
#plt.scatter(imagePoints[:,0], imagePoints[:,1], marker='+')
#plt.scatter(imagePointsProjected[:,0], imagePointsProjected[:,1], marker='x')
#
#
#
## %% pongo el cero de coords mapa cenrca de los puntos de calibracion
## params de rototraslacion en marco ref del mapa:
#rVecIniMap = - rVecIni
#tVecIniMap = - np.dot(cv2.Rodrigues(rVecIniMap)[0], tVecIni)
#
## tomo el promedio de los puntos como referencia, arbitrariamente
#tMapOrig = np.mean(objectPoints,axis=0)
## me centro ahí
#tVecIniMap -= tMapOrig
#objectPointsOrig = objectPoints - tMapOrig
#
### reescaleo para que la altura de la camara mida pi
##k = np.pi / tVecIniMap[2]
### reescaleo para que la desv estandar de los puntos sea pi
##k = np.pi / np.std(objectPointsOrig)
## escala de posicion en metros
#k = 62.07 / np.linalg.norm([-58.370731 + 58.370678, -34.629440 + 34.628883])
#
#tVecIniMap *= k
#objectPointsOrig *= k
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.axis('equal')
##fig.gca().set_aspect('equal', adjustable='box')
#ax.scatter(objectPointsOrig[:,0],
#           objectPointsOrig[:,1],
#           objectPointsOrig[:,2])
#ax.scatter(tVecIniMap[0], tVecIniMap[1], tVecIniMap[2])
#
#
## cond iniciales con el cero corrido (la rotacion es la misma) en marco ref de
## la camara
#rVecIniOrig = rVecIni
#tVecIniOrig = - np.dot(cv2.Rodrigues(rVecIni)[0], tVecIniMap)
#
#
#plt.figure()
#plt.scatter(objectPointsOrig[:,0], objectPointsOrig[:,1])
#plt.scatter(tVecIniMap[0], tVecIniMap[1])
#
#
## grafico los versores de la camara para ver cooq euda la rotacion
## convierto a marco de ref de la camara para ver que aeste todo bien
#tx, ty, tz = tVecIniOrig
#x, y, z = cv2.Rodrigues(tVecIniOrig)[0].T + tVecIniOrig
#
#rotMat = cv2.Rodrigues(rVecIniOrig)[0]
#objectPointsProjOrigCam = np.dot(rotMat, objectPointsOrig.T).T
#objectPointsProjOrigCam += tVecIniOrig
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.axis('equal')
#ax.plot([tx, x[0]], [ty, x[1]], [tz, x[2]], "-r")
#ax.plot([tx, y[0]], [ty, y[1]], [tz, y[2]], "-b")
#ax.plot([tx, z[0]], [ty, z[1]], [tz, z[2]], "-k")
#ax.scatter(objectPointsProjOrigCam[:,0],
#           objectPointsProjOrigCam[:,1],
#           objectPointsProjOrigCam[:,2])
#ax.plot([0, 1], [0, 0], [0, 0], "-r")
#ax.plot([0, 0], [0, 1], [0, 0], "-b")
#ax.plot([0, 0], [0, 0], [0, 1], "-k")
#
#
## %% implementacion de gradiente descedente con momentum
#rVec = dc(rVecIniOrig)
#tVec = dc(tVecIniOrig)
#
##objectPointsProj = cl.inverse(imagePoints, rVec, tVec, cameraMatrix, distCoeffs, model)
##cl.fiducialComparison(rVec, tVec, objectPointsOrig, objectPointsProj)
#
#
#data = [imagePoints, objectPointsOrig, cameraMatrix, model]
#
#alfaR = 1e-6
#alfaT = 1e-3
#beta = 0.95
#N = 100 # cantidad de iteraciones
#
#rVlis, tVlis, Elis, gRlis, gTlis = graDescMome(alfaR, alfaT, beta, N, data, rVec, tVec, data)
#
#
#rV = np.array(rVlis)
#tV = np.array(tVlis)
#gR = np.array(gRlis)
#gT = np.array(gTlis)
#
#plt.figure()
##plt.set_title('Error cuadrático')
#plt.plot(Elis,'-+')
##plt.show()
#
##plt.figure()
##plt.plot(np.linalg.norm(gR, axis=1))
##plt.plot(np.linalg.norm(gT, axis=1))
#
#
#objectPointsProj = cl.inverse(imagePoints, rV[0], tV[0], cameraMatrix, distCoeffs, model)
#cl.fiducialComparison(rV[0], tV[0], objectPointsOrig, objectPointsProj)
#
#
#objectPointsProj = cl.inverse(imagePoints, rV[-1], tV[-1], cameraMatrix, distCoeffs, model)
#cl.fiducialComparison(rV[-1], tV[-1], objectPointsOrig, objectPointsProj)
#
#
## levanto la camara 3 metros
#tVMap = - np.dot(cv2.Rodrigues(-rV[-1])[0], tV[-1])
#tVMap[2] = 19
#t2 = - np.dot(cv2.Rodrigues(rV[-1])[0], tVMap)
#
#Esq(imagePoints, objectPoints, rV[-1], tV[-1], cameraMatrix, distCoeffs, model)
#Esq(imagePoints, objectPoints, rV[-1], t2, cameraMatrix, distCoeffs, model)
#
#objectPointsProj = cl.inverse(imagePoints, rV[-1], t2, cameraMatrix, distCoeffs, model)
#cl.fiducialComparison(rV[-1], t2, objectPointsOrig, objectPointsProj)
#
#
#fig = plt.figure()
##ax.set_title('posicion de la camara')
#ax = fig.gca(projection='3d')
#ax.plot(tV[:,0], tV[:,1], tV[:,2])
#ax.scatter(objectPointsOrig[:,0], objectPointsOrig[:,1], objectPointsOrig[:,2])
##fig.show()
#
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
##ax.set_title('posicion de la camara')
#ax.plot(tV[:,0], tV[:,1], tV[:,2], 'b-+')
#ax.scatter(tV[0,0], tV[0,1], tV[0,2], 'ok')
#ax.scatter(tV[-1,0], tV[-1,1], tV[-1,2], 'or')
##fig.show()
#
#
##euV = rodrigues2euler(rV)
#fig = plt.figure()
#ax = fig.gca(projection='3d')
##ax.set_title('vector de rodrigues')
#ax.plot(rV[:,0], rV[:,1], rV[:,2], 'b-+')
#ax.scatter(rV[0,0], rV[0,1], rV[0,2], 'ok')
#ax.scatter(rV[-1,0], rV[-1,1], rV[-1,2], 'or')
##fig.show()
#
### %%
##rVecOpt, tVecOpt, params = cl.calibrateInverse(objectPoints, imagePoints, rVecIni, tVecIni, cameraMatrix, distCoeffs, model)
##
##rVecOpt = cv2.Rodrigues(rVecOpt)[0]
##
##
### %%
##objectPointsProj = cl.inverse(imagePoints, rVecOpt, tVecOpt, cameraMatrix, distCoeffs, model)
##
##cl.fiducialComparison(rVecOpt, tVecOpt, objectPoints, objectPointsProj)
#
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
##