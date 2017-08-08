#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 23:38:35 2017

para UNLP

calibracion con incerteza:
1- calibracion intrinseca chessboard con ocv
2- tomo como condicion inicial y optimizo una funcion error custom
3- saco el hessiano en el optimo
4- sacar asi la covarianza de los parametros optimizados

teste0:
1- con imagenes no usadas para calibrar calcula la probabilidad de los
parámetros optimos dado los datos de test

@author: sebalander
"""

# %%
#import glob
import numpy as np
import scipy.linalg as ln
from calibration import calibrator as cl
import matplotlib.pyplot as plt
#from importlib import reload
#from copy import deepcopy as dc

import numdifftools as ndf

from multiprocess import Process, Queue
# https://github.com/uqfoundation/multiprocess/tree/master/py3.6/examples

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

# load data
imagePoints = np.load(cornersFile)
n = len(imagePoints)  # cantidad de imagenes
indexes = np.arange(n)

np.random.shuffle(indexes)
indexes = indexes[:15]

imagePoints = imagePoints[indexes]
n = len(imagePoints)  # cantidad de imagenes

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
rVecs = np.load(rVecsFile)[indexes]
tVecs = np.load(tVecsFile)[indexes]

# %% funcion error
# MAP TO HOMOGENOUS PLANE TO GET RADIUS

def int2flat(cameraMatrix, distCoeffs):
    '''
    parametros intrinsecos concatenados como un solo vector
    '''
    kFlat = cameraMatrix[[0,1,0,1],[0,1,2,2]]
    dFlat = np.reshape(distCoeffs, -1)
    
    X = np.concatenate((kFlat, dFlat))
    Ns = np.array([len(kFlat), len(dFlat)])
    Ns = np.cumsum(Ns)
    
    return X, Ns


def ext2flat(rVec, tVec):
    '''
    toma un par rvec, tvec y devuelve uno solo concatenado
    '''
    rFlat = np.reshape(rVec, -1)
    tFlat = np.reshape(tVec, -1)
    
    X = np.concatenate((rFlat, tFlat))
    
    return X


def flat2int(X, Ns):
    '''
    hace lo inverso de int2flat
    '''
    kFlat = X[0:Ns[0]]
    dFlat = X[Ns[0]:Ns[1]]
    
    cameraMatrix = np.zeros((3,3), dtype=float)
    cameraMatrix[[0,1,0,1],[0,1,2,2]] = kFlat
    cameraMatrix[2,2] = 1
    
    distCoeffs = dFlat
    
    return cameraMatrix, distCoeffs

def flat2ext(X):
    '''
    hace lo inverso de ext2flat
    '''
    rFlat = X[0:3]
    tFlat = X[3:]
    
    rVecs = np.reshape(rFlat, 3)
    tVecs = np.reshape(tFlat, 3)
    
    return rVecs, tVecs

# %%

def errorCuadraticoImagen(Xext, Xint, Ns, params, j):
    '''
    el error asociado a una sola imagen, es para un par rvec, tvec
    necesita tambien los paramatros intrinsecos
    '''
    # saco los parametros de flat para que los use la func de projection
    cameraMatrix, distCoeffs = flat2int(Xint, Ns)
    rvec, tvec = flat2ext(Xext)
    # saco los parametros auxiliares
    n, m, imagePoints, model, chessboardModel = params
    # hago la proyeccion
    objectPointsProjected = cl.inverse(imagePoints[j,0], rvec, tvec,
                                        cameraMatrix, distCoeffs, model)
    # error
    er = objectPointsProjected - chessboardModel[0,:,:2].T
    # devuelvo error cuadratico
    return np.sum(er**2)



def errorCuadraticoInt(Xint, Ns, XextList, params):
    '''
    el error asociado a todas la imagenes, es para optimizar respecto a los
    parametros intrinsecos
    '''
    # saco los parametros de flat para que los use la func de projection
    cameraMatrix, distCoeffs = flat2int(Xint, Ns)
    # saco los parametros auxiliares
    n, m, imagePoints, model, chessboardModel = params
    # reservo lugar para el error
    er = np.zeros((2 * n, m), dtype=float)
    
    for j in range(n):
        # descomprimos los valores de pose para que los use la funcion
        rvec, tvec = flat2ext(XextList[j])
        # hago la proyeccion
        objectPointsProjected = cl.inverse(imagePoints[j,0], rvec, tvec,
                                            cameraMatrix, distCoeffs, model)
        # calculo del error
        er[2*j:2*j+2] = objectPointsProjected - chessboardModel[0,:,:2].T
    # devuelvo el error cuadratico total
    return np.sum(er**2)


# %%
testearFunc = True
if testearFunc:
    # parametros auxiliares
    params = [n, m, imagePoints, model, chessboardModel]
    
    # pongo en forma flat los valores iniciales
    Xint, Ns = int2flat(cameraMatrix, distCoeffs)
    XextList = [ext2flat(rVecs[i], tVecs[i])for i in range(n)]
    
    # pruebo con una imagen
    j=0
    Xext = XextList[0]
    errorCuadraticoImagen(Xext, Xint, Ns, params, j)
    # pruebo el error total
    errorCuadraticoInt(Xint, Ns, XextList, params)

# %% funciones para calcular jacobiano y hessiano in y externo
Jint = ndf.Jacobian(errorCuadraticoInt)  # (Ns,)
Hint = ndf.Hessian(errorCuadraticoInt)  #  (Ns, Ns)
Jext = ndf.Jacobian(errorCuadraticoImagen)  # (6,)
Hext = ndf.Hessian(errorCuadraticoImagen)  # (6,6)

# una funcion para cada hilo
def procJint(Xint, Ns, XextList, params, ret):
    ret.put(Jint(Xint, Ns, XextList, params))

def procHint(Xint, Ns, XextList, params, ret):
    ret.put(Hint(Xint, Ns, XextList, params))

def procJext(Xext, Xint, Ns, params, j, ret):
    ret.put(Jext(Xext, Xint, Ns, params, j))

def procHext(Xext, Xint, Ns, params, j, ret):
    ret.put(Hext(Xext, Xint, Ns, params, j))


# pruebo evaluar las funciones para los processos
testearFunc = True
if testearFunc:
    jInt = Queue()  # np.zeros((Ns[-1]), dtype=float)
    hInt = Queue()  # np.zeros((Ns[-1], Ns[-1]), dtype=float)
    
    pp = list()
    pp.append(Process(target=procJint, args=(Xint, Ns, XextList, params, jInt)))
    pp.append(Process(target=procHint, args=(Xint, Ns, XextList, params, hInt)))
    
    
    jExt = Queue()  # np.zeros(6, dtype=float)
    hExt = Queue()  # np.zeros((6, 6), dtype=float)
    
    pp.append(Process(target=procJext, args=(Xext, Xint, Ns, params, 0, jExt)))
    pp.append(Process(target=procHext, args=(Xext, Xint, Ns, params, 0, hExt)))
    
    [p.start() for p in pp]
    
    jInt = jInt.get()
    hInt = hInt.get()
    jExt = jExt.get()
    hExt = hExt.get()
    
    print(jInt)
    print(hInt)
    print(jExt)
    print(hExt)
    
    [p.join() for p in pp]


# %%
def jacobianos(Xint, Ns, XextList, params, hessianos=True):
    '''
    funcion que calcula los jacobianos y hessianos de las variables intrinsecas
    y extrinsecas. hace un hilo para cada cuenta
    '''
    # donde guardar resultado de derivadas de params internos
    jInt = Queue()
    if hessianos:
        hInt = Queue()
    
    # creo e inicializo los threads
    if hessianos:
        print('cuentas intrinsecas, 2 processos')
        pHInt = Process(target=procHint, args=(Xint, Ns, XextList,
                                               params, hInt))
        pHInt.start()
    else:
        print('cuentas intrinsecas, 1 processo')
    
    pJInt = Process(target=procJint, args=(Xint, Ns, XextList, params, jInt))
    
    pJInt.start()  # inicio procesos
    
    # donde guardar resultados de jaco y hess externos
    jExt = np.zeros((n, 1, 6), dtype=float)
    qJext = [Queue()]*n

    if hessianos:
        hExt = np.zeros((n, 6, 6), dtype=float)
        qHext = [Queue()]*n
    
    # lista de threads
    proJ = list()
    if hessianos:
        proH = list()
    
    # creo e inicializo los threads
    for i in range(n):
        # print('starting par de processos ', i + 3)
        pJ = Process(target=procJext, args=(XextList[i], Xint, Ns,
                                            params, i, qJext[i]))
        proJ.append(pJ)
        
        if hessianos:
            pH = Process(target=procHext, args=(XextList[i], Xint, Ns,
                                                params, i, qHext[i]))
            proH.append(pH)
        
        pJ.start()  # inicio procesos
        if hessianos:
            pH.start()
    
    jInt = jInt.get()  # saco los resultados
    if hessianos:
        hInt = hInt.get()
    
    for i in range(n):
        jExt[i] = qJext[i].get()  # guardo resultados
        if hessianos:
            hExt[i] = qHext[i].get()
    
    
    pJInt.join()  # espero a que todos terminen
    
    if hessianos:
        pHInt.join()
    
    [p.join() for p in proJ]
    
    if hessianos:
        [p.join() for p in proH]
    
    if hessianos:
        return jInt, hInt, jExt, hExt
    else:
        return jInt, jExt

# %%
testearFunc = True
if testearFunc:
    # pruebo de evaluar jacobianos y hessianos
    jInt, hInt, jExt, hExt = jacobianos(Xint, Ns, XextList, params)
    plt.matshow(hInt)
    [plt.matshow(h) for h in hExt]
    print(ln.det(hInt))
    plt.figure()
    plt.plot(jInt[0], 'b+-')
    plt.figure()
    [plt.plot(j[0], '-+') for j in jExt]

    # pruebo solo con los jacobianos
    jInt,  jExt = jacobianos(Xint, Ns, XextList, params, hessianos=False)
    plt.figure()
    plt.plot(jInt[0], 'b+-')
    plt.figure()
    [plt.plot(j[0], '-+') for j in jExt]

# %%
# parametros auxiliares
params = [n, m, imagePoints, model, chessboardModel]

# pongo en forma flat los valores iniciales
Xint, Ns = int2flat(cameraMatrix, distCoeffs)
XextList = [ext2flat(rVecs[i], tVecs[i])for i in range(n)]

def newtonOptE2(cameraMatrix, distCoeffs, rVecs, tVecs, Ns, params, ep=1e-3):
    n = params[0]
    
    errores = list()
    
    # pongo las variables en flat, son las cond iniciales
    Xint0, Ns = int2flat(cameraMatrix, distCoeffs)
    XextList0 = np.array([ext2flat(rVecs[i], tVecs[i]) for i in range(n)])
    
    
    errores.append(errorCuadraticoInt(Xint0, Ns, XextList0, params))
    print('error es ', errores[-1])

    # hago un paso
    # jacobianos y hessianos
    print('jacobianos')
    jInt, hInt, jExt, hExt = jacobianos(Xint0, Ns, XextList0, params)
    
    
    print('inicio correcciones')
    # corrijo intrinseco
    l = ln.svd(hInt)[1]**2
    print('autovals ', np.max(l), np.min(l))
    dX = - ln.inv(hInt).dot(jInt.T)
    Xint1 = Xint0 + dX[:,0]
    XextList1 = np.empty_like(XextList0)
    # corrijo extrinseco
    for i in range(n):
        l = ln.svd(hExt[i])[1]**2
        print('autovals ', np.max(l), np.min(l))
        dX = - ln.inv(hExt[i]).dot(jExt[i].T)
        XextList1[i] = XextList0[i] + dX[:,0]
    
    errores.append(errorCuadraticoInt(Xint1, Ns, XextList1, params))
    print('error es ', errores[-1])
    
    e = np.max([np.max(np.abs((Xint1 - Xint0) / Xint0)),
                np.max(np.abs((XextList1 - XextList0) / XextList0))])
    print('correccion hecha ', e)
    
    # mientras las correcciones sean mayores a un umbral
    while e > ep:
        Xint0[:] = Xint1
        XextList0[:] = XextList1
        
        # jacobianos y hessianos
        print('jacobianos')
        jInt, hInt, jExt, hExt = jacobianos(Xint0, Ns, XextList0, params)
        
        print('inicio correcciones')
        # corrijo intrinseco
        l = ln.svd(hInt)[0]**2
        print('autovals ', np.max(l), np.min(l))
        dX = - ln.inv(hInt).dot(jInt.T)
        Xint1 = Xint0 + dX[:,0]
        XextList1 = np.empty_like(XextList0)
        # corrijo extrinseco
        for i in range(n):
            l = ln.svd(hExt[i])[0]**2
            print('autovals ', np.max(l), np.min(l))
            dX = - ln.inv(hExt[i]).dot(jExt[i].T)
            XextList1[i] = XextList0[i] + dX[:,0]
        
        e = np.max([np.max(np.abs((Xint1 - Xint0) / Xint0)),
                    np.max(np.abs((XextList1 - XextList0) / XextList0))])
        print('correccion hecha ', e)

        errores.append(errorCuadraticoInt(Xint1, Ns, XextList1, params))
        print('error es ', errores[-1])
    
    return flat2int(XextList1, Ns), [ext2flat(x)for x in XextList1], errores

# %%
intr, extr, ers = newtonOptE2(cameraMatrix, distCoeffs, rVecs, tVecs, Ns, params)


# %% solo gradiente descendente
# parametros auxiliares
params = [n, m, imagePoints, model, chessboardModel]

# pongo en forma flat los valores iniciales
Xint, Ns = int2flat(cameraMatrix, distCoeffs)
XextList = [ext2flat(rVecs[i], tVecs[i])for i in range(n)]


def gradDescE2(cameraMatrix, distCoeffs, rVecs, tVecs, Ns, params, ep=1e-4, eta=0.01):
    n = params[0]
    
    errores = list()
    
    # pongo las variables en flat, son las cond iniciales
    Xint0, Ns = int2flat(cameraMatrix, distCoeffs)
    XextList0 = np.array([ext2flat(rVecs[i], tVecs[i]) for i in range(n)])
    
    
    errores.append(errorCuadraticoInt(Xint0, Ns, XextList0, params))
    print('error es ', errores[-1])

    # hago un paso
    # jacobianos y hessianos
    print('jacobianos')
    jInt, jExt = jacobianos(Xint0, Ns, XextList0, params, hessianos=False)
    
    
    print('inicio correcciones')
    # corrijo intrinseco
    dX = - eta * jInt #  ln.inv(hInt).dot(jInt.T)
    Xint1 = Xint0 + dX[:,0]
    XextList1 = np.empty_like(XextList0)
    # corrijo extrinseco
    for i in range(n):
        dX = - eta * jExt[i]#  ln.inv(hExt[i]).dot(jExt[i].T)
        XextList1[i] = XextList0[i] + dX[:,0]
    
    errores.append(errorCuadraticoInt(Xint1, Ns, XextList1, params))
    print('error es ', errores[-1])
    
    e = np.max([np.max(np.abs((Xint1 - Xint0) / Xint0)),
                np.max(np.abs((XextList1 - XextList0) / XextList0))])
    print('correccion hecha ', e)
    
    # mientras las correcciones sean mayores a un umbral
    while e > ep:
        Xint0[:] = Xint1
        XextList0[:] = XextList1
        
        # jacobianos y hessianos
        print('jacobianos')
        jInt, jExt = jacobianos(Xint0, Ns, XextList0, params, hessianos=False)
        
        print('inicio correcciones')
        # corrijo intrinseco
        dX = - eta * jInt #  ln.inv(hInt).dot(jInt.T)
        Xint1 = Xint0 + dX[:,0]
        XextList1 = np.empty_like(XextList0)
        # corrijo extrinseco
        for i in range(n):
            dX = - eta * jExt[i] #  ln.inv(hExt[i]).dot(jExt[i].T)
            XextList1[i] = XextList0[i] + dX[:,0]
        
        e = np.max([np.max(np.abs((Xint1 - Xint0) / Xint0)),
                    np.max(np.abs((XextList1 - XextList0) / XextList0))])
        print('correccion hecha ', e)

        errores.append(errorCuadraticoInt(Xint1, Ns, XextList1, params))
        print('error es', errores[-1],
              '. respecto anterior: %.3g'%(errores[-1] - errores[-2]))
    
    return flat2int(XextList1, Ns), [ext2flat(x)for x in XextList1], errores

# %%
intr, extr, ers = gradDescE2(cameraMatrix, distCoeffs, rVecs, tVecs, Ns,
                             params, eta=1e-5)

# %%
# parametros auxiliares
params = [n, m, imagePoints, model, chessboardModel]

# pongo en forma flat los valores iniciales
Xint0, Ns = int2flat(cameraMatrix, distCoeffs)
XextList0 = [ext2flat(rVecs[i], tVecs[i])for i in range(n)]
e0 = errorCuadraticoInt(Xint0, Ns, XextList0, params)

# saco los jacobianos y hessianos para analizar con cuidado cada paso
jInt, hInt, jExt, hExt = jacobianos(Xint0, Ns, XextList0, params)


# %% analizamos lo que pasa con la correccion en los intrinsecos
etasI = np.concatenate((np.linspace(-0.1, -0.0002, 30),
                       np.linspace(-0.0002, 0.001, 500),
                       np.linspace(0.001, 0.1, 30)))
npts = etasI.shape[0]
errsI = np.empty(npts, dtype=float)

deltaX = - np.array([jInt]*npts)[:,0,:] * etasI.reshape((-1,1))
Xmod = deltaX + Xint0  # modified parameters

for i in range(npts):
    errsI[i] = errorCuadraticoInt(Xmod[i], Ns, XextList0, params)

plt.figure()
plt.plot(etasI, errsI)


# %% analizamos lo que pasa con la correccion extrinseca
#etas =np.linspace(-0.1, 0.1, 30)

etasE = np.concatenate((np.linspace(-0.1, -0.001, 30),
                       np.linspace(-0.001, 0.001, 50),
                       np.linspace(0.001, 0.1, 30)))

npts = etasE.shape[0]
errsE = np.empty(npts, dtype=float)

Xext0 = np.hstack(XextList0)
jacob = np.hstack(jExt[:,0,:])

deltaX = - np.array([jacob]*npts) * etasE.reshape((-1,1))
Xmod = deltaX + Xext0  # modified parameters
Xmod = Xmod.reshape(npts, -1, 6)

for i in range(npts):
    listExt = [x for x in Xmod[i]]
    errsE[i] = errorCuadraticoInt(Xint0, Ns, listExt, params)

plt.figure()
plt.plot(etasE, errsE)

# %% pruebo con muchos valores de parametros alrededor del que tenemos para
# ver si es el optimo

npts = 100
# pongo los params extr como una gran lista
Xext0 = np.hstack(XextList0)

perturbInt = np.random.rand(npts, Xint0.shape[0]) * 0.02 - 0.01
perturbExt = np.random.rand(npts, Xext0.shape[0]) * 0.02 - 0.01

IntMod = perturbInt + Xint0
ExtMod = perturbExt + Xext0
ExtMod = ExtMod.reshape(npts, -1, 6)

dists = np.empty(npts,3)

for i in range(npts):
    listExt = [x for x in ExtMod[i]]
    errsE[i] = errorCuadraticoInt(IntMod[i], Ns, listExt, params)

# elijo el que tiene error mas chico de todos
emin = np.min(errsE)
if e0 < emin:
    # el optimo es la cond inicial
    
    
    









