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
import glob
import numpy as np
import scipy.linalg as ln
from calibration import calibrator as cl
import matplotlib.pyplot as plt
from importlib import reload
from copy import deepcopy as dc

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
m = chessboardModel.shape[1]  # cantidad de puntos


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

# %% funcion error
# MAP TO HOMOGENOUS PLANE TO GET RADIUS

def int2flat(cameraMatrix, distCoeffs):
    '''
    parametros intrinsecos concatenados
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
def jacobianos(Xint, Ns, XextList, params):
    '''
    funcion que calcula los jacobianos y hessianos de las variables intrinsecas
    y extrinsecas. hace un hilo para cada cuenta
    '''
    # donde guardar resultado de derivadas de params internos
    jInt = Queue()
    hInt = Queue()
    
    # creo e inicializo los threads
    print('cuentas intrinsecas, 2 processos')
    pJInt = Process(target=procJint, args=(Xint, Ns, XextList, params, jInt))
    pHInt = Process(target=procHint, args=(Xint, Ns, XextList, params, hInt))
    
    pJInt.start()  # inicio procesos
    pHInt.start()
    
    
    # donde guardar resultados de jaco y hess externos
    jExt = np.zeros((n, 1, 6), dtype=float)
    hExt = np.zeros((n, 6, 6), dtype=float)
    qJext = [Queue()]*n
    qHext = [Queue()]*n
    
    # lista de threads
    proJ = list()
    proH = list()
    
    # creo e inicializo los threads
    for i in range(n):
        print('starting par de processos ', i+3)
        pJ = Process(target=procJext, args=(XextList[i], Xint, Ns,
                                            params, i, qJext[i]))
        pH = Process(target=procHext, args=(XextList[i], Xint, Ns,
                                            params, i, qHext[i]))
        proJ.append(pJ)
        proH.append(pH)
        
        pJ.start()  # inicio procesos
        pH.start()
        
    jInt = jInt.get()  # saco los resultados
    hInt = hInt.get()
    for i in range(n):
        jExt[i] = qJext[i].get()  # guardo resultados
        hExt[i] = qHext[i].get()
    
    
    pJInt.join()  # espero a que todos terminen
    pHInt.join()
    [p.join() for p in proJ]
    [p.join() for p in proH]
    
    return jInt, hInt, jExt, hExt


testearFunc = True

if testearFunc:
    # pruebo de evaluar jacobianos y hessianos
    jInt, hInt, jExt, hExt = jacobianos(Xint, Ns, XextList, params)
    plt.matshow(hInt)
    print(ln.det(hInt))


# %%
# parametros auxiliares
params = [n, m, imagePoints, model, chessboardModel]

# pongo en forma flat los valores iniciales
Xint, Ns = int2flat(cameraMatrix, distCoeffs)
XextList = [ext2flat(rVecs[i], tVecs[i])for i in range(n)]

def newtonOptE2(cameraMatrix, distCoeffs, rVecs, tVecs, Ns, params, ep=1e-20):
    n = params[0]
    
    errores = list()
    
    # pongo las variables en flat, son las cond iniciales
    Xint0, Ns = int2flat(cameraMatrix, distCoeffs)
    XextList0 = np.array([ext2flat(rVecs[i], tVecs[i]) for i in range(n)])
    
    
    errores.append(errorCuadraticoInt(Xint0, Ns, XextList0, params))
    print(errores[-1])

    # hago un paso
    # jacobianos y hessianos
    jInt, hInt, jExt, hExt = jacobianos(Xint0, Ns, XextList0, params)
    
    # corrijo intrinseco
    dX = - ln.inv(hInt).dot(jInt.T)
    Xint1 = Xint0 + dX[:,0]
    XextList1 = np.empty_like(XextList0)
    # corrijo extrinseco
    for i in range(n):
            dX = - ln.inv(hExt[i]).dot(jExt[i].T)
            XextList1[i] = XextList0[i] + dX[:,0]
    
    errores.append(errorCuadraticoInt(Xint1, Ns, XextList1, params))
    
    # mientras las correcciones sean mayores a un umbral
    while np.max([np.max(np.abs(Xint1 - Xint0)),
                  np.max(np.abs(XextList1 - XextList0))]) > ep:
        Xint0[:] = Xint1
        XextList0[:] = XextList1
        
        # jacobianos y hessianos
        jInt, hInt, jExt, hExt = jacobianos(Xint0, Ns, XextList0, params)
        
        # corrijo intrinseco
        dX = - ln.inv(hInt).dot(jInt.T)
        Xint1 = Xint0 + dX[:,0]
        XextList1 = np.empty_like(XextList0)
        # corrijo extrinseco
        for i in range(n):
                dX = - ln.inv(hExt[i]).dot(jExt[i.T])
                XextList1[i] = XextList0[i] + dX[:,0]
        
        errores.append(errorCuadraticoInt(Xint1, Ns, XextList1, params))
    
    return flat2int(XextList1, Ns), [ext2flat(x)for x in XextList1], errores

# %%
intr, extr, ers = newtonOptE2(cameraMatrix, distCoeffs, rVecs, tVecs, Ns, params)




