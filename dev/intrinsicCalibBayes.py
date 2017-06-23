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

import numdifftools as ndf

from threading import Thread

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
    
    cameraMatrix, distCoeffs = flat2int(Xint, Ns)
    rvec, tvec = flat2ext(Xext)
    
    n, m, imagePoints, model, chessboardModel = params
    
    objectPointsProjected = cl.inverse(imagePoints[j,0], rvec, tvec,
                                        cameraMatrix, distCoeffs, model)
    
    er = objectPointsProjected - chessboardModel[0,:,:2].T
    
    return np.sum(er**2)



def errorCuadraticoInt(Xint, Ns, XextList, params):
    cameraMatrix, distCoeffs = flat2int(Xint, Ns)
    
    n, m, imagePoints, model, chessboardModel = params
    
    er = np.zeros((2 * n, m), dtype=float)
    
    for j in range(n):
        rvec, tvec = flat2ext(XextList[j])
        
        objectPointsProjected = cl.inverse(imagePoints[j,0], rvec, tvec,
                                            cameraMatrix, distCoeffs, model)
        
        er[2*j:2*j+2] = objectPointsProjected - chessboardModel[0,:,:2].T
    
    return np.sum(er**2)


# %%
params = [n, m, imagePoints, model, chessboardModel]

Xint, Ns = int2flat(cameraMatrix, distCoeffs)
XextList = [ext2flat(rVecs[i], tVecs[i])for i in range(n)]

# pruebo con una imagen
j=0
Xext = XextList[0]
errorCuadraticoInt(Xint, Ns, XextList, params)
errorCuadraticoImagen(Xext, Xint, Ns, params, j)

# %% funciones para calcular jacobiano y hessiano in y externo
Jint = ndf.Jacobian(errorCuadraticoInt)  # (Ns,)
Hint = ndf.Hessian(errorCuadraticoInt)  #  (Ns, Ns)
Jext = ndf.Jacobian(errorCuadraticoImagen)  # (6,)
Hext = ndf.Hessian(errorCuadraticoImagen)  # (6,6)


# %% una funcion para cada hilo
def hiloJint(Xint, Ns, XextList, params, ret):
    ret[:]=Jint(Xint, Ns, XextList, params)

def hiloHint(Xint, Ns, XextList, params, ret):
    ret[:]=Hint(Xint, Ns, XextList, params)

def hiloJext(Xext, Xint, Ns, params, j, ret):
    ret[:]=Jext(Xext, Xint, Ns, params, j)

def hiloHext(Xext, Xint, Ns, params, j, ret):
    ret[:]=Hext(Xext, Xint, Ns, params, j)


# %%
# donde guardar resultado de derivadas de params internos
jInt = np.zeros((Ns[-1]), dtype=float)
hInt = np.empty((Ns[-1], Ns[-1]), dtype=float)

# creo e inicializo los threads
tJInt = Thread(target=hiloJint, args=(Xint, Ns, XextList, params, jInt))
tHInt = Thread(target=hiloJint, args=(Xint, Ns, XextList, params, hInt))
tJInt.start()
tHInt.start()

# donde guardar resultados de jaco y hess externos
jExt = np.zeros((n, 6), dtype=float)
hExt = np.empty((n, 6, 6), dtype=float)

# lista de threads
threJ = list()
threH = list()

# creo e inicializo los threads
for i in range(n):
    tJ = Thread(target=hiloJext, args=(Xext, Xint, Ns, params, i, jExt[i]))
    tH = Thread(target=hiloHext, args=(Xext, Xint, Ns, params, i, hExt[i]))
    threJ.append(tJ)
    threH.append(tH)
    tJ.start()
    tH.start()

# wait for all threads
tJInt.join()
tHInt.join()
[t.join() for t in threJ]
[t.join() for t in threH]

