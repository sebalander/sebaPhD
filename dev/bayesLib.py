
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
from calibration import calibrator as cl
from numpy import any as anny

import numdifftools as ndf

from multiprocess import Process, Queue
# https://github.com/uqfoundation/multiprocess/tree/master/py3.6/examples

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

def fat2CamMAtrix(kFlat):
    cameraMatrix = np.zeros((3,3), dtype=float)
    cameraMatrix[[0,1,0,1],[0,1,2,2]] = kFlat
    cameraMatrix[2,2] = 1
    
    return cameraMatrix

def flat2int(X, Ns):
    '''
    hace lo inverso de int2flat
    '''
    kFlat = X[0:Ns[0]]
    dFlat = X[Ns[0]:Ns[1]]
    
    cameraMatrix = fat2CamMAtrix(kFlat)
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
    n, m, imagePoints, model, chessboardModel, Ci = params
    
    try: # check if there is covariance for this image
        Cov = Ci[j]
    except:
        Cov = None
    
    # hago la proyeccion
    xm, ym, Cm = cl.inverse(imagePoints[j,0], rvec, tvec, cameraMatrix,
                            distCoeffs, model, Cccd=Cov)
    # print(Cm)
    # error
    er = ([xm, ym] - chessboardModel[0,:,:2].T).T
    
    Cmbool = anny(Cm)
    
    if Cmbool:
        # devuelvo error cuadratico pesado por las covarianzas
        S = np.linalg.inv(Cm)  # inverse of covariance matrix
        # q1 = [np.sum(S[:, :, 0] * er.T, 1),  # fastest way I found to do product
        #       np.sum(S[:, :, 1] * er.T, 1)]
        
        # distancia mahalanobis
        Er = (er.reshape((-1,2,1)) * S * er.reshape((-1,1,2))).sum()
        Er += np.log(np.linalg.det(Cm)).sum()  # sumo termino de normalizacion
    else:
        # error cuadratico pelado, escalar
        Er = np.sum(er**2)
    
    return Er



def errorCuadraticoInt(Xint, Ns, XextList, params):
    '''
    el error asociado a todas la imagenes, es para optimizar respecto a los
    parametros intrinsecos
    '''
    # error
    Er = 0
    
    for j in range(len(XextList)):
        # print(j)
        Er += errorCuadraticoImagen(XextList[j], Xint, Ns, params, j)
    
    return Er

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
        # print('cuentas intrinsecas, 2 processos')
        pHInt = Process(target=procHint, args=(Xint, Ns, XextList,
                                               params, hInt))
        pHInt.start()
    #else:
        # print('cuentas intrinsecas, 1 processo')
    
    pJInt = Process(target=procJint, args=(Xint, Ns, XextList, params, jInt))
    
    pJInt.start()  # inicio procesos
    
    # donde guardar resultados de jaco y hess externos
    n = len(XextList)
    jExt = np.zeros((n, 1, 6), dtype=float)
    qJext = [Queue() for nn in range(n)]

    if hessianos:
        hExt = np.zeros((n, 6, 6), dtype=float)
        qHext = [Queue() for nn in range(n)]
    
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
