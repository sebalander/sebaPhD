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
import matplotlib.pyplot as plt
from importlib import reload
from copy import deepcopy as dc
import numdifftools as ndf
from dev import multipolyfit as mpf

from dev import bayesLib as bl

from multiprocess import Process, Queue, Value
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



intrinsicHessianFile = imagesFolder + camera + model + "intrinsicHessian.npy"


# %%
reload(bl)
reload(bl.cl)
testearFunc = True
if testearFunc:
    # parametros auxiliares
    # Ci = None
    Ci = np.repeat([np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
    params = [n, m, imagePoints, model, chessboardModel, Ci]
    
    # pongo en forma flat los valores iniciales
    Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs)
    XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]
    
    # pruebo con una imagen
    j=0
    Xext = XextList[0]
    print(bl.errorCuadraticoImagen(Xext, Xint, Ns, params, j))
    
    # pruebo el error total
    print(bl.errorCuadraticoInt(Xint, Ns, XextList, params))


# %% pruebo evaluar las funciones para los processos
testearFunc = False
if testearFunc:
    Ci = np.repeat([np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
    params = [n, m, imagePoints, model, chessboardModel, Ci]
    
    jInt = Queue()  # np.zeros((Ns[-1]), dtype=float)
    hInt = Queue()  # np.zeros((Ns[-1], Ns[-1]), dtype=float)
    
    pp = list()
    pp.append(Process(target=bl.procJint, args=(Xint, Ns, XextList, params, jInt)))
    pp.append(Process(target=bl.procHint, args=(Xint, Ns, XextList, params, hInt)))
    
    
    jExt = Queue()  # np.zeros(6, dtype=float)
    hExt = Queue()  # np.zeros((6, 6), dtype=float)
    
    j = 23
    pp.append(Process(target=bl.procJext, args=(Xext, Xint, Ns, params, j, jExt)))
    pp.append(Process(target=bl.procHext, args=(Xext, Xint, Ns, params, j, hExt)))
    
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
testearFunc = False
if testearFunc:
    Ci = np.repeat([np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
    params = [n, m, imagePoints, model, chessboardModel, Ci]
    # pruebo de evaluar jacobianos y hessianos
    Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs)
    XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]
    
    jInt, hInt, jExt, hExt = bl.jacobianos(Xint, Ns, XextList, params)
    plt.matshow(hInt)
    [plt.matshow(h) for h in hExt]
    print(ln.det(hInt))
    plt.figure()
    plt.plot(jInt[0], 'b+-')
    plt.figure()
    [plt.plot(j[0], '-+') for j in jExt]

    # pruebo solo con los jacobianos
    jInt,  jExt = bl.jacobianos(Xint, Ns, XextList, params, hessianos=False)
    plt.figure()
    plt.plot(jInt[0], 'b+-')
    plt.figure()
    [plt.plot(j[0], '-+') for j in jExt]

# %% calculo el hessiano en la solucion de opencv
calcSaveHessInt = False
if calcSaveHessInt:
    Ci = np.repeat([np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
    params = [n, m, imagePoints, model, chessboardModel, Ci]
    # pruebo de evaluar jacobianos y hessianos
    jInt, hInt, jExt, hExt = bl.jacobianos(Xint, Ns, XextList, params)
    
    np.save(intrinsicHessianFile, hInt)

w, v = ln.eig(hInt)

# %% grafico la función error en las direcciones de los autovalores del hessiano

plotErrorFunc = True
if plotErrorFunc:
    # parametros auxiliares
    params = [n, m, imagePoints, model, chessboardModel, Ci]
    
    # pongo en forma flat los valores iniciales
    Xint0, Ns = bl.int2flat(cameraMatrix, distCoeffs)
    XextList0 = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]
    e0 = bl.errorCuadraticoInt(Xint0, Ns, XextList0, params)
    
    # saco los hessianos si no estan definidos
    if 'hInt' not in locals():
        hInt = bl.Hint(Xint, Ns, XextList, params)
    
    # get eigenvectors
    u, s, v = np.linalg.svd(hInt)
    
    etasLista = list()
    erroresLista = list()
    
    # %%
    etasI = np.linspace(-1e-15, 1e-15, 200)
    
    npts = etasI.shape[0]
    errsI = np.empty(npts, dtype=float)
    
    direction = 0  # indice de la componente ppal que vamos a mirar
    direc = v[direction] * s[direction]
    deltaX = - np.array([direc]*npts) * etasI.reshape((-1,1))
    Xmod = deltaX + Xint0  # modified parameters
    
    # %%
    for i in range(npts):
        print(i)
        errsI[i] = bl.errorCuadraticoInt(Xmod[i], Ns, XextList0, params)
    
    etasLista.append(etasI)
    erroresLista.append(errsI)
    
    # %%
    
    errsI = np.hstack(erroresLista)
    etasI = np.hstack(etasLista)
    
    sortingIndexes = np.argsort(etasI)
    errsI = errsI[sortingIndexes]
    etasI = etasI[sortingIndexes]
    
    plt.figure()
    #plt.xlim([-1.5e-11, 0.0])
    #plt.ylim([0.722315, e0])
    plt.plot(etasI, errsI, '-kx')
    plt.plot([etasI[0], etasI[-1]],[e0, e0], '-r')
    # plt.yscale('log')
    
    # %%
    np.savetxt('dev/firstComponentErrorFunct.txt', np.array([etasI, errsI]).T,
               header='moving by eta in the direction of the most significant eigvect of the hessian of the error function in the optimal point according to OpenCV.')

etasI, errsI = np.loadtxt("./dev/firstComponentErrorFunct.txt").T

# %%
# parametros auxiliares
params = [n, m, imagePoints, model, chessboardModel, Ci]

# pongo en forma flat los valores iniciales
Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs)
XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]

def newtonOptE2(cameraMatrix, distCoeffs, rVecs, tVecs, Ns, params, ep=1e-3):
    n = params[0]
    
    errores = list()
    
    # pongo las variables en flat, son las cond iniciales
    Xint0, Ns = bl.int2flat(cameraMatrix, distCoeffs)
    XextList0 = np.array([bl.ext2flat(rVecs[i], tVecs[i]) for i in range(n)])
    
    
    errores.append(bl.errorCuadraticoInt(Xint0, Ns, XextList0, params))
    print('error es ', errores[-1])

    # hago un paso
    # jacobianos y hessianos
    print('jacobianos')
    jInt, hInt, jExt, hExt = bl.jacobianos(Xint0, Ns, XextList0, params)
    
    
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
    
    errores.append(bl.errorCuadraticoInt(Xint1, Ns, XextList1, params))
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
        jInt, hInt, jExt, hExt = bl.jacobianos(Xint0, Ns, XextList0, params)
        
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

        errores.append(bl.errorCuadraticoInt(Xint1, Ns, XextList1, params))
        print('error es ', errores[-1])
    
    return bl.flat2int(XextList1, Ns), [bl.ext2flat(x)for x in XextList1], errores

# %%

#intr, extr, ers = newtonOptE2(cameraMatrix, distCoeffs*1.001, rVecs, tVecs, Ns, params)
'''
este metodo no esta funcionando para esto. anduvo para la trilateracion pero
se ve que esta funcion de perdida es mas complicada y tiene cosas muy raras
'''

# %% solo gradiente descendente
# parametros auxiliares
params = [n, m, imagePoints, model, chessboardModel, Ci]

# pongo en forma flat los valores iniciales
Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs)
XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]


def gradDescE2(cameraMatrix, distCoeffs, rVecs, tVecs, Ns, params, ep=1e-6, eta=1e2):
    # %%
    ep=1e-6
    eta=1e-2
    n = params[0]
    
    errores = list()
    
    # pongo las variables en flat, son las cond iniciales
    # las saco de donde encontro opencv
    Xint0, Ns = bl.int2flat(cameraMatrix*1.02, distCoeffs*1.02)
    XextList0 = np.array([bl.ext2flat(rVecs[i], tVecs[i]) for i in range(n)])
    aux = [dc(Xint0), dc(XextList0)] # por si hay que volver atrás
    
    errores.append(bl.errorCuadraticoInt(Xint0, Ns, XextList0, params))
    print('error es ', errores[-1])

    # hago un paso
    # jacobianos y hessianos
    print('jacobianos')
    jInt, jExt = bl.jacobianos(Xint0, Ns, XextList0, params, hessianos=False)
    
    
    print('inicio correcciones')
    # corrijo intrinseco
    dX = - eta * jInt #  ln.inv(hInt).dot(jInt.T)
    Xint1 = Xint0 + dX[:,0]
    XextList1 = np.empty_like(XextList0)
    # corrijo extrinseco
    for i in range(n):
        dX = - eta * jExt[i]#  ln.inv(hExt[i]).dot(jExt[i].T)
        XextList1[i] = XextList0[i] + dX[:,0]
    
    errores.append(bl.errorCuadraticoInt(Xint1, Ns, XextList1, params))
    print('error es ', errores[-1])
    
    e = np.max([np.max(np.abs((Xint1 - Xint0) / Xint0)),
                np.max(np.abs((XextList1 - XextList0) / XextList0))])
    print('correccion hecha ', e)
    
    print('error es', errores[-1],
          '. respecto anterior: %.3g'%(errores[-1] - errores[-2]))
    
    if (errores[-1] - errores[-2]) > 0:
        eta /= 10.0 # hago el paso mas chico si la cosa se inestabiliza
        print('nuevo eta %g y vuelvo un paso atrás'%(eta))
        Xint1, XextList1 = dc(aux)
        errores.pop(-1)  # saco el valor de error que no va
    
    # mientras las correcciones sean mayores a un umbral
    while e > ep:
        Xint0[:] = Xint1
        XextList0[:] = XextList1
        aux = [dc(Xint1), dc(XextList1)] # por si hay que volver atrás
        # jacobianos y hessianos
        print('jacobianos')
        jInt, jExt = bl.jacobianos(Xint0, Ns, XextList0, params, hessianos=False)
        
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

        errores.append(bl.errorCuadraticoInt(Xint1, Ns, XextList1, params))
        print('error es', errores[-1],
              '. respecto anterior: %.3g'%(errores[-1] - errores[-2]))
        
        if (errores[-1] - errores[-2]) > 0:
            eta /= 10.0 # hago el paso mas chico si la cosa se inestabiliza
            print('nuevo eta %g y vuelvo un paso atrás'%(eta))
            Xint1, XextList1 = dc(aux)
            errores.pop(-1)  # saco el valor de error que no va
    
    print(XextList1.shape)
    # %%
    return bl.flat2int(XextList1, Ns), [bl.ext2flat(x)for x in XextList1], errores


# %%
intr, extr, ers = gradDescE2(cameraMatrix, distCoeffs, rVecs, tVecs, Ns,
                             params, eta=1e-6)

# %%

def relativeNoise(x, l=0.05):
    '''
    returns an array like x but with a uniform noise that perturbs it's values
    by a fraction l
    '''
    s = x.shape
    n = np.prod(s)
    y = np.random.rand(n) * 2 - 1
    y = y.reshape(s)
    
    return x * (l * y + 1)

# %%
# parametros auxiliares
params = [n, m, imagePoints, model, chessboardModel, Ci]

# pongo en forma flat los valores iniciales
Xint0, Ns = bl.int2flat(cameraMatrix, distCoeffs)
XextList0 = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]
e0 = bl.errorCuadraticoInt(Xint0, Ns, XextList0, params)

# saco los jacobianos y hessianos para analizar con cuidado cada paso
jInt, hInt, jExt, hExt = bl.jacobianos(Xint0, Ns, XextList0, params)


## %% analizamos lo que pasa con la correccion en los intrinsecos
##etasI = np.concatenate((np.linspace(-0.1, -0.001, 30),
##                       np.linspace(-0.001, -2e-6, 100),
##                       np.linspace(-2e-6, 2e-6, 100),
##                       np.linspace(2e-6, 0.001, 100),
##                       np.linspace(0.001, 0.1, 30)))
#rangos = [,,1e-5]


u, s, v = np.linalg.svd(hInt)

# %% testeo en las diferentes direcciones segun el hesiano
etasI = np.linspace(-5e-8, 4e-8, int(1e2))

npts = etasI.shape[0]
errsI = np.empty(npts, dtype=float)

i = 0
direc = v[i] * s[i]
deltaX = - np.array([direc]*npts) * etasI.reshape((-1,1))
Xmod = deltaX + Xint0  # modified parameters

for i in range(npts):
    print(i)
    errsI[i] = bl.errorCuadraticoInt(Xmod[i], Ns, XextList0, params)

plt.figure()
plt.plot(etasI, errsI, '-*')
plt.plot([etasI[0], etasI[-1]],[e0, e0], '-r')

# %%
#et = np.hstack([et, etasI])
#er = np.hstack([er, errsI])
#
#insort = np.argsort(et)
#
#et = et[insort]
#er = er[insort]
#
#plt.figure()
#plt.plot(et, er, '-*')
#plt.plot([et[0], et[-1]],[e0, e0], '-r')

'''
ver aca para plotear mejor
https://stackoverflow.com/questions/14084634/adaptive-plotting-of-a-function-in-python/14084715#14084715
'''

# np.savetxt('curvaerrorEig1.txt', [et, er])

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
    errsE[i] = bl.errorCuadraticoInt(Xint0, Ns, listExt, params)

plt.figure()
plt.plot(etasE, errsE)


# %% pruebo de moverme en cada direccion para ver si efetivamente estamos en
# un minimo

# tengo XextList0 y Xint0 como cond iniciales
e0 = bl.errorCuadraticoInt(Xint0, Ns, XextList0, params)

ers = []
axis = []

nPorc = 20
porc = np.linspace(0.99,1.01,nPorc)

# recorro cada param interno
for i in range(Xint0.shape[0]):
    # genero la version modif de ese param
    xPar = Xint0[i] * porc
    Xmod = dc(Xint0)
    Ci = np.repeat([np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
    params = [n, m, imagePoints, model, chessboardModel, Ci]
    # pruebo de evaluar jacobianos y hessianos
    Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs)
    XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]
    axis.append(xPar - Xint0[i])
    ers.append(np.empty_like(axis[-1]))
    
    for j in range(nPorc):
        Xmod[i] = xPar[j]  # pongo en el parametro i esimo el valor j esimo
        
        ers[-1][j] = bl.errorCuadraticoInt(Xmod, Ns, XextList0, params)

# recorro cada param externo
for k in range(len(XextList0)):
    for i in range(6):
        # genero la version modif de ese param
        xPar = XextList0[k][i] * porc  # los valores donde voy a evaluar
        XlistMod = dc(XextList0) # la lista donde ir poniedo valores auxiliares
        
        axis.append(xPar - XextList0[k][i])  # discrepancia respecto al minimo
        ers.append(np.empty_like(axis[-1]))  # donde guardar los errores
        
        for j in range(nPorc):
            XlistMod[k][i] = xPar[j]  # pongo en el parametro i esimo el valor j esimo
            
            ers[-1][j] = bl.errorCuadraticoInt(Xint0, Ns, XextList0, params)

axx = np.array(axis).T
erx = np.array(ers).T

# los normalizo para que se puedan ver en la grafica
axxNor = axx / axx.max(0)
erMin = erx.min(0)
erMax = erx.max(0)
erxNor = (erx - e0) / (erMax - e0)

plt.plot(axxNor, erxNor)

# %% trato de resolver el problema del hessiano no positivo en los parametros intrinsecos

Ci = np.repeat([np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
params = [n, m, imagePoints, model, chessboardModel, Ci]

# pongo en forma flat los valores iniciales
Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs)
XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]

jInt = bl.Jint(Xint, Ns, XextList, params)  # (Ns,)
hInt = bl.Hint(Xint, Ns, XextList, params)  #  (Ns, Ns)

# %%
deter = ln.det(hInt)

diagonal = np.diag(hInt)

u, s, v = ln.svd(hInt)

# %% intento optimizacion con leastsq
from scipy.optimize import minimize
Ci = np.repeat([np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
params = [n, m, imagePoints, model, chessboardModel, Ci]
# pongo en forma flat los valores iniciales
Xint0, Ns = bl.int2flat(cameraMatrix, distCoeffs)
XextList0 = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]

# %%
meths = ["Nelder-Mead", "Powell", "L-BFGS-B", "COBYLA", "SLSQP"]

res = minimize(bl.errorCuadraticoInt, Xint0,
               args=(Ns, XextList0, params), method=meths[0])
print(res)

import numdifftools as ndf

if res.success is True:
    Hint = ndf.Hessian(bl.errorCuadraticoInt)
    hInt = Hint(res.x, Ns, XextList, params)  #  (Ns, Ns)
    
    cov = ln.inv(hInt)
    plt.matshow(cov)
    #fun, hess_inv, jac, message, nfev, nit, njev, status, success, x = res
    
    sigs = np.sqrt(np.diag(cov))
    plt.figure()
    plt.plot(np.abs(sigs / res.x), 'x')

# %%
plt.figure()
plt.plot(Xint0, res.x, 'x')

(Xint0 - res.x) / Xint0

plt.figure()
plt.plot((Xint0 - res.x) / Xint0, 'x')


# %% ploteo en cada direccion
Npts = 10
etaM = 1e-4 # variacion total para cada lado
etas = 1 + np.linspace(-etaM, etaM, Npts)

XintFin = res.x
XmodAll = np.repeat([XintFin], Npts, axis=0) * etas.reshape((Npts, -1))
erAll = np.zeros_like(XmodAll)


for j in range(len(XintFin)):
    for i in range(Npts):
        print(j, i)
        X = dc(XintFin)
        X[j] = XmodAll[i, j]
        erAll[i, j] = bl.errorCuadraticoInt(X, Ns, XextList0, params)
        print(erAll[i, j])

plt.plot(etas, erAll, '-x')
plt.plot([etas[0], etas[-1]], [res.fun, res.fun])




# %%
# metodos que funcionan:
meths = ["Nelder-Mead", "Powell", "L-BFGS-B", "COBYLA", "SLSQP"]
resss = []

for me in meths:
    res = minimize(bl.errorCuadraticoInt, Xint0,
                   args=(Ns, XextList0, params), method=me)
    print(me)
    print(res)
    resss.append(res)

# %% como la libreria uqe hace ndiftools tiene problemas para calcularl el
# hessiano lo calculo por mi cuenta usando el resultado de los varias metodos


XX = np.array([r.x for r in resss])
YY = np.array([r.fun for r in resss])

armi = np.argmin(YY)
X0 = XX[armi]
Y0 = YY[armi] # bl.errorCuadraticoInt(X0, Ns, XextList0, params)


coefs, exps = mpf.multipolyfit(XX - X0, YY, 2, powers_out=True)

# asume first of 9 element in exponent is for 1
const = 0
jac = np.zeros((8))
hes = np.zeros((8,8))

const = coefs[0]
jac = coefs[1:9]
hes = np.zeros((8,8))
hes[np.tril_indices(8)] = hes[np.triu_indices(8)] = coefs[9:]


print(const)
print(jac)
print(hes)

# testeo
[const + jac.dot(x) + x.dot(hes).dot(x)/2 for x in (XX - X0)]

# %% ahora augmento las soluciones obtenidas
XXX = np.zeros((5,5,5,5,5,5,5,5,8))

for i in range(5):
    for j in range(5):
        for k in range(5):
            for l in range(5):
                for m in range(5):
                    for n in range(5):
                        for o in range(5):
                            for p in range(5):
                                XXX[i,j,k,l,m,n,o,p] = XX[[i,j,k,l,m,n,o,p],
                                                          [0,1,2,3,4,5,6,7]]

XXX.shape = (-1,8)

np.save('XXX.npy', XXX)

#import timeit
#statement = '''
#bl.errorCuadraticoInt(XX[0], Ns, XextList0, params)
#'''
#t1 = timeit.timeit(statement, globals=globals(), number=100) / 1e2
#print('horas que va atardar', t1 * XXX.shape[0] / 3600)

#indprueba = np.random.randint(0, XXX.shape[0], 500)
YYY = [bl.errorCuadraticoInt(x, Ns, XextList0, params) for x in XXX]
np.save('YYY.npy', YYY)
prob = np.exp(-np.array(YYY))  # peso proporcional a la probabilidad,
# YYY = np.load('YYY.npy')
# XXX = np.load('XXX.npy')

XXX0 = np.average(XXX,axis=0,weights=prob)


coefs, exps = mpf.multipolyfit(XXX - XXX0, YYY, 2, powers_out=True)

# asume first of 9 element in exponent is for 1
const = coefs[0]
jac = coefs[1:9]
hes = np.zeros((8,8))
hes.T[np.triu_indices(8)] = hes[np.triu_indices(8)] = coefs[9:]


print(const)
print(jac)
print(hes)
print(hes==hes)
# testeo
test = [const + jac.dot(x) + x.dot(hes).dot(x)/2 for x in (XXX - XXX0)]
# deberia dar la identidad +- error de taylor
plt.plot(YYY, test, 'x')


# %% saco el vertice de la hiperparabola calculada
vert = XXX0 - jac.dot(ln.inv(hes))

# ahora recalculo la hiperparabola
coefs, exps = mpf.multipolyfit(XXX - vert, YYY, 2, powers_out=True)

# asume first of 9 element in exponent is for 1
const = coefs[0]
jac = coefs[1:9]
hes = np.zeros((8,8))
hes.T[np.triu_indices(8)] = hes[np.triu_indices(8)] = coefs[9:]

cova = ln.inv(hes)  # esta seria la covarianza asociada

# %% un ultimo refinamiento, sampleo puntos de la gaussiana calculada
# y con esos puntos recalculo el vertice y todo eso

u,s,v = ln.svd(cova)  # para diagonalizar la elipse de la gaussiana

XXsamples = vert + np.random.randn(1000, 8).dot(s*v)  # sampleo

for i in range(6):
    print(i)
    plt.figure()
    plt.scatter(XXsamples.T[i], XXsamples.T[i+1])

# calculo los errores de esos samples
YYsamples = [bl.errorCuadraticoInt(x, Ns, XextList0, params) for x in XXsamples]
prob = np.exp(-np.array(YYsamples))  # peso proporcional a la probabilidad

# saco el valor esperado
XXsamples0 = np.average(XXsamples,axis=0,weights=prob)

coefs, exps = mpf.multipolyfit(XXsamples - XXsamples0, YYsamples, 2, powers_out=True)

constSam = coefs[0]
jacSam = coefs[1:9]
hesSam = np.zeros((8,8))
hesSam[np.triu_indices(8)] = hesSam.T[np.triu_indices(8)] = coefs[9:]

print(constSam)
print(jacSam)
print(hesSam)
print(ln.eig(hesSam)[0])

# testeo
testSam = [constSam + jacSam.dot(x) + x.dot(hesSam).dot(x)/2 for x in (XXsamples - XXsamples0)]

plt.plot(YYsamples, testSam, 'x')

# saco el vertice de la hiperparabola calculada
vert = XXsamples0 - jac.dot(ln.inv(hes))
cova = ln.inv(hes)  # esta seria la covarianza asociada

# %% grafico el error en cada direccion para ver que sea suave y en que
# escalas hacer el fiteo
Ci = np.repeat([np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
params = [n, m, imagePoints, model, chessboardModel, Ci]

# pongo en forma flat los valores iniciales
Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs)
XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]

Nsam = 500
#                   0    1    2    3    4     5     6     7
anchos = np.array([7e0, 7e0, 2e1, 2e1, 6e-3, 4e-3, 2e-3, 1e-3])
rangos = np.array([Xint-anchos, Xint+anchos]).T


#np.linspace(0.5,1.5, Nsam)

xList = list()
yList = list()

for i in [1,2,3]:
    print(i)
    xSam = np.repeat([Xint], Nsam, axis=0)
    xSam[:, i] = np.linspace(rangos[i][0], rangos[i][1], Nsam)
    
    ySam = [bl.errorCuadraticoInt(x, Ns, XextList, params) for x in xSam]
    
    xList.append(xSam[:, i])
    yList.append(ySam)
    
    # fiteo una parabola
    p = np.polyfit(xSam[:, i], ySam, 2)
    ypoly = np.polyval(p, xSam[:, i])
    
    plt.figure(i*2)
    plt.title(i)
    plt.plot(xSam[:, i], ySam, 'b-')
    plt.plot(xSam[:, i], ypoly, 'g-')
    plt.plot([Xint[i], Xint[i]], [np.min(ySam), np.max(ySam)], '-r')
    
    plt.figure(i*2+1)
    plt.title(i)
    ysqrt = np.sqrt(ySam)
    ypolysqrt = np.sqrt(ypoly)
    plt.plot(xSam[:, i], ysqrt, 'b-')
    plt.plot(xSam[:, i], ypolysqrt, 'g-')
    plt.plot([Xint[i], Xint[i]], [np.min(ysqrt), np.max(ysqrt)], '-r')

xList = np.array(xList).T - Xint
yList = np.array(yList).T

#xListInf = np.min(xList, axis=0)
#xListSup = np.max(xList, axis=0)
#
#yListInf = np.min(yList)
#yListSup = np.max(yList, axis=0)
#
#plt.plot((xList - xListInf) / (xListSup - xListInf),
#         (yList - yListInf) / (yListSup - yListInf), '-+')


# %%
# import funcion para ajustar hiperparabola
from dev import multipolyfit as mpf

def coefs2mats(coefs, n=8):
    const = coefs[0]
    jac = coefs[1:n+1]
    hes = np.zeros((n,n)) # los diagonales aparecen solo una vez
    hes[np.triu_indices(n)] = hes.T[np.triu_indices(n)] = coefs[n+1:]
    
    hes[np.diag_indices(n)] *= 2
    return const, jac, hes


# %% para hacer en muchos hilos
def mapListofParams(X, Ns, XextList, params):
    global mapCounter
    global npts
    ret = list()
    for  x in X:
        ret.append(bl.errorCuadraticoInt(x, Ns, XextList, params))
        mapCounter.value += 1
        print("%d de %d calculos: %2.3f por 100; error %g" % 
              (mapCounter.value, npts, mapCounter.value/npts*100, ret[-1]))
    
    return np.array(ret)



def procMapParams(X, Ns, XextList, params, ret):
    ret.put(mapListofParams(X, Ns, XextList, params))



def mapManyParams(Xlist, Ns, XextList, params, nThre=4):
    '''
    function to map in many threads values of params to error function
    '''
    N = Xlist.shape[0] # cantidad de vectores a procesar
    procs = list()  # lista de procesos
    er = [Queue() for i in range(nThre)]  # donde comunicar el resultado
    retVals = np.zeros(N)  # 
    
    inds = np.linspace(0, N, nThre + 1, endpoint=True, dtype=int)
    #print(inds)
    
    for i in range(nThre):
        #print('vecores', Xlist[inds[i]:inds[i+1]].shape)
        p = Process(target=procMapParams,
                    args=(Xlist[inds[i]:inds[i+1]], Ns, XextList, params,
                          er[i]))
        procs.append(p)
        p.start()
    
    [p.join() for p in procs]

    for i in range(nThre):
        ret = er[i].get()
        #print('loque dio', ret.shape)
        #print('donde meterlo', retVals[inds[i]:inds[i+1]].shape)
        retVals[inds[i]:inds[i+1]] = ret
    
    
    
    return retVals



# %% aca testeo que tan rapido es con multithreading para 4 nucleos
import timeit
mapCounter=0

statement = '''mapManyParams(Xrand, Ns, XextList, params, nThre=8)'''

timss = list()
nss = [50, 100, 200, 500, 1000, 2000]
mapCounter = Value('i', 0)

for npts in nss:
    print(npts)
    Xrand = (np.random.rand(npts, 8) * 2 - 1) * anchos + Xint
    timss.append(timeit.timeit(statement, globals=globals(), number=5) / 5)
    print(npts, timss[-1])

plt.plot(nss, timss, '-*')

p = np.polyfit(nss, timss, 1)
p = np.array([ 0.0338636, -0.2550772])
npts = int(1e5)
print('tiempo ', np.polyval(p, npts) / 60, 'hs; ptos por eje ', npts**0.125)
# tomaria como 10hs hacer un milon de puntos

# %%
# genero muchas perturbaciones de los params intrinsecos alrededor de las cond
# iniciales en un rango de +-10% que por las graficas parece ser bastante
# parabolico el comportamiento todavía
npts = int(2e4)
mapCounter = Value('i', 0)
Xrand = (np.random.rand(npts, 8) * 2 - 1) * anchos + Xint
Yrand = mapManyParams(Xrand, Ns, XextList, params,  nThre=8)

np.save('Xrand', Xrand)
np.save('Yrand', Yrand)

# prob = np.exp(-Yrand / np.mean(Yrand))  # peso proporcional a la probabilidad

# saco el valor esperado
# Xrand0 = np.average(Xrand, axis=0) # , weights=prob)
# centro la cuenta en Xint
DeltaX = Xrand - Xint
coefs, exps = mpf.multipolyfit(DeltaX, Yrand, 2, powers_out=True)

constRan, jacRan, hesRan = coefs2mats(coefs)
np.real(ln.eig(hesRan)[0])

nConsidered = np.arange(100, npts+100, 100, dtype=int)
autovalsConverg = np.zeros((nConsidered.shape[0], 8))

indShuff = np.arange(Yrand.shape[0])
np.random.shuffle(indShuff)

Xrand = Xrand[indShuff]
Yrand = Yrand[indShuff]


for i, ncon in enumerate(nConsidered):
    print(i)
    coefs, exps = mpf.multipolyfit(DeltaX[:ncon], Yrand[:ncon], 2, powers_out=True)
    constRan, jacRan, hesRan = coefs2mats(coefs)
    autovalsConverg[i] = np.real(ln.eig(hesRan)[0])
    print(autovalsConverg[i])

plt.figure()
plt.plot(nConsidered, autovalsConverg)


## autovectores son las columnas
#w, vr = ln.eig(hesRan, left=False, right=True)
#np.dot(vr, ln.diagsvd(np.real(w),8,8).dot(ln.inv(vr)))
#
## right singular vectors as rows
#u, s, v = ln.svd(hesRan)
#
#np.allclose(hesRan, np.dot(u, np.dot(ln.diagsvd(s,8,8), v)))
#
#plt.matshow(np.log(np.abs(hesRan)))
#
#print(l)

# testeo
testRan = [constRan + jacRan.dot(x) + x.dot(hesRan).dot(x) / 2 for x in DeltaX]

plt.figure()
plt.plot(Yrand, testRan, 'xk')
plt.plot(Yrand, testRan, 'xr')
emin, emax = [np.min(Yrand), np.max(Yrand)]
plt.plot([emin, emax],[emin, emax])

## saco el vertice de la hiperparabola calculada
#vert = Xint - jacRan.dot(ln.inv(hesRan))
#cova = ln.inv(hesRan)  # esta seria la covarianza asociada

#hesChancho = vr.dot(ln.diagsvd(s,8,8)).dot(ln.inv(vr))
#
#testChan = [constRan + jacRan.dot(x) + x.dot(hesChancho).dot(x) / 2 for x in DeltaX]
#
#plt.figure()
#plt.plot(Yrand, testChan, 'xk')
#plt.plot(Yrand, testChan, 'xr')
#emin, emax = [np.min(Yrand), np.max(Yrand)]
#plt.plot([emin, emax],[emin, emax])
# no le pega ni por asomo

# %%

#from scipy.stats import multivariate_normal
#rv = multivariate_normal(mean=vert, cov=cova)
#
## make positive semidefinite, asume simetry
#print(ln.eig(hesRan)[0])
## https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
#A = hesRan
#B = (A + A.T) / 2
#u, s, v = ln.svd(B)
##% Compute the symmetric polar factor of B. Call it H.
##% Clearly H is itself SPD.
#H = v.dot(s).dot(v.T)
#Ahat = (A + H) / 2
#Ahat = (Ahat + Ahat.T) /2
#
#ln.cho_factor(Ahat)
#
#ln.eig(Ahat)[0]



# %% ahora miro en los autovectores del hessiano
#u, s, v = np.linalg.svd(hesRan)
l, v = ln.eig(hesRan)
s = np.sqrt(np.abs(np.real(l)))
npts = int(2e1)
etasI = np.linspace(-1e3, 1e3, npts)

i = 0
for i in range(8):
    direc = v[:,i]
    perturb = np.repeat([direc], npts, axis=0) * etasI.reshape((-1,1)) / s[i]
    Xmod = vert + perturb  # modified parameters
    Ymod = np.array([bl.errorCuadraticoInt(x, Ns, XextList, params) for x in Xmod])
    Xdist = ln.norm(perturb, axis=1) * np.sign(etasI)
    
    plt.figure()
    plt.plot(Xdist, Ymod, '-*')



# %% pruebo de calcular el hessiano poniendo paso fijo sabiendo la escala

#                   0    1    2    3    4     5     6     7
anchos = np.array([7e0, 7e0, 2e1, 2e1, 6e-3, 4e-3, 2e-3, 1e-3])

autovalores = list() # aca guardar la lista de autovalores

# en cuánto reducir los anchos especificados a ojo
factores = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500,
            1e3, 2e3, 5e3, 1e4, 2e4, 5e4]

for fact in factores:
    step = anchos/fact
    Hint = ndf.Hessian(bl.errorCuadraticoInt, step=step)
    hes1step = Hint(Xint, Ns, XextList, params)
    
    autovalores.append(np.real(ln.eig(hes1step)[0]))
    print(fact, autovalores[-1])

autovalores = np.array(autovalores)

for i,aut in enumerate(autovalores.T):
    plt.figure(i)
    plt.plot(factores, aut, '-*')

# %% estos son los rangos de step dond eveo que hay estabilidad
# o algo razonable
rangoNunDiffStep = np.array([[1e-3, 1e-1],    # 0
                             [1e-5, 1e-1],  # 1
                             [1e-3, 1e-1],   # 2
                             [1e-3, 1e-1],   # 3
                             [1e-3, 1e-1],     # 4
                             [1e-3, 1e-1],      # 5
                             [1e-3, 1e-1],      # 6
                             [1e-3, 1e-1]]).T * anchos

nLogSteps = 100  # cantidad de pasos en los que samplear el diff step
logFactor = (rangoNunDiffStep[1] / rangoNunDiffStep[0])**(1 / (nLogSteps-1))

# calculate list of steps
diffSteps3 = (rangoNunDiffStep[0].reshape((1,-1))
             * np.cumproduct([logFactor]*nLogSteps, axis=0)
             / logFactor)

autovalores3 = np.zeros_like(diffSteps3)

for i in range(nLogSteps):
    print('paso', i)
    
    Hint = ndf.Hessian(bl.errorCuadraticoInt, step=diffSteps3[i])
    hes1step = Hint(Xint, Ns, XextList, params)
    
    autovalores3[i] = np.real(ln.eig(hes1step)[0])
    print('autovalores', autovalores3[i])


# %%
'''
falta explorar un poco mas para algunos steps, haciendolo mas chico.
se nota que no es facil determinar el paso optimo, es bastante complicado
y quiza haya que terminar haciendo algo iterativo tipo metropolis.
comparar con lo que de el hessiano de la hiperparabola. si da parecido ya 
está.
'''

autovTotal = np.concatenate([autovalores[:73],
                             autovalores2,
                             autovalores3], axis=0)

diffStepsTotal =  np.concatenate([diffSteps[:73],
                             diffSteps2,
                             diffSteps3], axis=0)


np.save('autovaloresHess', autovTotal)
np.save('diffSteps', diffStepsTotal)

diffStepsTotal = np.load('diffSteps.npy')
autovTotal = np.load('autovaloresHess.npy')

for j in range(8):
    plt.figure()
    plt.title(j)
    plt.plot(diffStepsTotal[:,j], autovTotal[:,j], '-+')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid('on')

'''
en base a los graficos defino los steps con los que sacar el hessiano numerico
'''
stepsGraph = [0.3, 0.7, 2.5, 3.0, 1e-3, 4e-4, 4e-6, 3e-6]


Hint = ndf.Hessian(bl.errorCuadraticoInt, step=stepsGraph)
hesGraphStep = Hint(Xint, Ns, XextList, params)

np.real(ln.eig(hesGraphStep)[0])













