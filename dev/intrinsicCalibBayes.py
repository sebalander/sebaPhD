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
#from importlib import reload
from copy import deepcopy as dc

from dev import bayesLib as bl

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






# %%
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
testearFunc = True
if testearFunc:
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
testearFunc = True
if testearFunc:
    # pruebo de evaluar jacobianos y hessianos
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

# %% grafico la función error en las direcciones de los autovalores del hessiano

# parametros auxiliares
params = [n, m, imagePoints, model, chessboardModel, Ci]

# pongo en forma flat los valores iniciales
Xint0, Ns = bl.int2flat(cameraMatrix, distCoeffs)
XextList0 = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]
e0 = bl.errorCuadraticoInt(Xint0, Ns, XextList0, params)

# saco los hessianos si no estan definidos
if 'hInt' not in locals():
    jInt, hInt, jExt, hExt = bl.jacobianos(Xint, Ns, XextList, params)

# get eigenvectors
u, s, v = np.linalg.svd(hInt)

etasLista = list()
erroresLista = list()

# %%
etasI = np.linspace(-1.0e-11, -2.5e-12, 1000)

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
plt.plot(etasI, errsI, '-*k')
plt.plot([-3e-9, 5e-9],[e0, e0], '-r')

np.savetxt('dev/firstComponentErrorFunct.txt', np.array([etasI, errsI]).T,
           header='moving by eta in the direction of the most significant eigvect of the hessian of the error function in the optimal point according to OpenCV.')


# %%
# parametros auxiliares
params = [n, m, imagePoints, model, chessboardModel]

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
intr, extr, ers = newtonOptE2(cameraMatrix, distCoeffs*1.001, rVecs, tVecs, Ns, params)
'''
este metodo no esta funcionando para esto. anduvo para la trilateracion pero
se ve que esta funcion de perdida es mas complicada y tiene cosas muy raras
'''

# %% solo gradiente descendente
# parametros auxiliares
params = [n, m, imagePoints, model, chessboardModel]

# pongo en forma flat los valores iniciales
Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs)
XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]


def gradDescE2(cameraMatrix, distCoeffs, rVecs, tVecs, Ns, params, ep=1e-4, eta=0.01):
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
    
    # mientras las correcciones sean mayores a un umbral
    while e > ep:
        Xint0[:] = Xint1
        XextList0[:] = XextList1
        
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
    
    return bl.flat2int(XextList1, Ns), [bl.ext2flat(x)for x in XextList1], errores

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
intr, extr, ers = gradDescE2(cameraMatrix, distCoeffs, rVecs, tVecs, Ns,
                             params, eta=1e-6)

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
#
#rangos = [,,1e-5]
#
#
u, s, v = np.linalg.svd(hInt)

# %% testeo en las diferentes direcciones segun el hesiano
etasI = np.linspace(-5e-8, 4e-8, int(5e2))

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
et = np.hstack([et, etasI])
er = np.hstack([er, errsI])

insort = np.argsort(et)

et = et[insort]
er = er[insort]

plt.figure()
plt.plot(et, er, '-*')
plt.plot([et[0], et[-1]],[e0, e0], '-r')

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

