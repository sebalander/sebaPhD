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
import scipy.stats as sts
import matplotlib.pyplot as plt
from importlib import reload
from copy import deepcopy as dc
import numdifftools as ndf
from dev import multipolyfit as mpf
from calibration import calibrator as cl
import corner
from dev import bayesLib as bl
import time

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

# model data files
distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

imageSelection = np.arange(33) # selecciono con que imagenes trabajar

# load data
imagePoints = np.load(cornersFile)[imageSelection]
n = len(imagePoints)  # cantidad de imagenes

chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
# images = glob.glob(imagesFolder+'*.png')

# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)
m = chessboardModel.shape[1]  # cantidad de puntos

# load model specific data
distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)
rVecs = np.load(rVecsFile)[imageSelection]
tVecs = np.load(tVecsFile)[imageSelection]


# pongo en forma flat los valores iniciales
Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs)
XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]


# standar deviation from subpixel epsilon used
std = 1.0

# output file
intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
intrinsicParamsOutFile = intrinsicParamsOutFile + str(std) + ".npy"

Ci = np.repeat([ std**2 * np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
params = dict()
params["n"] = n
params["m"] = m
params["imagePoints"] = imagePoints
params["model"] = model
params["chessboardModel"] = chessboardModel
params["Cccd"] = Ci
params["Cf"] = False
params["Ck"] = False
params["Crt"] = False
#params = [n, m, imagePoints, model, chessboardModel, Ci]

# pruebo con una imagen
j = 0
ErIm = bl.errorCuadraticoImagen(XextList[j], Xint, Ns, params, j, mahDist=False)
print(ErIm.sum())

# pruebo el error total
def etotal(Xint, Ns, XextList, params):
    '''
    calcula el error total como la suma de los errore de cada punto en cada
    imagen
    '''
    return bl.errorCuadraticoInt(Xint, Ns, XextList, params).sum()

Erto = bl.errorCuadraticoInt(Xint, Ns, XextList, params, mahDist=False)
E0 = etotal(Xint, Ns, XextList, params)
print(Erto.sum(), E0)

# saco distancia mahalanobis de cada proyeccion
mahDistance = bl.errorCuadraticoInt(Xint, Ns, XextList, params, mahDist=True)

plt.figure()
nhist, bins, _ = plt.hist(mahDistance, 50, normed=True)
chi2pdf = sts.chi2.pdf(bins, 2)
plt.plot(bins, chi2pdf)
plt.yscale('log')

# %% grafico para saber en que rangos hayq ue tomar en cuenta el error
'''
me quedo con una region tal que la probabilidad no sea mucho menor a 10^3 veces
la probabilidad del optimo de OCV. o sea que el error tiene que ser como mucho
Emax = E0 - 2 * np.log(1e-3)
donde la diferencia de error Emax - E0 ~ 14
'''
difE = 6 * np.log(10)
Emax = E0 + difE


# %%muevo en cada dirección buscando esa cota de error
Xint2 = dc(Xint)
Xint2[7] = -4.21e-03
E2 = etotal(Xint2, Ns, XextList, params)
print(difE, E2 - E0)




# %%
'''
encontre por prueba y error estos rangos:

Xint2[0] = [397.9 : 398.6]
Xint2[1] = [410.5 : 411.7]
Xint2[2] = [807.9 : 808.4]
Xint2[3] = [466.8 : 467.4]
Xint2[4] = [9.52e-02 : 9.7e-02]
Xint2[5] = [-1.78e-02 : -1.815e-02]
Xint2[6] = [1.704e-02 : 1.726e-02]
Xint2[7] = [-4.08e-03 : -4.22e-03]

Xint = 
array([  398.213410,   411.227681,   808.169868,
         467.122082,   9.58412207e-02,  -1.79782432e-02,
         1.71556081e-02,  -4.14991879e-03])
    '''
'''
# estas cotas son para desv estandar de 1pixel
cotas = np.array([[398.1, 398.34],          #  398.213410
                  [411.03, 411.42],         #  411.227681
                  [808.07, 808.27],         #  808.169868
                  [467.03, 467.21],         #  467.122082
                  [9.56e-02, 9.61e-02],     #  9.58412207e-02
                  [-1.815e-02, -1.78e-02],  #  -1.79782432e-02
                  [1.704e-02, 1.727e-02],   #  1.71556081e-02
                  [-4.08e-03, -4.22e-03]])  #  -4.14991879e-03
'''

## %%
## standar deviation from subpixel epsilon used
#std = 0.01
#
## output file
#intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
#intrinsicParamsOutFile = intrinsicParamsOutFile + str(std) + ".npy"
#
#Ci = np.repeat([ std**2 * np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
#params = [n, m, imagePoints, model, chessboardModel, Ci]
#
## %%estas cotas para 0.01pix
#cotas = np.array([[398.211, 398.21415],        # [0] 398.213410
#                  [411.2204, 411.2287],          # [1]  411.227681
#                  [808.16982, 808.171],        # [2]  808.169868
#                  [467.1217, 467.1224],         # [3]  467.122082
#                  [9.58345e-02, 9.584205e-02], # [4]  9.58412207e-02
#                  [-1.7981e-02, -1.7977e-02],  # [5]  -1.79782432e-02
#                  [1.71532e-02, 1.715599e-02],   # [6]  1.71556081e-02
#                  [-4.15123e-03, -4.14956e-03]])  # [7]  -4.14991879e-03
#
#errores = np.zeros((8,2))
#
#for i in range(8): #  [7]
#    Xint2 = dc(Xint)
#    Xint2[i] = cotas[i,0]
#    errores[i,0] = etotal(Xint2, Ns, XextList, params)
#
#    Xint2[i] = cotas[i,1]
#    errores[i,1] = etotal(Xint2, Ns, XextList, params)
#    print((errores[i] - E0) / difE)
#
##print((errores - E0) / difE)
#
#print(cotas[:,0] < Xint)
#
#print(Xint < cotas[:,1])
#
#
## %% ploteo en cada direccion, duplicando los intervalos si quiero
#cotasDuplicadas = ((cotas.T - Xint) * 2 + Xint).T
#
#Npts = 10
#etas = np.linspace(0,1,Npts).reshape((1,-1))
#
#intervalo = (cotasDuplicadas[:,1] - cotasDuplicadas[:,0])
#
#barreParams = intervalo.reshape((-1,1)) * etas
#barreParams += cotasDuplicadas[:,0].reshape((-1,1))
#
#ers = np.zeros((8, Npts))
#
#for i in range(8):
#    Xint2 = dc(Xint)
#    for j in range(Npts):
#        Xint2[i] = barreParams[i,j]
#        ers[i,j] = etotal(Xint2, Ns, XextList, params)
#        print(i,j,ers[i,j])
#
#paramsNormaliz = (barreParams.T - Xint) / intervalo
#
#plt.plot(paramsNormaliz, ers.T, '-+')
#
#
#'''
#me quedo con las cotas que defini como la region de donde sacar samples
#'''

# %% metropolis hastings
from numpy.random import rand as rn

def nuevo(old, oldE):
    global generados
    global aceptados
    global avance
    global retroceso
    global sampleador
    
    # genero nuevo
    new = sampleador.rvs() # rn(8) * intervalo + cotas[:,0]
    newE = etotal(new, Ns, XextList, params)
    generados += 1
    print(generados, aceptados, avance, retroceso)
    
    # cambio de error
    deltaE = newE - oldE
    
    if deltaE < 0:
        aceptados +=1
        avance += 1
        print("avance directo")
        return new, newE # tiene menor error, listo
    else:
        # nueva opoertunidad, sampleo contra rand
        pb = np.exp(- deltaE / 2)
         
        if pb > rn():
            aceptados += 1
            retroceso += 1
            print("retroceso, pb=", pb)
            return new, newE # aceptado a la segunda oportunidad
        else:
            # vuelvo recursivamente al paso2 hasta aceptar
            print('rechazado, pb=', pb)
            new, newE = nuevo(old, oldE)
    
    return new, newE

class pdfSampler:
    '''
    para samplear de la uniforme entre las cotas puestas a mano
    '''
    def __init__(self, a, ab):
        self.a = a
        self.ab = ab
    
    def rvs(self, n=False):
        if not n:
            return np.random.rand(8) * self.ab + self.a
        else:
            return np.random.rand(n,8) * self.ab + self.a



## %%
#sampleador = pdfSampler(cotas[:,0],  cotas[:,1] - cotas[:,0])
#
## saco 1000 muestras
#Xsamples = sampleador.rvs(1000)
#esamples = np.array([etotal(x, Ns, XextList, params) for x in Xsamples]) - E0
#probsamples = np.exp(-esamples/2)
#
## media pesada por la probabilidad
#xaverg = np.average(Xsamples, axis=0, weights=probsamples)
#eaverg = etotal(xaverg, Ns, XextList, params) - E0
#paverg = np.exp(-eaverg/2)
## covarianza pesada por la probabilidad
#xcovar = np.cov(Xsamples.T, ddof=0, aweights=probsamples)
#
#'''
#para el caso de 0.01pix no se puede seguir, se ve que la gaussian es tan
#delgada que la precisiond e maquina no le da para samplear de ahi. quedamos con este resultado de 1e3 muestras y sus promedi pesados pro la probabilidad:
#In [605]: xaverg
#Out[605]: 
#array([  3.98213026e+02,   4.11224737e+02,   8.08170193e+02,
#         4.67122212e+02,   9.58407914e-02,  -1.79776236e-02,
#         1.71549092e-02,  -4.14964878e-03])
#In [606]: xcovar
#Out[606]: 
#array([[  6.85800443e-08,   1.31206035e-07,  -1.45551229e-08,
#          2.44610849e-08,  -1.12849690e-10,   1.03414534e-11,
#         -4.66874789e-11,   4.70931149e-11],
#       [  1.31206035e-07,   2.64439268e-07,  -2.79405274e-08,
#          4.56025943e-08,  -2.31231060e-10,   1.13446555e-11,
#         -8.34053911e-11,   8.86781723e-11],
#       [ -1.45551229e-08,  -2.79405274e-08,   5.10576258e-09,
#         -5.13012253e-09,   2.10102671e-11,  -5.84561655e-12,
#          1.36781236e-11,  -1.10723016e-11],
#       [  2.44610849e-08,   4.56025943e-08,  -5.13012253e-09,
#          1.26867969e-08,  -4.77800981e-11,   5.14399464e-12,
#         -1.51466889e-11,   1.79526904e-11],
#       [ -1.12849690e-10,  -2.31231060e-10,   2.10102671e-11,
#         -4.77800981e-11,   2.62114897e-13,  -2.16803353e-15,
#          5.00418223e-14,  -7.56914305e-14],
#       [  1.03414534e-11,   1.13446555e-11,  -5.84561655e-12,
#          5.14399464e-12,  -2.16803353e-15,   1.59767172e-14,
#         -1.86103848e-14,   1.07514983e-14],
#       [ -4.66874789e-11,  -8.34053911e-11,   1.36781236e-11,
#         -1.51466889e-11,   5.00418223e-14,  -1.86103848e-14,
#          4.58597406e-14,  -3.46909652e-14],
#       [  4.70931149e-11,   8.86781723e-11,  -1.10723016e-11,
#          1.79526904e-11,  -7.56914305e-14,   1.07514983e-14,
#         -3.46909652e-14,   3.35212763e-14]])
#'''
#
## %%
## standar deviation from subpixel epsilon used
#std = 0.1
#
## output file
#intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
#intrinsicParamsOutFile = intrinsicParamsOutFile + str(std) + ".npy"
#
#Ci = np.repeat([ std**2 * np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
#params = [n, m, imagePoints, model, chessboardModel, Ci]
#
#E0 = etotal(Xint, Ns, XextList, params)
#
##estas cotas andan para 0.1pix
#cotas = np.array([[398.2, 398.225],        #  398.213410
#                  [411.2, 411.25],          #  411.227681
#                  [808.16, 808.18],        #  808.169868
#                  [467.112, 467.132],         #  467.122082
#                  [9.581e-02, 9.5865e-02], #  9.58412207e-02
#                  [-1.8e-02, -1.796e-02],  #  -1.79782432e-02
#                  [1.7143e-02, 1.7167e-02],   #  1.71556081e-02
#                  [-4.1575e-03, -4.143e-03]])  #  -4.14991879e-03
#errores = np.zeros((8,2))
#
#for i in range(8): #  [7]
#    Xint2 = dc(Xint)
#    Xint2[i] = cotas[i,0]
#    errores[i,0] = etotal(Xint2, Ns, XextList, params)
#
#    Xint2[i] = cotas[i,1]
#    errores[i,1] = etotal(Xint2, Ns, XextList, params)
#    print((errores[i] - E0) / difE)
#
##print((errores - E0) / difE)
#
#print(cotas[:,0] < Xint)
#
#print(Xint < cotas[:,1])
#
#
## %%
#sampleador = pdfSampler(cotas[:,0],  cotas[:,1] - cotas[:,0])
#
## saco 1000 muestras
#Xsamples = sampleador.rvs(1000)
#esamples = np.array([etotal(x, Ns, XextList, params) for x in Xsamples]) - E0
#probsamples = np.exp(-esamples/2)
#
## media pesada por la probabilidad
#xaverg = np.average(Xsamples, axis=0, weights=probsamples)
#eaverg = etotal(xaverg, Ns, XextList, params) - E0
#paverg = np.exp(-eaverg/2)
## covarianza pesada por la probabilidad
#xcovar = np.cov(Xsamples.T, ddof=0, aweights=probsamples)
#
#ln.det(xcovar)
#
## %%
#for i in range(8):
#    plt.figure()
#    plt.hist(Xsamples[:,i], weights=probsamples, normed=True)

covar0 = np.array([[  6.85800443e-08,   1.31206035e-07,  -1.45551229e-08,
          2.44610849e-08,  -1.12849690e-10,   1.03414534e-11,
         -4.66874789e-11,   4.70931149e-11],
       [  1.31206035e-07,   2.64439268e-07,  -2.79405274e-08,
          4.56025943e-08,  -2.31231060e-10,   1.13446555e-11,
         -8.34053911e-11,   8.86781723e-11],
       [ -1.45551229e-08,  -2.79405274e-08,   5.10576258e-09,
         -5.13012253e-09,   2.10102671e-11,  -5.84561655e-12,
          1.36781236e-11,  -1.10723016e-11],
       [  2.44610849e-08,   4.56025943e-08,  -5.13012253e-09,
          1.26867969e-08,  -4.77800981e-11,   5.14399464e-12,
         -1.51466889e-11,   1.79526904e-11],
       [ -1.12849690e-10,  -2.31231060e-10,   2.10102671e-11,
         -4.77800981e-11,   2.62114897e-13,  -2.16803353e-15,
          5.00418223e-14,  -7.56914305e-14],
       [  1.03414534e-11,   1.13446555e-11,  -5.84561655e-12,
          5.14399464e-12,  -2.16803353e-15,   1.59767172e-14,
         -1.86103848e-14,   1.07514983e-14],
       [ -4.66874789e-11,  -8.34053911e-11,   1.36781236e-11,
         -1.51466889e-11,   5.00418223e-14,  -1.86103848e-14,
          4.58597406e-14,  -3.46909652e-14],
       [  4.70931149e-11,   8.86781723e-11,  -1.10723016e-11,
          1.79526904e-11,  -7.56914305e-14,   1.07514983e-14,
         -3.46909652e-14,   3.35212763e-14]])

covar0 += np.eye(8) * 1e-9  # regulaizo un poco
# %% ahora arranco con Metropolis, primera etapa
# defino una pdf sacada de la galera a partir ed lo que me daba opencv
mu0 = Xint.copy()
#covar0 = np.diag(np.abs(mu0 * 1e-5))
sampleador = sts.multivariate_normal(mu0, covar0) # np.eye(8)*1e-4)

Nmuestras = int(1e3)

generados = 0
aceptados = 0
avance = 0
retroceso = 0

paraMuest = np.zeros((Nmuestras,8))
errorMuestras = np.zeros(Nmuestras)

# primera
start = sampleador.rvs() # dc(Xint)
startE = etotal(start, Ns, XextList, params)
paraMuest[0], errorMuestras[0] = nuevo(start, startE)

# primera parte saco 10 puntos asi como esta
for i in range(1, 20):
    paraMuest[i], errorMuestras[i] = nuevo(paraMuest[i-1], errorMuestras[i-1])
    sampleador.mean = paraMuest[i]

# ahora para hacerlo mas rapido voy actualizando la distribucion de propuesta
for i in range(20, 200):
    paraMuest[i], errorMuestras[i] = nuevo(paraMuest[i-1], errorMuestras[i-1])
    sampleador.mean = paraMuest[i]
    sampleador.cov = sampleador.cov * 0.7 + 0.3 * np.cov(paraMuest[i-10:i].T)

# ahora actualizo pesando por la probabilidad
for i in range(200, Nmuestras):
    paraMuest[i], errorMuestras[i] = nuevo(paraMuest[i-1], errorMuestras[i-1])
#    probMuestras[i] = np.exp(- errorMuestras[i] / 2)
    
    sampleador.mean = paraMuest[i]
    sampleador.cov = np.cov(paraMuest[100:i].T)


corner.corner(paraMuest)

mu1 = np.mean(paraMuest, 0)
covar1 = np.cov(paraMuest.T)

## %%
## saco la media pesada y la covarinza pesada
#esamples2 = np.array([etotal(x, Ns, XextList, params) for x in paraMuest]) - E0
#psamples2 = np.exp(- esamples2 / 2)
#mu2 = np.average(paraMuest, axis=0, weights=psamples2)
#covar2 = np.cov(paraMuest.T, ddof=0, aweights=psamples2)

# %% ultima etapa de metropolis
sampleador = sts.multivariate_normal(mu1, covar1)

Nmuestras = int(2e3)

generados = 0
aceptados = 0
avance = 0
retroceso = 0

paraMuest2 = np.zeros((Nmuestras,8))
errorMuestras2 = np.zeros(Nmuestras)

# primera
start = dc(Xint) # sampleador() # rn(8) * intervalo + cotas[:,0]
startE = etotal(start, Ns, XextList, params)
paraMuest2[0], errorMuestras2[0] = (start, startE)



tiempoIni = time.time()
for i in range(1, Nmuestras):
    paraMuest2[i], errorMuestras2[i] = nuevo(paraMuest2[i-1], errorMuestras2[i-1])
    sampleador.mean = paraMuest2[i]
    
    tiempoNow = time.time()
    Dt = tiempoNow - tiempoIni
    frac = i / Nmuestras
    DtEstimeted = (tiempoNow - tiempoIni) / frac
    stringTimeEst = time.asctime(time.localtime(tiempoIni + DtEstimeted))
    print('Transcurrido: %.2fmin. Avance %.4f. Tfin: %s'
          %(Dt/60, frac, stringTimeEst) )

mu2 = np.mean(paraMuest2, 0)
covar2 = np.cov(paraMuest2.T)

corner.corner(paraMuest2)


# %% ultima etapa de metropolis
sampleador = sts.multivariate_normal(mu2, covar2)

Nmuestras = int(2e3)

generados = 0
aceptados = 0
avance = 0
retroceso = 0

paraMuest3 = np.zeros((Nmuestras,8))
errorMuestras3 = np.zeros(Nmuestras)

# primera
start = dc(Xint) # sampleador() # rn(8) * intervalo + cotas[:,0]
startE = etotal(start, Ns, XextList, params)
paraMuest3[0], errorMuestras3[0] = (start, startE)



tiempoIni = time.time()
for i in range(1, Nmuestras):
    paraMuest3[i], errorMuestras3[i] = nuevo(paraMuest3[i-1], errorMuestras3[i-1])
    sampleador.mean = paraMuest3[i]
    
    tiempoNow = time.time()
    Dt = tiempoNow - tiempoIni
    frac = i / Nmuestras
    DtEstimeted = (tiempoNow - tiempoIni) / frac
    stringTimeEst = time.asctime(time.localtime(tiempoIni + DtEstimeted))
    print('Transcurrido: %.2fmin. Avance %.4f. Tfin: %s'
          %(Dt/60, frac, stringTimeEst) )

mu3 = np.mean(paraMuest3, 0)
covar3 = np.cov(paraMuest3.T)

corner.corner(paraMuest3)

# %% new estimated covariance run Metropolis again
mu3 = np.mean(paraMuest2, axis=0)
covar3 = np.cov(paraMuest2.T)
cameraMatrixOut, distCoeffsOut = bl.flat2int(mu3, Ns)

mu3Covar = covar3 / Nmuestras
covar3Covar = bl.varVarN(covar3, Nmuestras)

resultsML = dict()

resultsML['Nsamples'] = Nmuestras
resultsML['paramsMU'] = mu3
resultsML['paramsVAR'] = covar3
resultsML['paramsMUvar'] = mu3Covar
resultsML['paramsVARvar'] = covar3Covar
resultsML['Ns'] = Ns

# %%
save = False
if save:
    np.save(intrinsicParamsOutFile, resultsML)

load = False
if load:
    resultsML = np.load(intrinsicParamsOutFile).all()  # load all objects
    
    Nmuestras = resultsML["Nsamples"]
    mu3 = resultsML['paramsMU']
    covar3 = resultsML['paramsVAR']
    mu3Covar = resultsML['paramsMUvar']
    covar3Covar = resultsML['paramsVARvar']
    Ns = resultsML['Ns']
    
    cameraMatrixOut, distCoeffsOut = bl.flat2int(mu3, Ns)



# %%
import corner

# el error relativo aproximadamente
np.sqrt(np.diag(covar3)) / mu3

corner.corner(paraMuest2, 50)

print(np.concatenate([[xaverg], [mu2], [mu3]], axis=0).T)

# %% ahora para proyectar con los datos de chessboard a ver como dan

def sacoParmams(pars):
    r = pars[4:7]
    t = pars[7:10]
    d = pars[10:]
    
    if model is 'poly' or model is 'rational':
        d = np.concatenate([d[:2],np.zeros_like(d[:2]), d[2].reshape((1,-1))])
    
    if len(pars.shape) > 1:
        k = np.zeros((3, 3, pars.shape[1]), dtype=float)
    else:
        k = np.zeros((3, 3), dtype=float)
    k[[0,1,2,2],[0,1,0,1]] = pars[:4]
    k[2,2] = 1
    
    return r.T, t.T, k.T, d.T

def sdt2covs(covAll):
    Cf = np.diag(covAll[:4])
    Crt = np.diag(covAll[4:10])
    
    Ck = np.diag(covAll[10:])
    return Cf, Ck, Crt


j = 14 # elijo una imagen para trabajar

imagenFile = glob.glob(imagesFolder+'*.png')[j]
plt.figure()
plt.imshow(plt.imread(imagenFile), origin='lower')
plt.scatter(imagePoints[j,0,:,0], imagePoints[j,0,:,1])

nSampl = int(1e3)  # cantidad de samples

Cccd = Ci
Crt = np.zeros((6,6))

# meto los resultados de ML
Cf = covar3[:4,:4]
Ck = covar3[4:,4:]

nparams = 4 + Ck.shape[0] + 6

# dejo los valores preparados
parsGen = np.random.multivariate_normal(mu3, covar3, nSampl)
posIgen = np.random.multivariate_normal([0,0], Cccd[j,0], (nSampl, m))
posIgen += imagePoints[j,0].reshape((1,-1,2))
posMap = np.zeros_like(posIgen)


r = rVecs[j]
t = tVecs[j]

# %% TESTEO LOS TRES PASOS JUNTOS 
# hago todos los mapeos de Monte Carlo
for posM, posI, pars in zip(posMap, posIgen, parsGen):
    #r, t, k, d = sacoParmams(pars)
    k, d = bl.flat2int(pars, Ns)
    posM[:,0], posM[:,1], _ = cl.inverse(posI, r, t, k, d, model)


# %% saco media y varianza de cada nube de puntos
posMapMean = np.mean(posMap, axis=0)
dif = (posMap - posMapMean).T
posMapVar = np.mean([dif[0] * dif, dif[1] * dif], axis=-1).T



# %% mapeo propagando incerteza los valores con los que comparar
xmJ, ymJ, CmJ = cl.inverse(imagePoints[j,0], r, t, cameraMatrix, distCoeffs,
                           model, Cccd[j], Cf, Ck)

XJ = np.array([xmJ, ymJ]).T

# %% grafico
xm, ym = posMap.reshape(-1,2).T

fig = plt.figure()
ax = fig.gca()

ax.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1],
            marker='+', c='k', s=100)

cl.plotPointsUncert(ax, CmJ, xmJ, ymJ, 'k')

ax.scatter(xm, ym, marker='.', c='b', s=1)
cl.plotPointsUncert(ax, posMapVar, posMapMean[:,0], posMapMean[:,1], 'b')


# %% calculo la distancia mahalanobis entre los puntos proyectados y los reales
# saco distancia mahalanobis de cada proyeccion
mahDistance = bl.errorCuadraticoInt(mu3, Ns, XextList, params, mahDist=True)

plt.figure()
nhist, bins, _ = plt.hist(mahDistance, 200, normed=True)
chi2pdf = sts.chi2.pdf(bins, 2)
plt.plot(bins, chi2pdf)
plt.yscale('log')


# %% para ver de optimizar la pose de cada imagen poque quiza eso jutifica que
# dé tan lejos de lo esperado
from scipy.optimize import minimize

def extrError(Xext, Xint, Ns, params, j):
    return bl.errorCuadraticoImagen(Xext, Xint, Ns, params, j).sum()


XextOptiList = list()

for i in range(n):
    res = minimize(extrError, XextList[i],
                   args=(mu3, Ns, params, i), method='Powell')
    print(i,res.success, res.nit)
    XextOptiList.append(res.x)





# %%
# standar deviation set manually so that distribution
std = 0.5

# output file
intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
intrinsicParamsOutFile = intrinsicParamsOutFile + str(std) + ".npy"

Ci = np.repeat([ std**2 * np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
params = [n, m, imagePoints, model, chessboardModel, Ci]


Erto = bl.errorCuadraticoInt(Xint, Ns, XextList, params, mahDist=False)
E0 = etotal(Xint, Ns, XextList, params)
print(Erto.sum(), E0)

# %% saco distancia mahalanobis de cada proyeccion
mahDistance0 = bl.errorCuadraticoInt(Xint, Ns, XextList, params, mahDist=True)

plt.figure()
nhist, bins, _ = plt.hist(mahDistance0, 50, normed=True)
chi2pdf = sts.chi2.pdf(bins, 2)
plt.plot(bins, chi2pdf)
plt.yscale('log')


# %%
# estas cotas son para desv estandar de 1pixel
cotas = np.array([[398.16, 398.27],         # [0]  398.213410
                  [411.13, 411.32],         # [1]  411.227681
                  [808.125, 808.22],         # [2]  808.169868
                  [467.07, 467.17],         # [3]  467.122082
                  [9.571e-02, 9.597e-02],     # [4]  9.58412207e-02
                  [-1.807e-02, -1.79e-02],  # [5]  -1.79782432e-02
                  [1.71e-02, 1.721e-02],   # [6]  1.71556081e-02
                  [-4.187e-03, -4.115e-03]])  # [7]  -4.14991879e-03

errores = np.zeros((8,2))

for i in range(8): # [7]
    Xint2 = dc(Xint)
    Xint2[i] = cotas[i,0]
    errores[i,0] = etotal(Xint2, Ns, XextList, params)

    Xint2[i] = cotas[i,1]
    errores[i,1] = etotal(Xint2, Ns, XextList, params)
    print((errores[i] - E0) / difE)

#print((errores - E0) / difE)

print(cotas[:,0] < Xint)

print(Xint < cotas[:,1])


# %%
sampleador = pdfSampler(cotas[:,0],  cotas[:,1] - cotas[:,0])

# saco 1000 muestras de la uniforme
Xsamples = sampleador.rvs(1000)
esamples = np.array([etotal(x, Ns, XextList, params) for x in Xsamples]) - E0
probsamples = np.exp(-esamples/2)

# media pesada por la probabilidad
xaverg = np.average(Xsamples, axis=0, weights=probsamples)
eaverg = etotal(xaverg, Ns, XextList, params) - E0
paverg = np.exp(-eaverg/2)
# covarianza pesada por la probabilidad
xcovar = np.cov(Xsamples.T, ddof=0, aweights=probsamples)

ln.det(xcovar)

# %%
for i in range(8):
    plt.figure()
    plt.hist(Xsamples[:,i], weights=probsamples, normed=True)


# %% ahora arranco con Metropolis, primera etapa
Nmuestras = int(5e2)

generados = 0
aceptados = 0
avance = 0
retroceso = 0

sampleador = sts.multivariate_normal(xaverg, xcovar)

paraMuest = np.zeros((Nmuestras,8))
errorMuestras = np.zeros(Nmuestras)

# primera
start = sampleador.rvs() # dc(Xint)
startE = etotal(start, Ns, XextList, params)
paraMuest[0], errorMuestras[0] = nuevo(start, startE)

# primera parte saco 10 puntos asi como esta
for i in range(1, 10):
    paraMuest[i], errorMuestras[i] = nuevo(paraMuest[i-1], errorMuestras[i-1])

# ahora para hacerlo mas rapido voy actualizando la distribucion de propuesta
for i in range(10, Nmuestras):
    paraMuest[i], errorMuestras[i] = nuevo(paraMuest[i-1], errorMuestras[i-1])
    sampleador.cov = np.cov(paraMuest[:i].T)
    sampleador.mean = np.mean(paraMuest[:i],0)


for i in range(8):
    plt.figure()
    plt.hist(paraMuest[:,i],30)

# saco la media pesada y la covarinza pesada
esamples2 = np.array([etotal(x, Ns, XextList, params) for x in paraMuest]) - E0
psamples2 = np.exp(- esamples2 / 2)
mu2 = np.average(paraMuest, axis=0, weights=psamples2)
covar2 = np.cov(paraMuest.T, ddof=0, aweights=psamples2)

# %% ultima etapa de metropolis
sampleador = sts.multivariate_normal(mu2, covar2)

Nmuestras = int(1e4)

generados = 0
aceptados = 0
avance = 0
retroceso = 0

paraMuest2 = np.zeros((Nmuestras,8))
errorMuestras2 = np.zeros(Nmuestras)

# primera
start = dc(Xint) # sampleador() # rn(8) * intervalo + cotas[:,0]
startE = etotal(start, Ns, XextList, params)
paraMuest2[0], errorMuestras2[0] = (start, startE)



tiempoIni = time.time()
for i in range(1, Nmuestras):
    paraMuest2[i], errorMuestras2[i] = nuevo(paraMuest2[i-1], errorMuestras2[i-1])
    tiempoNow = time.time()
    Dt = tiempoNow - tiempoIni
    frac = i / Nmuestras
    DtEstimeted = (tiempoNow - tiempoIni) / frac
    stringTimeEst = time.asctime(time.localtime(tiempoIni + DtEstimeted))
    print('Transcurrido: %.2fmin. Avance %.4f. Tfin: %s'
          %(Dt/60, frac, stringTimeEst) )


for i in range(8):
    plt.figure()
    plt.hist(paraMuest2[:,i],30)


# %% new estimated covariance run Metropolis again
mu3 = np.mean(paraMuest2, axis=0)
covar3 = np.cov(paraMuest2.T)
cameraMatrixOut, distCoeffsOut = bl.flat2int(mu3, Ns)

mu3Covar = covar3 / Nmuestras
covar3Covar = bl.varVarN(covar3, Nmuestras)

resultsML = dict()

resultsML['Nsamples'] = Nmuestras
resultsML['paramsMU'] = mu3
resultsML['paramsVAR'] = covar3
resultsML['paramsMUvar'] = mu3Covar
resultsML['paramsVARvar'] = covar3Covar
resultsML['Ns'] = Ns

# %%
save = False
if save:
    np.save(intrinsicParamsOutFile, resultsML)

load = False
if load:
    resultsML = np.load(intrinsicParamsOutFile).all()  # load all objects
    
    Nmuestras = resultsML["Nsamples"]
    mu3 = resultsML['paramsMU']
    covar3 = resultsML['paramsVAR']
    mu3Covar = resultsML['paramsMUvar']
    covar3Covar = resultsML['paramsVARvar']
    Ns = resultsML['Ns']
    
    cameraMatrixOut, distCoeffsOut = bl.flat2int(mu3, Ns)



# %%
import corner

# el error relativo aproximadamente
np.sqrt(np.diag(covar3)) / mu3

corner.corner(paraMuest2, 50)

print(np.concatenate([[xaverg], [mu2], [mu3]], axis=0).T)



