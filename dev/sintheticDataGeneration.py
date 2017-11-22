#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:03:56 2017

generate sinthetic data for simulation

@author: sebalander
"""

# %%
import glob
import numpy as np
from calibration import calibrator as cl
import matplotlib.pyplot as plt
import scipy.stats as sts
from copy import deepcopy as dc


import corner
import time
import cv2

import sys
sys.path.append("/home/sebalander/Code/sebaPhD")
#from dev import multipolyfit as mpf
from dev import bayesLib as bl

# %% LOAD DATA
# input
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye', 'stereographic']
model = modelos[3]

# %% parametros sinteticos de la camara
imgSize = (1600, 930)
pixmax = imgSize[0] / 2
fov = 183 * np.pi / 180 # field of view of the camera
thMax = fov / 2
k = pixmax / np.tan(thMax / 2)

# paramteros intrinsecos
mu = np.array([pixmax, imgSize[1] / 2, k])

# seed data files
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



# project from chessboard model to image asuming rvec, tvecs as True
corners = list()
for i in imageSelection:
    corners.append(cl.direct(objpoints[i], rVecs[i], tVecs[i],
                    cameraMatrix, distCoeffs, model, ocv=False))

corners = np.array(corners)

# standar deviation from subpixel epsilon used
# sumo ruid de std
std = 1.0
corners += np.random.randn(np.prod(corners.shape)).reshape(corners.shape) * std

'''
ahora que tengo los datos sinteticos deberia sacar condiciones iniciales con
opencv. esa deberia ser la idea. el tema es que ocv no tiene el modelo este.
algo como lo que hago en lo extrinseco, para cada modelo genero rvec, tvecs
'''

# pongo en forma flat los valores iniciales
Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs, model=model)
XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]

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



# %% metropolis hastings
from numpy.random import rand as rn

def nuevo(old, oldE):
    global generados
    global gradPos
    global gradNeg
    global mismo
    global sampleador

    # genero nuevo
    new = sampleador.rvs() # rn(8) * intervalo + cotas[:,0]
    generados += 1

    # cambio de error
    newE = etotal(new, Ns, XextList, params)
    deltaE = newE - oldE

    if deltaE < 0:
        gradPos += 1
        print("Gradiente Positivo")
        print(generados, gradPos, gradNeg, mismo)
        return new, newE # tiene menor error, listo
    else:
        # nueva opoertunidad, sampleo contra rand
        pb = np.exp(- deltaE / 2)

        if pb > rn():
            gradNeg += 1
            print("Gradiente Negativo, pb=", pb)
            print(generados, gradPos, gradNeg, mismo)
            return new, newE # aceptado a la segunda oportunidad
        else:
#            # vuelvo recursivamente al paso2 hasta aceptar
#            print('rechazado, pb=', pb)
#            new, newE = nuevo(old, oldE)
            mismo +=1
            print("Mismo punto,        pb=", pb)
            print(generados, gradPos, gradNeg, mismo)
            return old, oldE

    return new, newE
#


# %% cargo datos

mu0 = dc(Xint)
covar0 = np.diag(mu*5e-5)**2

# %% ultima etapa de metropolis
# achico la covarianza para perder menos puntos y que haya uqe iterar menos
sampleador = sts.multivariate_normal(mu0, covar0)

Nmuestras = int(1e4)
Mmuestras = int(50)
nTot = Nmuestras * Mmuestras

generados = 0
gradPos = 0
gradNeg = 0
mismo = 0

#paraMuest2 = np.zeros((Nmuestras,Xint.shape[0]))
#errorMuestras2 = np.zeros(Nmuestras)

paraMuest0 = list() # np.zeros((Nmuestras, Xint.shape[0]))
errorMuestras0 = list() # np.zeros(Nmuestras)

paraMuest0.append(Xint)
errorMuestras0.append(E0)
# primera
start = dc(Xint) # sampleador() # rn(8) * intervalo + cotas[:,0]
startE = etotal(start, Ns, XextList, params)
paraMuest0[0], errorMuestras0[0] = (start, startE)

mH = bl.metHas(Ns, XextList, params, sampleador)

# %% corro eta parte hasta que ve en el grafico que ya no se desplaza demasiado
Nini = 1000
for i in range(1, Nini):
    paM, erM = mH.nuevo(paraMuest0[-1], errorMuestras0[-1])
    print(paM)
    paraMuest0.append(paM)
    errorMuestras0.append(erM)
    
    mH.sampleador.mean = paraMuest0[-1]


# grafico para ver que haya pasado el burn in period
plt.figure()
plt.plot(paraMuest0 - paraMuest0[-1])

# %%


# largo desde la corrida que guarde antes
mu1 = paraMuest0[-1]
covar1 = dc(covar0)/100

sampleador = sts.multivariate_normal(mu1, covar1)
mH = bl.metHas(Ns, XextList, params, sampleador)

paM = sampleador.rvs()
erM = bl.etotalInt(paM, Ns, XextList, params)


paraMuest1 = list() # np.zeros((Nmuestras, Xint.shape[0]))
errorMuestras1 = list() # np.zeros(Nmuestras)
covMuestras1 = list()
covMuestras1.append(covar1)

paraMuest1.append(paM)
errorMuestras1.append(erM)

dims = Xint.shape[0]
Ntot = np.int(20.0**dims)

# segunda tanda de muestras solo cambiando el mean
# estas ya se usan en la tirada definitva
for i in range(1, Ntot):
    paM, erM = mH.nuevo(paraMuest1[-1], errorMuestras1[-1])
    paraMuest1.append(paM)
    errorMuestras1.append(erM)
    
    mH.sampleador.mean = paraMuest1[-1]
    covar1 = np.cov(np.array(paraMuest1).T)
    rango = np.linalg.matrix_rank(covar1)
    print(rango)
    if rango == dims:
        # leave this loop if covariance matrix has rank 6
        covMuestras1.append(covar1)
        break

# muestras cambiando el mean y covarianza, arranco desde donde se dejó
for i in range(i, Ntot):
    paM, erM = mH.nuevo(paraMuest1[-1], errorMuestras1[-1])
    paraMuest1.append(paM)
    errorMuestras1.append(erM)
    
    mH.sampleador.mean = paraMuest1[-1]
    mH.sampleador.cov = np.cov(np.array(paraMuest1).T)
    
    covMuestras1.append(sampleador.cov)



plt.figure()
covarianzasFlattened = np.array(covMuestras1).reshape((-1, dims**2))
plt.plot(covarianzasFlattened - covarianzasFlattened[-1])

paraMuest1 = np.array(paraMuest1)
corner.corner(paraMuest1,50)

plt.figure()
plt.plot(paraMuest1 - paraMuest1[-1])

dif = Xint -  np.mean(paraMuest1,axis=0)
mahDist = dif.dot(covMuestras1[-1]).dot(dif)
print('CDF de mahalanobis', sts.chi2.cdf(mahDist,df=3))




















# %%


tiempoIni = time.time()

for j in range(Mmuestras):

    for i in range(1, Nmuestras):
        paraMuest2[i], errorMuestras2[i] = nuevo(paraMuest2[i-1], errorMuestras2[i-1])
        sampleador.mean = paraMuest2[i]

        if True: # i < 50: # no repito estas cuentas despues de cierto tiempo
            tiempoNow = time.time()
            Dt = tiempoNow - tiempoIni
            frac = (i  + Nmuestras * j)/ nTot
            DtEstimeted = (tiempoNow - tiempoIni) / frac
            stringTimeEst = time.asctime(time.localtime(tiempoIni + DtEstimeted))

        print('Epoch: %d/%d-%d/%d. Transcurrido: %.2fmin. Avance %.4f. Tfin: %s'
              %(j,Mmuestras,i,Nmuestras,Dt/60, frac, stringTimeEst) )
    # guardo estos datos
    np.save("/home/sebalander/Documents/datosMHintrinsic2-%d"%j, paraMuest2)
    # para la proxima realizacion pongo la primera donde dejamos esta
    paraMuest2[0], errorMuestras2[0] = (paraMuest2[-1], errorMuestras2[-1])


# %% cargo los datos guardados durante la simulacion
metropHastingSamps = np.empty((Mmuestras, Nmuestras-1,8))

for j in range(Mmuestras):
    # no cargo el ultimo dato porque esta repetido en el primero del siguiente
    metropHastingSamps[j] = np.load("/home/sebalander/Documents/datosMHintrinsic2-%d.npy"%j)[:-1]

metropHastingSamps = metropHastingSamps.reshape((-1,8))

nSam = metropHastingSamps.shape[0]

corner.corner(metropHastingSamps,50)

for i in range(0,7):
    for j in range(i + 1,8):
        plt.figure()
        plt.hist2d(metropHastingSamps[:,i], metropHastingSamps[:,j],100)

mu = np.mean

results = dict()
results['Nsamples'] = nSam
results['paramsMU'] = np.mean(metropHastingSamps,0)
results['paramsVAR'] = np.cov(metropHastingSamps.T)
results['paramsMUvar'] = results['paramsVAR'] / nSam
results['paramsVARvar'] = bl.varVarN(results['paramsVAR'], nSam)
results['Ns'] = Ns

np.save("/home/sebalander/Documents/datosMHintrinsic2.npy", metropHastingSamps)

for i in range(8):
    plt.figure()
    plt.plot(metropHastingSamps[:,i],'.')

# %% veo de hacer pca para determinar el periodo burn in

from sklearn.decomposition import PCA


pca = PCA(n_components=2)
X_r = pca.fit(metropHastingSamps).transform(metropHastingSamps)
