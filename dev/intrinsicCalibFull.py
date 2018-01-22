#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 2018

do metropolis sampling to estimate PDF of chessboard calibration. this involves
intrinsic and extrinsic parameters, so it's a very high dimensional search
space (before it was only intrinsic)

@author: sebalander
"""

# %%
#import glob
import numpy as np
#import scipy.linalg as ln
import scipy.stats as sts
import matplotlib.pyplot as plt
#from importlib import reload
from copy import deepcopy as dc
#import numdifftools as ndf
from calibration import calibrator as cl
import corner
import time

import sys
sys.path.append("/home/sebalander/Code/sebaPhD")
#from dev import multipolyfit as mpf
from dev import bayesLib as bl
#from multiprocess import Process, Queue, Value
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

imageSelection = np.arange(0,33) # selecciono con que imagenes trabajar
n = len(imageSelection)  # cantidad de imagenes

# ## load data
imagePoints = np.load(cornersFile)[imageSelection]

chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
# images = glob.glob(imagesFolder+'*.png')

# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)
m = chessboardModel.shape[1]  # cantidad de puntos

# load model specific data from opencv Calibration
distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)
rVecs = np.load(rVecsFile)[imageSelection]
tVecs = np.load(tVecsFile)[imageSelection]


# pongo en forma flat los valores iniciales
Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs, model)
XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]

NfreeParams = n*6 + distCoeffs.shape[0] + 4
NdataPoints = n*m

# 0.3pix as image std
stdPix = 0.3
Ci = np.repeat([ stdPix**2 * np.eye(2)],n*m, axis=0).reshape(n,m,2,2)

#Cf = np.eye(distCoeffs.shape[0])
#Ck = np.eye(4)
#Cfk = np.eye(distCoeffs.shape[0], 4)  # rows for distortion coeffs
#
#Crt = np.eye(6) # 6x6 covariance of pose
#Crt[[0,1,2],[0,1,2]] *= np.deg2rad(1)**2 # 5 deg stdev in every angle
#Crt[[3,4,5],[3,4,5]] *= 0.01**2 # 1/100 of the unit length as std for pos
#Crt = np.repeat([Crt] , n, axis=0)
Crt = np.repeat([False], n) # no RT error

# output file
intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
intrinsicParamsOutFile = intrinsicParamsOutFile + str(stdPix) + ".npy"


params = dict()
params["n"] = n
params["m"] = m
params["imagePoints"] = imagePoints
params["model"] = model
params["chessboardModel"] = chessboardModel
params["Cccd"] = Ci
params["Cf"] = False # Cf
params["Ck"] = False # Ck
params["Cfk"] = False # Cfk
params["Crt"] = Crt  # list of 6x6 covariance matrices

# pruebo con una imagen
for j in range(n):
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

print(mahDistance)

# plot to check that tha proposed std is reasonable because of statistical
# behaviour
plt.figure()
nhist, bins, _ = plt.hist(mahDistance, 200, normed=True)
chi2pdf = sts.chi2.pdf(bins, 2)
plt.plot(bins, chi2pdf)
plt.yscale('log')


# %% ========== initial approach based on iterative importance sampling
# in each iteration sample points from search space assuming gaussian pdf
# calculate total error and use that as log of probability

# propongo covarianzas de donde samplear el espacio de busqueda
Cfk = np.diag((Xint/100)**2 + 1e-6)  # 1/100 de desv est
# regularizado para que no sea singular

Crt = np.eye(6) # 6x6 covariance of pose
Crt[[0,1,2],[0,1,2]] *= np.deg2rad(1)**2 # 5 deg stdev in every angle
Crt[[3,4,5],[3,4,5]] *= 0.01**2 # 1/100 of the unit length as std for pos
Crt = np.repeat([Crt] , n, axis=0)

sampleadorInt = sts.multivariate_normal(Xint, Cfk)

class sampleadorExtrinsecoNormal:
    '''
    manages the sampling of extrinsic parameters
    '''
    def __init__(self, mulist, covarList):
        '''
        receive a list of rtvectors and define mu and covar and start sampler
        '''
        self.muList = mulist
        self.matrixTransf = np.array([x for x in
                                      map(cl.unit2CovTransf, covarList)])

    def rvs(self):
        deltas = sts.multivariate_normal.rvs(size=(len(self.matrixTransf),6))
        deltas = (deltas.reshape(-1,6,1) * self.matrixTransf).sum(2)
        return deltas + self.muList
    
    def set(self, muList=None, covarList=None):
        if muList is not None:
            self.muList = muList
        if covarList is not None:
            self.matrixTransf = np.array([x for x in
                                      map(cl.unit2CovTransf, covarList)])


sampleadorExt = sampleadorExtrinsecoNormal(XextList, Crt)

intSamp = sampleadorInt.rvs()
extSamp = sampleadorExt.rvs()

etotal(intSamp, Ns, extSamp, params)


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

# %% ultima etapa de metropolis
# achico la covarianza para perder menos puntos y que haya uqe iterar menos
sampleador = sts.multivariate_normal(mu2, covar2)

Nmuestras = int(1e4)
Mmuestras = int(50)
nTot = Nmuestras * Mmuestras

generados = 0
gradPos = 0
gradNeg = 0
mismo = 0

paraMuest2 = np.zeros((Nmuestras,Xint.shape[0]))
errorMuestras2 = np.zeros(Nmuestras)

# primera
start = dc(Xint) # sampleador() # rn(8) * intervalo + cotas[:,0]
startE = etotal(start, Ns, XextList, params)
paraMuest2[0], errorMuestras2[0] = (start, startE)

tiempoIni = time.time()

for j in range(Mmuestras):

    for i in range(1, Nmuestras):
        paraMuest2[i], errorMuestras2[i] = nuevo(paraMuest2[i-1], errorMuestras2[i-1])
        sampleador.mean = paraMuest2[i]

        if i < 500: # no repito estas cuentas despues de cierto tiempo
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
