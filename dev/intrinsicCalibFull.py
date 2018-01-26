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
import os
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

NintrParams = distCoeffs.shape[0] + 4
NfreeParams = n*6 + NintrParams
NdataPoints = n*m

# 0.3pix as image std
stdPix = 0.1
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
#print(Erto.sum(), E0)

# saco distancia mahalanobis de cada proyeccion
mahDistance = bl.errorCuadraticoInt(Xint, Ns, XextList, params, mahDist=True)

#print(mahDistance)

# plot to check that tha proposed std is reasonable because of statistical
# behaviour
plt.figure()
nhist, bins, _ = plt.hist(mahDistance, 200, normed=True)
chi2pdf = sts.chi2.pdf(bins, 2)
plt.plot(bins, chi2pdf)
#plt.yscale('log')

# %% ========== initial approach based on iterative importance sampling
# in each iteration sample points from search space assuming gaussian pdf
# calculate total error and use that as log of probability

class sampleadorExtrinsecoNormal:
    '''
    manages the sampling of extrinsic parameters
    '''
    def __init__(self, muList, covarList):
        '''
        receive a list of rtvectors and define mu and covar and start sampler
        '''
        self.mean = np.array(muList)
        # gaussian normalizing factor
        self.piConst = (np.pi*2)**(np.prod(self.mean.shape)/2)
        self.setCov(covarList)

    def rvs(self, retPdf=False):
        x = sts.multivariate_normal.rvs(size=(len(self.matrixTransf),6))
        x2 = (x.reshape(-1,6,1) * self.matrixTransf).sum(2)
        
        if retPdf:
            pdf = np.exp(- np.sum(x**2) / 2) / self.piConst
            return x2 + self.mean, pdf
        
        return x2 + self.mean
    
    def setCov(self, covarList):
        self.cov = np.array(covarList)
        self.matrixTransf = np.array([cl.unit2CovTransf(x) for x in self.cov])


# %% 
'''
se propone una pdf de donde samplear. se trata de visualizar y de hacer un
ajuste grueso
'''

# propongo covarianzas de donde samplear el espacio de busqueda
Cfk = np.diag((Xint/1000)**2 + 1e-6)  # 1/1000 de desv est
# regularizado para que no sea singular

Crt = np.eye(6) # 6x6 covariance of pose
Crt[[0,1,2],[0,1,2]] *= np.deg2rad(1)**2 # 1 deg stdev in every angle
Crt[[3,4,5],[3,4,5]] *= 0.1**2 # 1/10 of the unit length as std for pos

# reduzco la covarianza en un factor (por la dimensionalidad??)
fkFactor = 1e-4
rtFactor = 1e-4
Cfk *= fkFactor
Crt = np.repeat([Crt * rtFactor] , n, axis=0)

# instancio los sampleadres
sampleadorInt = sts.multivariate_normal(Xint, Cfk)
sampleadorExt = sampleadorExtrinsecoNormal(XextList, Crt)
# evaluo
intSamp = sampleadorInt.rvs()
extSamp, pdfExtSamp = sampleadorExt.rvs(retPdf=True)

errSamp = etotal(intSamp, Ns, extSamp, params)

# trayectoria de medias
meanList = list()





# %% trato de hacer una especie de gradiente estocástico con momentum
# cond iniciales
Xint0 = dc(Xint)
Xext0 = dc(np.array(XextList))
Xerr0 = etotal(Xint0, Ns, Xext0, params)

beta = 0.9
beta1 = 1 - beta

deltaInt = np.zeros_like(Xint0)
deltaExt = np.zeros_like(Xext0)

sampleIntList = list([Xint0])
sampleExtList = list([Xext0])
sampleErrList = list([Xerr0])

# %% loop


for i in range(10000):
    print(i, "%.20f"%sampleErrList[-1])
    Xint1 = sampleadorInt.rvs()
    Xext1 = sampleadorExt.rvs()
    Xerr1 = etotal(Xint1, Ns, Xext1, params)
    
    if Xerr0 > Xerr1: # caso de que dé mejor
        print('a la primera')
        deltaInt = deltaInt * beta + beta1 * (Xint1 - Xint0)
        deltaExt = deltaExt * beta + beta1 * (Xext1 - Xext0)
        
        # salto a ese lugar
        Xint0 = Xint1
        Xext0 = Xext1
        Xerr0 = Xerr1
        
        sampleadorInt.mean = Xint0 + deltaInt
        sampleadorExt.mean = Xext0 + deltaExt
        sampleIntList.append(Xint0)
        sampleExtList.append(Xext0)
        sampleErrList.append(Xerr0)
        
    else: # busco para el otro lado a ver que dá
        Xint2 = 2 * Xint0 - Xint1
        Xext2 = 2 * Xext0 - Xext1
        Xerr2 = etotal(Xint2, Ns, Xext2, params)
        
        if Xerr0 > Xerr2: # caso de que dé mejor la segunda opcion
            print('a la segunda')
            deltaInt = deltaInt * beta + beta1 * (Xint2 - Xint0)
            deltaExt = deltaExt * beta + beta1 * (Xext2 - Xext0)
            
            # salto a ese lugar
            Xint0 = Xint2
            Xext0 = Xext2
            Xerr0 = Xerr2
            
            sampleadorInt.mean = Xint0 + deltaInt
            sampleadorExt.mean = Xext0 + deltaExt
            sampleIntList.append(Xint0)
            sampleExtList.append(Xext0)
            sampleErrList.append(Xerr0)
        else: # las dos de los costados dan peor
            ## mido la distancia hacia la primera opcion
            #dist = np.sqrt(np.sum((Xint1 - Xint0)**2) + np.sum((Xext1 - Xext0)**2))
            # distancia al vertice de la parabola
            r = (Xerr2 - Xerr1) / 2 / (Xerr1 + Xerr2 - 2 * Xerr0) #* dist / dist
            if np.isnan(r) or np.isinf(r):
                print('r is nan inf')
                sampleadorInt.cov *= 1.5  # agrando covarianzas
                sampleadorExt.setCov(sampleadorExt.cov * 1.5)
                continue # empiezo loop nuevo
            
            # calculo un nuevo lugar como vertice de la parabola en 1D
            Xint3 = Xint0 + (Xint1 - Xint0) * r
            Xext3 = Xext0 + (Xext1 - Xext0) * r
            Xerr3 = etotal(Xint3, Ns, Xext3, params)
            
            if Xerr0 > Xerr3:
                print('a la tercera')
                deltaInt = deltaInt * beta + beta1 * (Xint3 - Xint0)
                deltaExt = deltaExt * beta + beta1 * (Xext3 - Xext0)
                
                # salto a ese lugar
                Xint0 = Xint3
                Xext0 = Xext3
                Xerr0 = Xerr3
                
                sampleadorInt.mean = Xint0 + deltaInt
                sampleadorExt.mean = Xext0 + deltaExt
                sampleIntList.append(Xint0)
                sampleExtList.append(Xext0)
                sampleErrList.append(Xerr0)
            else:
                print('no anduvo, probar de nuevo corrigiendo')
                sampleadorInt.cov *= 0.9  # achico covarianzas
                sampleadorExt.setCov(sampleadorExt.cov * 0.9)
                
                deltaInt *= 0.9 # achico el salto para buscar mas cerca
                deltaExt *= 0.9
                
                sampleadorInt.mean = Xint0 + deltaInt
                sampleadorExt.mean = Xext0 + deltaExt
    


intrinsicGradientList = np.array(sampleIntList)
extrinsicGradientList = np.array(sampleExtList)
errorGradientList = np.array(sampleErrList)


plt.figure()
plt.plot(errorGradientList - errorGradientList[-1])


sampleadorIntBkp = dc(sampleadorInt)
sampleadorExtBkp = dc(sampleadorExt)

# %%
plt.figure()
minErr = np.min(errorGradientList)
for i in range(NintrParams):
    plt.plot(intrinsicGradientList[:,i] - intrinsicGradientList[-1,i],
             errorGradientList - minErr, '-x')

for i in range(n):
    for j in range(6):
        plt.plot(extrinsicGradientList[:,i,j] - extrinsicGradientList[-1,i,j],
                 errorGradientList - minErr, '-x')
plt.semilogy()



    # %% metropolis hastings
from numpy.random import rand as rn

def nuevo(oldInt, oldExt, oldErr, retPb=False):
    global generados
    global gradPos
    global gradNeg
    global mismo
    global sampleador

    # genero nuevo
    newInt = sampleadorInt.rvs() # rn(8) * intervalo + cotas[:,0]
    newExt = sampleadorExt.rvs()
    generados += 1

    # cambio de error
    newErr = etotal(newInt, Ns, newExt, params)
    deltaErr = newErr - oldErr

    if deltaErr < 0:
        gradPos += 1
        print(generados, gradPos, gradNeg, mismo, "Gradiente Positivo")
        if retPb:
            return newInt, newExt, newErr, 1.0
        return newInt, newExt, newErr # tiene menor error, listo
    else:
        # nueva oportunidad, sampleo contra rand
        pb = np.exp(- deltaErr / 2)

        if pb > rn():
            gradNeg += 1
            print(generados, gradPos, gradNeg, mismo, "Gradiente Negativo, pb=", pb)
            if retPb:
                return newInt, newExt, newErr, pb
            return newInt, newExt, newErr  # aceptado a la segunda oportunidad
        else:
#            # vuelvo recursivamente al paso2 hasta aceptar
#            print('rechazado, pb=', pb)
#            new, newE = nuevo(old, oldE)
            mismo +=1
            print(generados, gradPos, gradNeg, mismo, "Mismo punto,        pb=", pb)
            if retPb:
                return oldInt, oldExt, oldErr, pb
            return oldInt, oldExt, oldErr

#    return newInt, newExt, newErr

generados = 0
gradPos = 0
gradNeg = 0
mismo = 0


intSamp = sampleadorInt.rvs()
extSamp = sampleadorExt.rvs()
errSamp = etotal(intSamp, Ns, extSamp, params)

# pruebo una iteracion
newInt, newExt, newErr = nuevo(intSamp, extSamp, errSamp)


# %% metropolis para hacer burnin y sacar una covarianza para q

Nmuestras = 1000
#Mmuestras = int(50)
#nTot = Nmuestras * Mmuestras

generados = 0
gradPos = 0
gradNeg = 0
mismo = 0

sampleIntList = np.zeros((Nmuestras, distCoeffs.shape[0] + 4))
sampleExtList = np.zeros((Nmuestras,n,6))
sampleErrList = np.zeros(Nmuestras)

# cargo samplers
sampleadorInt = dc(sampleadorIntBkp)
sampleadorExt = dc(sampleadorExtBkp)

sampleadorInt.cov = Cfk
sampleadorExt.setCov(Crt / 1000) # no se porque hay que dividir por mil

# primera
sampleIntList[0] = sampleadorInt.rvs()
sampleExtList[0] = sampleadorExt.rvs()
sampleErrList[0] = etotal(intSamp, Ns, extSamp, params)

tiempoIni = time.time()

for i in range(1, Nmuestras):
    sampleIntList[i], sampleExtList[i], sampleErrList[i], pb = nuevo(sampleIntList[i-1], sampleExtList[i-1], sampleErrList[i-1], retPb=True)
    
    if np.isnan(pb):
        break
    
    # actualizo centroide
    sampleadorInt.mean = sampleIntList[i]
    sampleadorExt.mean = sampleExtList[i]
    
    tiempoNow = time.time()
    Dt = tiempoNow - tiempoIni
    frac = i / Nmuestras # (i  + Nmuestras * j)/ nTot
    DtEstimeted = (tiempoNow - tiempoIni) / frac
    stringTimeEst = time.asctime(time.localtime(tiempoIni + DtEstimeted))

    print('Epoch: %d/%d. Tfin: %s'
          %(i, Nmuestras, stringTimeEst),
          np.linalg.norm(sampleadorInt.cov),
          np.linalg.norm(sampleadorExt.cov))




corner.corner(sampleIntList)


os.system("speak 'aadfafañfañieñiweh'")
