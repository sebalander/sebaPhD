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

imageSelection = np.arange(0,33,4) # selecciono con que imagenes trabajar
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
trato de estimar la pdf usando importance weighting. parecido a importance
sampling pero adaptado para hacerlo medio iterativo. es heuristico solo para
hacer una aproximacion grosera
'''

# propongo covarianzas de donde samplear el espacio de busqueda
Cfk = np.diag((Xint/1000)**2 + 1e-6)  # 1/1000 de desv est
# regularizado para que no sea singular

Crt = np.eye(6) # 6x6 covariance of pose
Crt[[0,1,2],[0,1,2]] *= np.deg2rad(1)**2 # 1 deg stdev in every angle
Crt[[3,4,5],[3,4,5]] *= 0.1**2 # 1/10 of the unit length as std for pos

# reduzco notablemten la covarianza (por la dimensionalidad??)
fkFactor = 1e-2
rtFactor = 1e-2
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


# %%
Nsampl = 1000

sampleIntList = np.zeros((Nsampl, distCoeffs.shape[0] + 4))
sampleExtList = np.zeros((Nsampl,n,6))
samplePerr = np.zeros(Nsampl)
sampleQ = np.zeros(Nsampl)


for i in range(Nsampl):
    sampleIntList[i] = sampleadorInt.rvs()
    pdfIntSamp = sampleadorInt.pdf(intSamp)
    sampleExtList[i], pdfExtSamp = sampleadorExt.rvs(retPdf=True)
    samplePerr[i] = etotal(sampleIntList[i], Ns, sampleExtList[i], params)
    
    sampleQ[i] = pdfIntSamp*pdfExtSamp
    
    print(i, samplePerr[i])



# con este sampleo me fijo si el histograma de los errores es razonable
plt.figure()
plt.hist(samplePerr,100,normed=True)
plt.title("fk:%g  -- rt:%g"%(fkFactor,rtFactor))
# parece qeu anda asia uq euso tambien estos datos para refinar un poco la pdf


# calculo los importance weights
pHat = np.exp( - (samplePerr - np.min(samplePerr)) / 2)
r = pHat / sampleQ
w = r / np.sum(r)

# nueva pdf
#newIntMu = np.average(sampleIntList, axis=0, weights=w)
#newExtMu = np.average(sampleExtList, axis=0, weights=w)
bestSample = np.argmax(w)
sampleadorInt.mean = sampleIntList[bestSample]
sampleadorExt.mean = sampleExtList[bestSample]

sampleadorIntBkp = dc(sampleadorInt)
sampleadorExtBkp = dc(sampleadorExt)

##if np.sum(w != 0) != 1: # actualizo la covarianza
#print('calculo nueva covarianza segun los pesos')
#sampleadorInt.cov = np.cov(sampleIntList.T, ddof=0, aweights=w)
#
#newCovarList = [np.cov(sampleExtList[:,i].T, ddof=0, aweights=w) for i in range(n)]

## regularizo las nuevas covarianzas
#for c in newCovarList:
#    crank = np.linalg.matrix_rank(c)
#    if crank != 6:
#        u,s,v = np.linalg.svd(c)
#        print(s)
#        c2 = c + np.eye(6) * 1e-15
#        u,s2,v = np.linalg.svd(c2)
#        print(s2)
#        print(crank,np.linalg.matrix_rank(np.eye(6)))
#        c=c2
#
#sampleadorExt.setCov(np.array(newCovarList))


#else: # achico en 10 la escala
#    print('achico escala de covarianza')
#    sampleadorInt.cov /= 10
#    sampleadorExt.matrixTransf /= 100





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

Nmuestras = 20000
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

# primera
sampleIntList[0] = sampleadorInt.rvs()
sampleExtList[0] = sampleadorExt.rvs()
sampleErrList[0] = etotal(intSamp, Ns, extSamp, params)

tiempoIni = time.time()

#for j in range(Mmuestras):
for i in range(1, 1000):
    sampleIntList[i], sampleExtList[i], sampleErrList[i], pb = nuevo(sampleIntList[i-1], sampleExtList[i-1], sampleErrList[i-1], retPb=True)
    
    if np.isnan(pb):
        break
    
    # actualizo centroide
    sampleadorInt.mean = sampleIntList[i]
    sampleadorExt.mean = sampleExtList[i]
    
    # actualizo covarianzas
    deltaInt = sampleIntList[i] - sampleIntList[i-1]
    deltaIntCov = deltaInt.reshape((-1,1)) * deltaInt.reshape((1,-1))
    sampleadorInt.cov = sampleadorInt.cov * 0.99 + 0.01 * deltaIntCov
    
    deltaExt = sampleExtList[i] - sampleExtList[i-1]
    deltaExtCov = deltaExt.reshape((-1,6,1)) * deltaExt.reshape((-1,1,6))
    sampleadorExt.setCov(sampleadorExt.cov * 0.99 + 0.01 * deltaExtCov)

    tiempoNow = time.time()
    Dt = tiempoNow - tiempoIni
    frac = i / Nmuestras # (i  + Nmuestras * j)/ nTot
    DtEstimeted = (tiempoNow - tiempoIni) / frac
    stringTimeEst = time.asctime(time.localtime(tiempoIni + DtEstimeted))

    print('Epoch: %d/%d. Tfin: %s'
          %(i, Nmuestras, stringTimeEst),
          np.linalg.norm(sampleadorInt.cov),
          np.linalg.norm(sampleadorExt.cov))


indexInterval = 2*NfreeParams
# matrix para regularizar la covarianza de intrinsecos
regIntr = np.eye(NintrParams) * 1e-12
regExtr = np.eye(6) * 1e-12

for i in range(1000, Nmuestras):
    sampleIntList[i], sampleExtList[i], sampleErrList[i], pb = nuevo(sampleIntList[i-1], sampleExtList[i-1], sampleErrList[i-1], retPb=True)
    
    if np.isnan(pb):
        break
    
    # actualizo centroide
    sampleadorInt.mean = sampleIntList[i]
    sampleadorExt.mean = sampleExtList[i]
    
    # actualizo covarianzas
#    newCov = np.cov(sampleIntList[i-indexInterval:i].T)
#    if np.linalg.matrix_rank(newCov) < NintrParams:
#        newCov += regIntr # regularizo si pierde rango
#    sampleadorInt.cov = newCov
#
##    print('intervalo',i,indexInterval)
#    
#    meanExt = np.mean(sampleExtList[i-indexInterval:i], axis=0)
#    deltaExt = sampleExtList[i-indexInterval:i] - meanExt
#    deltaExtCov = (deltaExt.reshape((indexInterval, n, 6, 1)) *
#                   deltaExt.reshape((indexInterval, n, 1, 6)))
#    
#    newExtCov = deltaExtCov.sum(0) / indexInterval
#    ranks = np.linalg.matrix_rank(newExtCov)
#    newExtCov[ranks < 6] += regExtr # regularizo si perdieron rango
#    
#    sampleadorExt.setCov(newExtCov)
    
        # actualizo covarianzas
    deltaInt = sampleIntList[i] - sampleIntList[i-1]
    deltaIntCov = deltaInt.reshape((-1,1)) * deltaInt.reshape((1,-1))
    sampleadorInt.cov = sampleadorInt.cov * 0.9999 + 0.0001 * deltaIntCov
    
    deltaExt = sampleExtList[i] - sampleExtList[i-1]
    deltaExtCov = deltaExt.reshape((-1,6,1)) * deltaExt.reshape((-1,1,6))
    sampleadorExt.setCov(sampleadorExt.cov * 0.9999 + 0.0001 * deltaExtCov)

    
    
    if i%2==0: # cada dos iteraciones aumento el ancho de la ventana movil
#        print('aumento intervalo')
        indexInterval += 1


#    if i < 500: # no repito estas cuentas despues de cierto tiempo
    tiempoNow = time.time()
    Dt = tiempoNow - tiempoIni
    frac = i / Nmuestras # (i  + Nmuestras * j)/ nTot
    DtEstimeted = (tiempoNow - tiempoIni) / frac
    stringTimeEst = time.asctime(time.localtime(tiempoIni + DtEstimeted))

    print('Epoch: %d/%d. Tfin: %s'
          %(i, Nmuestras, stringTimeEst),
          np.linalg.norm(sampleadorInt.cov),
          np.linalg.norm(sampleadorExt.cov))


sampleadorIntBkp2 = dc(sampleadorInt)
sampleadorExtBkp2 = dc(sampleadorExt)

os.system("speak 'aadfafañfañieñiweh'")

# %%
rescaled = np.concatenate([sampleIntList, sampleExtList.reshape((Nmuestras,-1))], axis=1)
rescaledMin = np.min(rescaled, axis=0)
rescaledMax = np.max(rescaled, axis=0)

rescaled = (rescaled - rescaledMin) / (rescaledMax - rescaledMin)
indices = np.arange(Nmuestras)

# %%
Nsubplots = 14  # cantidad de subplots
fig, axes = plt.subplots(1, Nsubplots, sharey='row')
# Remove horizontal space between axes
fig.subplots_adjust(wspace=0, hspace=0,left=None, bottom=0, right=1, top=1)

axes[0].plot(rescaled[:,0::Nsubplots], indices)
axes[0].set_xticklabels([])

for i in range(1,Nsubplots):
    axes[i].plot(rescaled[:,i::Nsubplots], indices)
    axes[i].set_xticklabels([])
#fig.tight_layout()

'''
ahora pareciera que esta sampleando de manera mas razonable.
'''

# %%

# %% metropolis

Nmuestras = 10000
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
sampleadorInt = dc(sampleadorIntBkp2)
sampleadorExt = dc(sampleadorExtBkp2)

# primera
sampleIntList[0] = sampleadorInt.rvs()
sampleExtList[0] = sampleadorExt.rvs()
sampleErrList[0] = etotal(intSamp, Ns, extSamp, params)

tiempoIni = time.time()

#for j in range(Mmuestras):
for i in range(1, Nmuestras):
    sampleIntList[i], sampleExtList[i], sampleErrList[i], pb = nuevo(sampleIntList[i-1], sampleExtList[i-1], sampleErrList[i-1], retPb=True)
    
    if np.isnan(pb):
        break
    
    # actualizo centroide
    sampleadorInt.mean = sampleIntList[i]
    sampleadorExt.mean = sampleExtList[i]
    
    # no actualizo covarianzas

    tiempoNow = time.time()
    Dt = tiempoNow - tiempoIni
    frac = i / Nmuestras # (i  + Nmuestras * j)/ nTot
    DtEstimeted = (tiempoNow - tiempoIni) / frac
    stringTimeEst = time.asctime(time.localtime(tiempoIni + DtEstimeted))

    print('Epoch: %d/%d. Transcurrido: %.2fmin. Avance %.4f. Tfin: %s'
          %(i, Nmuestras, Dt/60, frac, stringTimeEst) )








# %%
'''
parece que da bien, ya estamos lejos del burn in. aca guardo para arrancar de
aca la proxima:

sampleadorInt.mean = np.array([  4.00830443e+02,   4.11000890e+02,   8.08200736e+02,
                                 4.67206963e+02,   9.42249003e-02,  -1.68690746e-02,
                                 1.60233632e-02,  -3.38643120e-03])

sampleadorInt.cov = np.array([[  2.84777073e-25,   4.50291856e-25,   2.25925198e-25,
                                  2.38707325e-25,  -1.13124870e-29,  -7.96742986e-30,
                                  1.37889181e-29,  -3.92082475e-30],
                               [  4.50291856e-25,   7.23472889e-25,   3.38178591e-25,
                                  3.85859123e-25,  -2.34793049e-29,  -1.46818722e-29,
                                  2.25574649e-29,  -6.25611895e-30],
                               [  2.25925198e-25,   3.38178591e-25,   2.11464173e-25,
                                  1.75071106e-25,   3.00330618e-31,  -2.77267436e-30,
                                  9.68842750e-30,  -3.02308525e-30],
                               [  2.38707325e-25,   3.85859123e-25,   1.75071106e-25,
                                  2.06504282e-25,  -1.35827622e-29,  -8.26488814e-30,
                                  1.21077311e-29,  -3.32346286e-30],
                               [ -1.13124870e-29,  -2.34793049e-29,   3.00330618e-31,
                                 -1.35827622e-29,   3.17839499e-33,   1.33115267e-33,
                                 -9.15273955e-34,   1.83366709e-34],
                               [ -7.96742986e-30,  -1.46818722e-29,  -2.77267436e-30,
                                 -8.26488814e-30,   1.33115267e-33,   6.15889446e-34,
                                 -5.22081140e-34,   1.18857600e-34],
                               [  1.37889181e-29,   2.25574649e-29,   9.68842750e-30,
                                  1.21077311e-29,  -9.15273955e-34,  -5.22081140e-34,
                                  7.17396925e-34,  -1.93627453e-34],
                               [ -3.92082475e-30,  -6.25611895e-30,  -3.02308525e-30,
                                 -3.32346286e-30,   1.83366709e-34,   1.18857600e-34,
                                 -1.93627453e-34,   5.43454045e-35]])

sampleadorExt.mean = np.array([[ -2.34131298e-01,   1.03418388e-01,  -7.35788326e-02,
                                 -3.16086149e+00,  -5.38339371e+00,   6.26006209e+00],
                               [  9.19247621e-01,   5.82316022e-02,  -7.78247678e-02,
                                 -3.03675245e+00,  -4.08710762e+00,   3.99647732e+00],
                               [  9.19748560e-01,  -3.26328271e-01,   9.94304690e-01,
                                  9.60992797e+00,  -5.73962184e+00,   3.90133094e+00],
                               [  1.54398490e-01,  -5.60396345e-01,   6.35913923e-01,
                                 -6.42575717e+00,  -4.00407574e+00,   3.73784223e+00],
                               [ -1.03769975e-02,   8.83892343e-01,  -5.38625746e-03,
                                 -1.05919547e-01,  -2.44542492e+00,   7.63644883e+00],
                               [  9.19014966e-01,   5.33666029e-01,   8.95467502e-01,
                                  7.81670985e+00,  -6.70147656e+00,   5.22848831e+00],
                               [  7.35171403e-01,   9.21546282e-01,   1.78199054e+00,
                                  1.86248753e+01,   5.99688683e+00,   7.28989790e+00],
                               [ -8.06483567e-01,   8.67808451e-02,   1.11194830e+00,
                                 -4.98084059e+00,  -3.50888683e+00,   1.14557529e+01],
                               [  1.72329384e-02,   1.24150305e-02,  -1.58814639e+00,
                                 -2.35436281e+00,   4.91907132e+00,   5.37140609e+00],
                               [ -9.97057628e-02,  -9.16056255e-03,  -3.07540385e+00,
                                  4.01327520e+00,   4.88368008e+00,   5.55660718e+00],
                               [  2.11925167e-01,   7.70639930e-02,   2.36317385e-02,
                                 -3.80300945e+00,   1.98446190e+00,   5.14341824e+00],
                               [  5.64751128e-01,   2.17464399e-01,   2.27481182e-01,
                                 -4.72242702e-01,  -1.98042934e+01,   1.48829820e+01],
                               [ -3.07427601e-02,  -3.05242078e-02,  -1.69982188e-02,
                                 -4.58975183e+00,  -5.74671917e-01,   1.97725018e+01],
                               [ -1.40459131e+00,   1.43567052e-02,   2.79116409e+00,
                                 -1.75823776e+01,  -2.94860490e+00,   1.23502826e+01],
                               [ -1.42907216e-01,  -1.05159994e+00,   1.69376307e+00,
                                 -1.69690264e+00,  -5.06812024e+00,   7.90852123e+00],
                               [ -4.17616998e-01,  -5.00029189e-01,   1.35904002e+00,
                                 -5.02952999e+00,  -5.42828007e+00,   1.26989741e+01],
                               [ -2.35210805e-01,   7.96081456e-01,   1.71337049e+00,
                                  1.48243992e+00,   1.60880711e+01,   2.09456805e+01],
                               [ -2.62870870e-01,   1.02528810e+00,   6.41328780e-01,
                                  1.68348198e+01,   6.70149151e+00,   1.73052333e+01],
                               [ -9.15995231e-03,   1.12989678e-01,  -1.59165241e-01,
                                 -2.18629120e+00,  -1.01304500e+01,   1.87036751e+01],
                               [ -9.62528305e-03,  -2.42512832e-01,  -1.76577363e-01,
                                 -9.52840373e+00,  -7.73684671e-01,   2.41853856e+00],
                               [  3.58513680e-01,   1.05811328e+00,   1.43700542e+00,
                                  1.22161187e+01,  -4.13270618e+00,   7.53694734e+00],
                               [  6.35770470e-01,  -6.85678709e-03,  -2.97832550e+00,
                                  3.67005914e+00,   5.44393040e-01,   1.45578946e+01],
                               [ -5.19639732e-02,  -5.94714856e-01,   2.83329467e-01,
                                 -1.12228972e+01,   1.66171894e+00,   3.24701236e+00],
                               [  2.50080941e-02,  -1.25611285e+00,   1.36720709e-02,
                                 -5.27382024e+00,  -2.39662203e+00,   1.42777025e+00],
                               [ -1.19324105e+00,   2.39090211e-01,   2.53830922e+00,
                                 -1.73814687e+01,   7.20412671e+00,   1.22964638e+01],
                               [ -1.02023769e+00,   6.23090162e-01,   2.05844564e+00,
                                 -1.08866229e+01,   1.41217052e+01,   1.53530795e+01],
                               [ -9.74332555e-02,  -8.23053354e-01,  -2.92085047e+00,
                                  1.47988087e+00,   1.61368496e+01,   1.23589382e+01],
                               [  7.34588193e-01,   4.69070380e-01,   1.34190324e+00,
                                  9.36026456e+00,  -6.56060477e+00,   4.39680591e+00],
                               [ -6.49139711e-01,  -1.26015047e-01,  -3.01090607e+00,
                                  1.92397959e+00,   7.00984795e+00,   1.01457467e+01],
                               [  3.91235225e-02,  -2.29144921e-02,  -7.63660280e-02,
                                 -3.89691361e+00,  -2.70071017e+00,   4.33726887e+00],
                               [  3.72740096e-01,   5.95999980e-01,  -2.88472639e-01,
                                  1.38716339e+01,  -9.09133098e+00,   1.55872224e+01],
                               [  3.40742740e-02,   8.92743000e-01,   1.53193264e-02,
                                  1.74191535e+01,  -2.79191974e+00,   1.07848992e+01],
                               [ -4.13942602e-02,   1.08076915e+00,   5.02421689e-01,
                                  1.81379941e+01,   1.97141547e+00,   1.03298217e+01]])

sampleadorExt.matrixTransf = np.array([[[ -1.10160141e-10,   2.16564051e-11,  -3.57633016e-13,
           9.70660129e-15,  -9.09808651e-17,   6.06477114e-19],
        [  3.22495601e-13,   8.39054638e-14,   2.09769344e-14,
           2.01724452e-14,  -5.56332988e-14,  -1.68159604e-15],
        [  4.83680807e-11,   4.86187218e-11,   2.31587107e-13,
           1.92179184e-14,   3.88837581e-16,  -1.56943383e-17],
        [  1.31337903e-13,   1.09175740e-13,   3.75115731e-15,
           9.29488944e-16,  -9.12313599e-15,   1.02785788e-14],
        [ -3.35612069e-12,   3.03702780e-12,   3.02045160e-13,
          -3.47085640e-13,  -3.11772453e-15,  -6.97625191e-17],
        [  2.22417175e-11,   1.98858818e-12,  -2.22967877e-12,
          -4.63877019e-14,  -9.06114585e-16,  -9.70592624e-18]],

       [[  1.74946932e-12,  -6.99521629e-13,  -5.47243010e-14,
           1.53266425e-13,   6.52870517e-14,  -1.26420975e-15],
        [  6.80789220e-11,  -2.98378438e-11,   3.60775723e-15,
          -3.21503857e-15,  -1.36736133e-15,  -8.31428806e-17],
        [  1.84366400e-10,   1.10221042e-11,  -1.82740419e-15,
           1.94015312e-16,  -7.28045906e-16,   3.20559238e-17],
        [  1.38852135e-12,   3.12206246e-13,   1.28431041e-13,
          -6.54998156e-14,   8.31786906e-14,   9.69521881e-16],
        [ -3.04960776e-14,   1.06417587e-13,  -1.57935582e-14,
          -2.42256421e-14,   6.13366858e-15,  -2.65210040e-14],
        [ -5.09349831e-14,  -3.16311560e-13,  -1.61525690e-13,
          -1.01711078e-13,   4.33953414e-14,   3.79012268e-15]],

       [[  4.87649424e-13,  -1.83939260e-12,   1.99937387e-13,
          -8.98528084e-14,  -2.45950229e-15,   7.76926624e-18],
        [ -7.43245223e-12,  -5.42370612e-12,  -1.48588915e-14,
          -1.14449182e-14,  -5.89578629e-15,   6.08873874e-18],
        [  6.75896466e-14,  -5.96212911e-10,  -1.04589688e-15,
           2.60050187e-16,   5.23459164e-17,  -1.46288559e-20],
        [ -7.97290849e-13,  -3.55806568e-13,   7.40591352e-14,
           1.04706969e-14,   6.39772344e-14,  -4.77401756e-17],
        [ -5.36487052e-13,  -1.11771270e-12,   2.77392448e-13,
           6.13562437e-14,  -1.56211613e-14,   1.59957714e-17],
        [  3.58373345e-15,   1.18063766e-14,  -7.06987671e-16,
           8.55804684e-17,   1.00442704e-15,   3.34435854e-15]],

       ..., 
       [[ -4.91079345e-11,   1.72281555e-11,  -1.28318556e-12,
           3.89101229e-15,  -1.72604502e-17,   1.56383751e-16],
        [  6.93975541e-14,   9.26208483e-14,   3.64869956e-14,
          -3.20548264e-15,  -8.32465020e-15,  -3.83397702e-15],
        [  1.32179944e-12,   7.62990417e-12,   2.76726290e-12,
           1.77231114e-14,   3.89597141e-17,   1.49963650e-15],
        [ -2.21004512e-12,   2.17418910e-12,   4.71875347e-13,
          -9.09998810e-14,   2.47919656e-16,   6.93321713e-15],
        [  6.71529357e-14,  -4.45715865e-13,  -1.09266788e-13,
           9.12888069e-15,  -5.20017269e-16,   6.47884856e-14],
        [  8.05411271e-10,   1.04391696e-12,  -8.14797234e-14,
          -4.20297616e-17,   3.24582855e-19,   2.10271687e-17]],

       [[  6.70466183e-09,  -5.52047236e-12,   1.33001108e-13,
          -5.23481894e-13,   2.09732788e-18,   6.63018322e-18],
        [ -4.70343917e-10,  -2.92447153e-11,   1.69998023e-12,
          -8.55488459e-12,  -1.04471335e-16,   6.45477088e-17],
        [  3.22362582e-13,  -2.49473432e-13,  -5.12059088e-14,
          -5.38667273e-14,   2.30104028e-14,  -3.24988144e-14],
        [  9.60757813e-13,   3.70567722e-13,   5.60136967e-14,
           3.92322208e-14,  -7.95224391e-14,  -9.40796989e-15],
        [  1.93679472e-10,   1.19665366e-10,  -5.54070491e-13,
          -2.70069195e-12,   9.07701280e-18,   3.23900899e-17],
        [  3.23607211e-12,   2.49765032e-11,   4.67315951e-12,
           2.80568826e-12,   1.24433243e-15,  -2.63167009e-16]],

       [[  1.39413074e-11,  -4.58172339e-12,   2.30816675e-12,
          -2.42038673e-13,   8.50326350e-14,   7.88450125e-16],
        [  1.13349199e-12,  -7.75423392e-13,   7.65114553e-14,
          -4.51902868e-13,  -4.27693691e-13,   1.90166811e-15],
        [ -5.01858382e-13,  -4.38593628e-10,   1.19311305e-15,
           1.32079527e-14,  -2.76569768e-15,   1.93875687e-17],
        [ -4.58371472e-13,   6.10489407e-13,   8.49358301e-15,
           6.01629975e-14,   5.83976986e-15,   4.75516659e-14],
        [ -1.74234522e-12,   6.41819650e-12,   9.65357470e-13,
           8.03876798e-13,  -2.07309917e-13,  -2.28755739e-15],
        [ -2.97939251e-11,   4.82968231e-12,   1.02635384e-12,
          -1.78606888e-13,   3.55977059e-14,  -1.56837042e-16]]])


'''

# %% guardo estos datos
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
